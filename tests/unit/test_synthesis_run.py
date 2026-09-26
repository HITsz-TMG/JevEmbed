import json
from pathlib import Path
from queue import Queue
import traceback

import httpx
import pytest

from jevembed.config import ModelConfig
from jevembed.errors import ValidationError
from jevembed.synthesis.config import load_config
from jevembed.synthesis.provider import ChatTeacher, TeacherError
from jevembed.synthesis.run import _messages, run
from jevembed.training.data import load_examples


ROOT = Path(__file__).resolve().parents[2]


def config(*, count=1, verify=True, attempts=1):
    result = load_config(ROOT / "configs/synthesis/choice-dinosaur.yaml")
    result["generation"].update(count=count, concurrency=1, verify_labels=verify,
                                max_attempts_per_sample=attempts, label_quotas=None)
    return result


def state(distance=5):
    return {"distance_to_obstacle": distance}


def generated(distance=5, choice="Jump"):
    return json.dumps({"state": state(distance), "answer": {"choice": choice}})


class FakeTeacher:
    def __init__(self, *responses):
        self.responses = Queue()
        for response in responses:
            self.responses.put(response)
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        response = self.responses.get_nowait()
        return response, {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def test_generation_verification_training_parity_and_completed_resume(tmp_path):
    teacher = FakeTeacher(generated(), '{"answer":{"choice":"Jump"}}')
    report = run(config(), tmp_path / "job", teacher=teacher)
    assert report["accepted"] == 1
    assert report["teacher_usage"]["calls"] == 2
    assert report["teacher_usage"]["total_tokens"] == 20
    assert "answer" not in json.loads(teacher.messages[1][1]["content"])
    record = json.loads((tmp_path / "job/data.jsonl").read_text())
    assert record["request"]["state"] == state()
    assert list(record["request"]["questions"]["action"]["criteria"]) == [
        "Jump", "Keep running"]
    assert record["answers"]["action"] == {"choice": "Jump"}
    examples = load_examples(tmp_path / "job/data.jsonl",
                             ModelConfig.load(ROOT / "configs/kalm-embedding-v2.5.yaml"))
    assert len(examples) == 1 and examples[0].target == [1.0, 0.0]
    assert run(config(), tmp_path / "job", resume=True)["accepted"] == 1


def test_invalid_and_disagreeing_answers_retry_then_shortfall(tmp_path):
    teacher = FakeTeacher(
        '{"state":{"bad":true},"answer":{"choice":"Jump"}}',
        generated(), '{"answer":{"choice":"Keep running"}}')
    report = run(config(attempts=2), tmp_path / "job", teacher=teacher)
    assert report["accepted"] == 0 and report["shortfall"] == 1
    assert report["rejections"] == {"invalid_generation": 1, "label_disagreement": 1}
    assert (tmp_path / "job/data.jsonl").read_text() == ""


def test_duplicate_and_conflicting_label_quarantine(tmp_path):
    teacher = FakeTeacher(generated(), generated(), generated(choice="Keep running"),
                          generated(distance=6, choice="Keep running"))
    report = run(config(count=4, verify=False), tmp_path / "job", teacher=teacher)
    assert report["accepted"] == 1
    assert report["rejections"]["duplicate"] == 1
    assert report["rejections"]["conflicting_duplicate"] == 2
    assert json.loads((tmp_path / "job/data.jsonl").read_text())["request"]["state"] == state(6)


def test_validate_only_and_resume_identity_and_torn_tail(tmp_path):
    output = tmp_path / "job"
    assert run(config(), output, validate_only=True)["valid"]
    assert not output.exists()
    run(config(verify=False), output, teacher=FakeTeacher(generated()))
    with pytest.raises(ValidationError, match="configuration"):
        run(config(count=2, verify=False), output, resume=True)
    with (output / "journal.jsonl").open("ab") as handle:
        handle.write(b'{"slot":')
    assert run(config(verify=False), output, resume=True)["accepted"] == 1
    assert (output / "journal.jsonl").read_bytes().endswith(b"\n")
    (output / "journal.jsonl").unlink()
    with pytest.raises(ValidationError, match="journal is missing"):
        run(config(verify=False), output, resume=True)


def test_provider_retries_counts_and_rejects_redirect_truncation(monkeypatch):
    statuses = iter([429, 200])
    def handler(request):
        status = next(statuses)
        return httpx.Response(status, json={"choices": [{"message": {"content": "{}"},
                                                            "finish_reason": "stop"}]})
    settings = config()["teacher"]
    settings["max_retries"] = 1
    monkeypatch.setattr("jevembed.synthesis.provider.time.sleep", lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        teacher = ChatTeacher(settings, client=client)
        assert teacher.complete([{"role": "user", "content": "hello"}])[0] == "{}"
        assert teacher.last_http_requests() == 2
    for status, payload, match in [
        (302, {}, "HTTP 302"),
        (200, {"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}]},
         "did not finish")]:
        with httpx.Client(transport=httpx.MockTransport(
                lambda request: httpx.Response(status, json=payload))) as client:
            teacher = ChatTeacher(settings, client=client)
            with pytest.raises(TeacherError, match=match):
                teacher.complete([{"role": "user", "content": "hello"}])


@pytest.mark.parametrize("payload", [
    [], "not-an-object", {}, {"choices": []}, {"choices": "not-a-list"},
    {"choices": ["PRIVATE_BODY_MARKER"]}, {"choices": [{"message": "not-an-object"}]},
    {"choices": [{"message": {}}]}, {"choices": [{"message": {"content": 123}}]},
    {"choices": [{"finish_reason": ["length"], "message": {"content": "{}"}}]},
])
def test_provider_rejects_malformed_response_containers_without_leaking_body(payload):
    settings = config()["teacher"]
    with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload))) as client:
        teacher = ChatTeacher(settings, client=client)
        with pytest.raises(TeacherError, match="Malformed teacher response") as error:
            teacher.complete([{"role": "user", "content": "hello"}])
        assert teacher.last_http_requests() == 1
        assert "PRIVATE_BODY_MARKER" not in "".join(traceback.format_exception(error.value))


def test_malformed_response_is_nonfatal_and_next_sample_attempt_can_succeed(tmp_path):
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        payload = {"choices": ["bad"]} if calls == 1 else {
            "choices": [{"message": {"content": generated()}, "finish_reason": "stop"}]}
        return httpx.Response(200, json=payload)
    cfg = config(verify=False, attempts=2)
    cfg["teacher"]["max_retries"] = 0
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = run(cfg, tmp_path / "job", teacher=ChatTeacher(cfg["teacher"], client=client))
    assert calls == 2
    assert report["accepted"] == 1
    assert report["rejections"]["teacher_error"] == 1
    assert report["teacher_usage"]["http_requests"] == 2


def test_http_attempts_are_reported_and_permanent_error_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr("jevembed.synthesis.provider.time.sleep", lambda _: None)
    attempts = 0
    def transient(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": generated()},
                                                       "finish_reason": "stop"}], "usage": {}})
    cfg = config(verify=False)
    cfg["teacher"]["max_retries"] = 1
    with httpx.Client(transport=httpx.MockTransport(transient)) as client:
        report = run(cfg, tmp_path / "transient", teacher=ChatTeacher(cfg["teacher"], client=client))
    assert report["accepted"] == 1
    assert report["teacher_usage"]["http_requests"] == 2
    assert report["teacher_usage"]["calls_without_usage"] == 1

    calls = 0
    def permanent(request):
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="super-secret-server-body")
    with httpx.Client(transport=httpx.MockTransport(permanent)) as client:
        with pytest.raises(TeacherError, match="HTTP 401") as error:
            run(config(verify=False, attempts=3), tmp_path / "permanent",
                teacher=ChatTeacher(cfg["teacher"], client=client))
    assert calls == 1 and "super-secret" not in str(error.value)


def test_resume_rejects_corrupt_completed_journal(tmp_path):
    output = tmp_path / "job"
    cfg = config(verify=False)
    run(cfg, output, teacher=FakeTeacher(generated()))
    journal = output / "journal.jsonl"
    event = json.loads(journal.read_text())
    event["case_type"] = "not-allocated"
    journal.write_text(json.dumps(event) + "\n")
    with pytest.raises(ValidationError, match="Invalid synthesis journal"):
        run(cfg, output, resume=True)


@pytest.mark.parametrize("filename,answer_key,answer_type", [
    ("choice-dinosaur.yaml", "choice", "string"),
    ("score-risk.yaml", "level", "integer"),
    ("noul-human-escalation.yaml", "noul", "boolean"),
    ("noul-repeat-contact.yaml", "noul", "boolean"),
])
def test_teacher_prompts_require_nested_hard_answer(filename, answer_key, answer_type):
    cfg = load_config(ROOT / "configs/synthesis" / filename)
    for verification in (False, True):
        messages = _messages(cfg, "boundary", 0, state={} if verification else None)
        prompt = json.loads(messages[1]["content"])
        schema = prompt["required_response_schema"]
        assert schema["properties"]["answer"]["properties"][answer_key]["type"] == answer_type
        assert schema["properties"]["answer"]["required"] == [answer_key]
        assert '"answer":{' in prompt["response_shape_replace_placeholders"]
        assert "never a" in messages[0]["content"] or "never a string" in messages[0]["content"]
        if verification:
            assert "answer" not in prompt and "state" in prompt
            assert schema["required"] == ["answer"]
        else:
            assert schema["required"] == ["state", "answer"]


def test_validator_exception_traceback_hides_callback_details(tmp_path):
    def bad_validator(state, answer, question):
        raise RuntimeError("PRIVATE_CALLBACK_MARKER")

    with pytest.raises(ValidationError) as error:
        run(config(verify=False), tmp_path / "job", teacher=FakeTeacher(generated()),
            sample_validator=bad_validator, validator_id="trusted-v1")
    rendered = "".join(traceback.format_exception(error.value))
    assert "PRIVATE_CALLBACK_MARKER" not in rendered
    assert "Trusted sample validator failed" in rendered
