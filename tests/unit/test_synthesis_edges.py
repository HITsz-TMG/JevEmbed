"""Failure and concurrency boundaries for durable synthesis jobs."""

import json
import importlib
from pathlib import Path
from threading import Barrier, Lock

import pytest

from jevembed.errors import ValidationError
from jevembed.synthesis.config import load_config
from jevembed.synthesis.provider import TeacherError
from jevembed.synthesis.run import run


_CONFIG = Path(__file__).resolve().parents[2] / "configs/synthesis/choice-dinosaur.yaml"


def _config(*, count=1, concurrency=1, attempts=1):
    config = load_config(_CONFIG)
    config["generation"].update(count=count, concurrency=concurrency,
                                max_attempts_per_sample=attempts, verify_labels=False,
                                label_quotas=None)
    return config


def _generated(distance, *, choice="Jump"):
    return json.dumps({"state": {"distance_to_obstacle": distance},
                       "answer": {"choice": choice}})


def _dinosaur_reference(state, answer, question):
    expected = "Jump" if state["distance_to_obstacle"] <= 5 else "Keep running"
    return answer["choice"] == expected


def _doom_config():
    """An arithmetic fixture kept only to test the trusted reference hook."""
    config = _config(attempts=2)
    config["task"].update(
        id="arithmetic-test", domain="Synthetic arithmetic regression",
        question={"type": "choice", "instructions": (
                      "Compute error_deg = bearing_deg - 0.25 * lateral_units_per_tic * "
                      "action_window_tics. Choose Strafe left if error_deg > 2, Strafe right "
                      "if error_deg < -2, otherwise Hold lateral movement."),
                  "criteria": {"Strafe left": None, "Strafe right": None,
                               "Hold lateral movement": None}},
        state_description="A test snapshot with bearing in degrees, lateral units per tic, and a tic window.",
        state_schema={"type": "object", "additionalProperties": False,
                      "required": ["visible_enemy_count", "nearest_visible_enemy", "movement",
                                   "action_window_tics"],
                      "properties": {
                          "visible_enemy_count": {"type": "integer"},
                          "nearest_visible_enemy": {"type": "object", "required": ["bearing_deg"],
                                                    "properties": {"bearing_deg": {"type": "number"}}},
                          "movement": {"type": "object", "required": ["lateral_units_per_tic"],
                                       "properties": {"lateral_units_per_tic": {"type": "number"}}},
                          "action_window_tics": {"type": "integer"}}})
    return config


def _doom_generated(bearing, lateral, choice):
    state = {"visible_enemy_count": 1, "nearest_visible_enemy": {"bearing_deg": bearing},
             "movement": {"lateral_units_per_tic": lateral}, "action_window_tics": 2}
    return json.dumps({"state": state, "answer": {"choice": choice}})


def _doom_reference(state, answer, question):
    if state["visible_enemy_count"] == 0:
        expected = "Hold lateral movement"
    else:
        error = (state["nearest_visible_enemy"]["bearing_deg"]
                 - 0.25 * state["movement"]["lateral_units_per_tic"] * state["action_window_tics"])
        expected = "Strafe left" if error > 2 else "Strafe right" if error < -2 else "Hold lateral movement"
    return answer["choice"] == expected


class _SequenceTeacher:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def test_partial_resume_calls_teacher_only_for_uncommitted_slot(tmp_path):
    output = tmp_path / "job"
    config = _config(count=2)
    first = _SequenceTeacher((_generated(5), {}), TeacherError("HTTP 401", fatal=True))
    with pytest.raises(TeacherError, match="HTTP 401"):
        run(config, output, teacher=first)
    assert first.calls == 2
    assert len((output / "journal.jsonl").read_text().splitlines()) == 1
    assert json.loads((output / "report.json").read_text())["processed"] == 1

    remaining = _SequenceTeacher((_generated(6, choice="Keep running"), {}))
    report = run(config, output, resume=True, teacher=remaining)
    assert remaining.calls == 1
    assert report["processed"] == 2 and report["accepted"] == 2
    assert report["teacher_usage"]["calls_without_usage"] == 2
    assert report["teacher_usage"]["total_tokens"] == 0
    assert len((output / "journal.jsonl").read_text().splitlines()) == 2
    assert len((output / "data.jsonl").read_text().splitlines()) == 2


def test_validate_only_token_budget_includes_http_retries():
    config = _config(count=2)
    config["teacher"].update(max_retries=2, max_tokens=128)
    budget = run(config, validate_only=True)
    assert budget["maximum_teacher_calls"] == 2
    assert budget["maximum_http_requests"] == 6
    assert budget["maximum_completion_tokens_requested"] == 6 * 128


def test_concurrency_is_real_and_never_exceeds_configured_limit(tmp_path):
    config = _config(count=4, concurrency=2)

    class ParallelTeacher:
        def __init__(self):
            self.barrier = Barrier(2, timeout=3)
            self.lock = Lock()
            self.calls = 0
            self.active = 0
            self.peak = 0

        def complete(self, messages):
            with self.lock:
                index = self.calls
                self.calls += 1
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                if index < 2:
                    self.barrier.wait()
                distance = index + 5
                return _generated(distance, choice="Jump" if distance <= 5 else "Keep running"), {}
            finally:
                with self.lock:
                    self.active -= 1

    teacher = ParallelTeacher()
    report = run(config, tmp_path / "job", teacher=teacher)
    assert teacher.calls == 4
    assert teacher.peak == 2
    assert report["accepted"] == 4


def test_trusted_reference_rejects_wrong_doom_arithmetic_then_accepts_retry(tmp_path):
    config = _doom_config()
    teacher = _SequenceTeacher((_doom_generated(5, 20, "Strafe left"), {}),
                               (_doom_generated(5, 20, "Strafe right"), {}))
    report = run(config, tmp_path / "job", teacher=teacher,
                 sample_validator=_doom_reference, validator_id="doom-arithmetic-v1")
    assert teacher.calls == 2
    assert report["accepted"] == 1
    assert report["rejections"] == {"reference_rejected": 1}
    assert json.loads((tmp_path / "job/data.jsonl").read_text())["answers"]["action"] == {
        "choice": "Strafe right"}


def test_trusted_reference_receives_isolated_copies(tmp_path):
    config = _config()

    def mutate_inputs(state, answer, question):
        state["distance_to_obstacle"] = 100
        answer["choice"] = "Keep running"
        question["criteria"].clear()
        return True

    report = run(config, tmp_path / "job", teacher=_SequenceTeacher((_generated(5), {})),
                 sample_validator=mutate_inputs, validator_id="mutating-test-v1")
    record = json.loads((tmp_path / "job/data.jsonl").read_text())
    assert report["accepted"] == 1
    assert record["request"]["state"]["distance_to_obstacle"] == 5
    assert record["answers"]["action"] == {"choice": "Jump"}
    assert len(record["request"]["questions"]["action"]["criteria"]) == 2
    assert len(config["task"]["question"]["criteria"]) == 2


@pytest.mark.parametrize("behavior", ["one", "none", "exception"])
def test_trusted_reference_failure_aborts_without_exposing_exception(tmp_path, behavior):
    def bad_reference(state, answer, question):
        if behavior == "exception":
            raise RuntimeError("private validator detail")
        return 1 if behavior == "one" else None

    output = tmp_path / behavior
    with pytest.raises(ValidationError) as error:
        run(_config(), output, teacher=_SequenceTeacher((_generated(5), {})),
            sample_validator=bad_reference, validator_id="bad-test-v1")
    assert "private validator detail" not in str(error.value)
    assert json.loads((output / "report.json").read_text())["processed"] == 0
    assert not (output / "journal.jsonl").exists()


def test_resume_rejects_changed_or_removed_reference_identity(tmp_path):
    output = tmp_path / "job"
    config = _config()
    run(config, output, teacher=_SequenceTeacher((_generated(5), {})),
        sample_validator=_dinosaur_reference, validator_id="dinosaur-v1")
    for options in ({}, {"sample_validator": _dinosaur_reference, "validator_id": "dinosaur-v2"}):
        with pytest.raises(ValidationError, match="configuration or format differs"):
            run(config, output, resume=True, **options)
    assert run(config, output, resume=True,
               sample_validator=_dinosaur_reference, validator_id="dinosaur-v1")["accepted"] == 1


def test_resume_rejects_changed_prompt_logic(tmp_path, monkeypatch):
    output = tmp_path / "job"
    config = _config()
    run(config, output, teacher=_SequenceTeacher((_generated(5), {})))
    module = importlib.import_module("jevembed.synthesis.run")
    original = module._messages

    def revised_messages(*args, **kwargs):
        messages = original(*args, **kwargs)
        messages[0]["content"] += " Revised instructions."
        return messages

    monkeypatch.setattr(module, "_messages", revised_messages)
    with pytest.raises(ValidationError, match="configuration or format differs"):
        run(config, output, resume=True)


def test_resume_rechecks_accepted_journal_rows_with_reference(tmp_path):
    output = tmp_path / "job"
    config = _config()
    run(config, output, teacher=_SequenceTeacher((_generated(5), {})),
        sample_validator=_dinosaur_reference, validator_id="dinosaur-v1")
    original = (output / "data.jsonl").read_bytes()
    with pytest.raises(ValidationError, match="Invalid synthesis journal"):
        run(config, output, resume=True, sample_validator=lambda *args: False,
            validator_id="dinosaur-v1")
    assert (output / "data.jsonl").read_bytes() == original
