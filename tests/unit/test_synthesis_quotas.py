"""Quota scheduling and enforcement across teacher, journal, and export."""

import json
from pathlib import Path

import pytest

from jevembed.errors import ValidationError
from jevembed.synthesis.config import load_config, validate_config
from jevembed.synthesis.quotas import quota_answers
from jevembed.synthesis.run import _allocation, _messages, run
from jevembed.synthesis import run as run_module


ROOT = Path(__file__).resolve().parents[2]


def choice_config(*, count=4, verify=True, attempts=1):
    config = load_config(ROOT / "configs/synthesis/choice-dinosaur.yaml")
    config["generation"].update(count=count, concurrency=1, label_quotas="balanced",
                                 verify_labels=verify, max_attempts_per_sample=attempts)
    return validate_config(config)


class QuotaTeacher:
    def __init__(self, *, mismatch_first=False, duplicate_jump=False):
        self.mismatch_first = mismatch_first
        self.duplicate_jump = duplicate_jump
        self.calls = []

    def complete(self, messages):
        prompt = json.loads(messages[1]["content"])
        self.calls.append(prompt)
        if "state" in prompt:
            choice = "Jump" if prompt["state"]["distance_to_obstacle"] <= 5 else "Keep running"
            payload = {"answer": {"choice": choice}}
        else:
            target = prompt["required_target_answer"]["choice"]
            if self.mismatch_first and prompt["attempt_number"] == 1:
                target = "Jump" if target == "Keep running" else "Keep running"
            slot = prompt["sample_number"]
            distance = (0 if self.duplicate_jump else slot % 6) if target == "Jump" else 6 + slot
            payload = {"state": {"distance_to_obstacle": distance},
                       "answer": {"choice": target}}
        return json.dumps(payload), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


@pytest.mark.parametrize("filename", ["choice-dinosaur.yaml", "score-risk.yaml",
                                       "noul-human-escalation.yaml", "noul-repeat-contact.yaml"])
def test_balanced_targets_and_case_types_are_distributed_per_label(filename):
    config = load_config(ROOT / "configs/synthesis" / filename)
    allocation = _allocation(config)
    assert allocation == _allocation(config)
    for answer, target_count in quota_answers(config):
        cases = [case for case, target in allocation.values() if target == answer]
        assert len(cases) == target_count
        assert len(set(cases)) == min(len(config["generation"]["case_types"]), target_count)
    assert len(allocation) == config["generation"]["count"]


def test_targeted_generation_blind_verification_and_clean_export(tmp_path):
    teacher = QuotaTeacher()
    report = run(choice_config(), tmp_path / "job", teacher=teacher)
    assert report["complete"] and report["quota_satisfied"]
    assert [(row["requested"], row["accepted"], row["shortfall"])
            for row in report["quota_labels"]] == [(2, 2, 0), (2, 2, 0)]
    generated = [call for call in teacher.calls if "state" not in call]
    verified = [call for call in teacher.calls if "state" in call]
    assert len(generated) == len(verified) == 4
    assert all("required_target_answer" in call and call["attempt_number"] == 1
               for call in generated)
    assert all("required_target_answer" not in call and "attempt_number" not in call
               for call in verified)
    records = [json.loads(line) for line in (tmp_path / "job/data.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert all(set(row) == {"id", "group", "request", "answers"} for row in records)
    assert all("required_target_answer" not in json.dumps(row) for row in records)
    assert run(choice_config(), tmp_path / "job", resume=True)["quota_satisfied"]


def test_wrong_target_retries_without_verifier_and_changes_attempt_prompt(tmp_path):
    teacher = QuotaTeacher(mismatch_first=True)
    report = run(choice_config(count=2, attempts=2), tmp_path / "job", teacher=teacher)
    assert report["quota_satisfied"]
    assert report["rejections"]["target_label_mismatch"] == 2
    generated = [call for call in teacher.calls if "state" not in call]
    verified = [call for call in teacher.calls if "state" in call]
    assert len(generated) == 4 and len(verified) == 2
    assert sorted(call["attempt_number"] for call in generated) == [1, 1, 2, 2]


def test_exact_dedup_reports_quota_deficit_after_all_slots_complete(tmp_path):
    report = run(choice_config(), tmp_path / "job", teacher=QuotaTeacher(duplicate_jump=True))
    assert report["complete"] is True
    assert report["quota_satisfied"] is False
    assert report["accepted"] == 3 and report["shortfall"] == 1
    assert report["rejections"]["duplicate"] == 1
    assert report["quota_labels"][0] == {
        "answer": {"choice": "Jump"}, "requested": 2, "accepted": 1, "shortfall": 1}
    assert report["quota_labels"][1]["shortfall"] == 0


def test_resume_rejects_journal_answer_tampered_against_target(tmp_path):
    config = choice_config(count=2, verify=False)
    output = tmp_path / "job"
    run(config, output, teacher=QuotaTeacher())
    journal = output / "journal.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    wrong = events[0]["answer"]["choice"]
    events[0]["answer"]["choice"] = "Jump" if wrong == "Keep running" else "Keep running"
    journal.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(ValidationError, match="Invalid synthesis journal"):
        run(config, output, resume=True)


def test_validate_only_shows_resolved_quota_without_output(tmp_path):
    output = tmp_path / "job"
    result = run(choice_config(), output, validate_only=True)
    assert result["quota_labels"] == [
        {"answer": {"choice": "Jump"}, "requested": 2},
        {"answer": {"choice": "Keep running"}, "requested": 2}]
    assert not output.exists()


def test_prompt_hash_uses_bounded_representative_labels(monkeypatch):
    config = load_config(ROOT / "configs/synthesis/choice-dinosaur.yaml")
    config["task"]["question"]["criteria"] = {f"candidate-{i}": None for i in range(300)}
    config["generation"].update(count=300, label_quotas="balanced")
    config = validate_config(config)
    calls = 0
    original = run_module._messages
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(run_module, "_messages", counted)
    assert len(run_module._prompt_hash(config)) == 64
    assert calls <= 7


def test_target_meaning_is_explicit_only_in_generation_prompt():
    human = load_config(ROOT / "configs/synthesis/noul-human-escalation.yaml")
    negative = _messages(human, "hard_negative", 0, target_answer={"noul": False})
    positive = _messages(human, "standard", 1, target_answer={"noul": True})
    verifier = _messages(human, "hard_negative", 0, state="Please send me the support instructions.")
    assert "correct answer must be false" in negative[0]["content"]
    assert "clear negative example" in negative[0]["content"]
    assert "correct answer must be true" in positive[0]["content"]
    assert "clear positive example" in positive[0]["content"]
    assert "correct answer must be" not in verifier[0]["content"]
    assert "required_target_answer" not in json.loads(verifier[1]["content"])

    choice = load_config(ROOT / "configs/synthesis/choice-dinosaur.yaml")
    choice_prompt = _messages(choice, "standard", 0, target_answer={"choice": "Jump"})
    assert 'exact candidate key "Jump"' in choice_prompt[0]["content"]
    score = load_config(ROOT / "configs/synthesis/score-risk.yaml")
    score_prompt = _messages(score, "standard", 0, target_answer={"level": 2})
    assert "zero-based level 2" in score_prompt[0]["content"]
