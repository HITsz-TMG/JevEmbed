"""Quota resume identity must depend on label counts, not YAML map order."""

import json
from pathlib import Path

import yaml

from jevembed.synthesis.config import load_config
from jevembed.synthesis.run import run


_CONFIG = Path(__file__).resolve().parents[2] / "configs/synthesis/choice-dinosaur.yaml"


def test_explicit_quota_key_order_does_not_change_resume_identity(tmp_path):
    config = load_config(_CONFIG)
    config["generation"].update(count=2, concurrency=1, verify_labels=False,
                                label_quotas={"Keep running": 1, "Jump": 1})
    first = tmp_path / "first.yaml"
    first.write_text(yaml.safe_dump(config, sort_keys=False))
    config["generation"]["label_quotas"] = {"Jump": 1, "Keep running": 1}
    reordered = tmp_path / "reordered.yaml"
    reordered.write_text(yaml.safe_dump(config, sort_keys=False))
    assert load_config(first) == load_config(reordered)

    class LabelAwareTeacher:
        def __init__(self):
            self.calls = 0

        def complete(self, messages):
            self.calls += 1
            target = json.loads(messages[1]["content"])["required_target_answer"]
            distance = 5 if target == {"choice": "Jump"} else 6
            return json.dumps({"state": {"distance_to_obstacle": distance},
                               "answer": target}), {}

    output = tmp_path / "job"
    teacher = LabelAwareTeacher()
    initial = run(first, output, teacher=teacher)
    assert teacher.calls == 2 and initial["quota_satisfied"] is True
    journal_before = (output / "journal.jsonl").read_bytes()

    class NoCallTeacher:
        def complete(self, messages):
            raise AssertionError("Completed resume must not call the teacher")

    resumed = run(reordered, output, resume=True, teacher=NoCallTeacher())
    assert resumed["quota_satisfied"] is True
    assert (output / "journal.jsonl").read_bytes() == journal_before


def test_all_positive_teacher_cannot_fill_negative_quota_or_invoke_its_verifier(tmp_path):
    config = load_config(_CONFIG)
    config["generation"].update(count=4, concurrency=1, max_attempts_per_sample=1,
                                verify_labels=True, label_quotas="balanced")

    class PositiveOnlyTeacher:
        generation_calls = 0
        verification_calls = 0

        def complete(self, messages):
            prompt = json.loads(messages[1]["content"])
            if "state" in prompt:
                self.verification_calls += 1
                return '{"answer":{"choice":"Jump"}}', {}
            self.generation_calls += 1
            distance = prompt["sample_number"]
            return json.dumps({"state": {"distance_to_obstacle": distance},
                               "answer": {"choice": "Jump"}}), {}

    teacher = PositiveOnlyTeacher()
    report = run(config, tmp_path / "job", teacher=teacher)
    assert teacher.generation_calls == 4
    assert teacher.verification_calls == 2
    assert report["complete"] is True and report["quota_satisfied"] is False
    assert report["rejections"]["target_label_mismatch"] == 2
    assert report["quota_labels"] == [
        {"answer": {"choice": "Jump"}, "requested": 2, "accepted": 2, "shortfall": 0},
        {"answer": {"choice": "Keep running"}, "requested": 2, "accepted": 0, "shortfall": 2},
    ]
