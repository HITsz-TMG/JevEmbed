import json

import pytest

from jevembed.benchmark import load_benchmark, run_benchmark
from jevembed.benchmark.format import BENCHMARK_SCHEMA, BenchmarkTarget
from jevembed.benchmark.metrics import MetricAccumulator
from jevembed.config import ModelConfig
from jevembed.errors import ValidationError


def _request(kind, criteria):
    return {"state": "state", "questions": {"q": {
        "type": kind, "instructions": "instruction", "criteria": criteria,
    }}}


def _write_benchmark(tmp_path, rows, *, data="cases.jsonl"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": BENCHMARK_SCHEMA,
        "name": "test-decisions",
        "version": "1.0",
        "split": "test",
        "data": data,
    }
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / data).write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return manifest_path


def _rows():
    return [
        {"id": "choice", "request": _request("choice", {"a": "alpha", "b": "beta"}),
         "answers": {"q": {"choice": "a"}}},
        {"id": "score", "request": _request("score", ["low", "medium", "high"]),
         "answers": {"q": {"level": 2}}},
        {"id": "noul", "request": _request("noul", {"true": "yes", "false": "no"}),
         "answers": {"q": {"noul": True}}},
    ]


def test_loads_choice_score_noul_and_soft_targets(tmp_path):
    rows = _rows()
    rows[0]["answers"] = {"q": {"probabilities": {"a": 0.25, "b": 0.75}}}
    rows[1]["answers"] = {"q": {"score": 1.5}}
    rows[2]["answers"] = {"q": {"noul": 0.8}}
    rows[0]["metadata"] = {"domain": "test"}
    spec = load_benchmark(_write_benchmark(tmp_path, rows))

    assert spec.name == "test-decisions"
    assert spec.question_count == 3
    assert len(spec.manifest_sha256) == len(spec.data_sha256) == 64
    assert spec.cases[0].targets["q"].values == (0.25, 0.75)
    assert spec.cases[1].targets["q"].mode == "score"
    assert spec.cases[2].targets["q"].values == (0.8,)
    assert spec.cases[0].metadata == {"domain": "test"}


def test_rejects_model_duplicate_ids_and_escaping_data_path(tmp_path):
    rows = _rows()
    rows[0]["request"]["model"] = "fixed"
    with pytest.raises(ValidationError, match="must omit model"):
        load_benchmark(_write_benchmark(tmp_path, rows))

    rows = _rows()
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValidationError, match="unique nonempty"):
        load_benchmark(_write_benchmark(tmp_path, rows))

    outside = tmp_path.parent / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": BENCHMARK_SCHEMA, "name": "escape", "version": "1",
        "split": "test", "data": "../outside.jsonl",
    }
    (tmp_path / "benchmark.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValidationError, match="stay inside"):
        load_benchmark(tmp_path / "benchmark.json")


def test_data_accepts_multiple_jsonl_shards_and_checks_cross_file_ids(tmp_path):
    input_dir = tmp_path / "shards"
    input_dir.mkdir()
    first, second = _rows()[:2]
    (input_dir / "cases-00000.jsonl").write_text(json.dumps(first) + "\n", encoding="utf-8")
    (input_dir / "cases-00001.jsonl").write_text(json.dumps(second) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": BENCHMARK_SCHEMA, "name": "sharded", "version": "1",
        "split": "test", "data": ["cases-00000.jsonl", "cases-00001.jsonl"],
    }
    manifest_path = input_dir / "benchmark.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec = load_benchmark(manifest_path)
    assert spec.data_files == ("cases-00000.jsonl", "cases-00001.jsonl")
    assert spec.data_file is None
    assert [case.record_id for case in spec.cases] == ["choice", "score"]

    duplicate = dict(second, id=first["id"])
    (input_dir / "cases-00001.jsonl").write_text(json.dumps(duplicate) + "\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="unique nonempty"):
        load_benchmark(manifest_path)


def test_invalid_utf8_is_reported_as_validation_error(tmp_path):
    manifest = tmp_path / "benchmark.json"
    manifest.write_bytes(b"\xff")
    with pytest.raises(ValidationError, match="benchmark.json"):
        load_benchmark(manifest)


def test_soft_target_metrics_keep_their_own_denominators():
    metrics = MetricAccumulator()
    metrics.add(BenchmarkTarget("choice", ("a", "b"), "distribution", (0.25, 0.75)),
                {"type": "choice", "choice": "a", "probabilities": {"a": 0.5, "b": 0.5}}, "choice")
    metrics.add(BenchmarkTarget("score", ("0", "1", "2"), "score", (1.5,)),
                {"type": "score", "score": 1.5,
                 "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5}}, "score")
    metrics.add(BenchmarkTarget("noul", (), "noul", (0.8,)),
                {"type": "noul", "noul": 0.6}, "noul")
    result = metrics.result()

    assert result["overall_hard_accuracy"] is None
    assert result["overall_hard_accuracy_n"] == 0
    assert result["score_mae"] == pytest.approx(0.0)
    assert result["score_mae_n"] == 1
    assert result["noul_mae"] == pytest.approx(0.2)
    assert result["target_distribution_brier"] == pytest.approx((0.125 + 0.08) / 2)
    assert result["target_distribution_brier_n"] == 2
    assert result["target_distribution_tvd"] == pytest.approx((0.25 + 0.2) / 2)


class _PerfectClient:
    def __init__(self, config):
        self.config = config

    def evaluate(self, request):
        answers = {}
        for question_id, question in request["questions"].items():
            if question["type"] == "choice":
                labels = list(question["criteria"])
                answers[question_id] = {
                    "type": "choice", "choice": labels[0], "confidence": 1.0,
                    "probabilities": {label: float(index == 0) for index, label in enumerate(labels)},
                }
            elif question["type"] == "score":
                labels = [str(index) for index in range(len(question["criteria"]))]
                last = len(labels) - 1
                answers[question_id] = {
                    "type": "score", "score": float(last), "confidence": 1.0,
                    "probabilities": {label: float(index == last) for index, label in enumerate(labels)},
                    "legend": dict(zip(labels, question["criteria"])),
                }
            else:
                answers[question_id] = {"type": "noul", "noul": 1.0}
        return {"model": self.config.model_id, "answers": answers,
                "usage": {"input_tokens": 7, "output_tokens": 0}}


def test_run_writes_organized_results_and_requires_explicit_overwrite(tmp_path):
    spec_path = _write_benchmark(tmp_path / "input", _rows())
    config = ModelConfig(model_id="perfect/model", backend="custom")
    output = tmp_path / "output"
    summary = run_benchmark(spec_path, [config], output, save_predictions=True,
                            client_factory=_PerfectClient)

    root = output / "test-decisions"
    result_path = root / "perfect-model" / "results.json"
    predictions_path = root / "perfect-model" / "predictions.jsonl"
    assert summary["models"][0]["main_score"] == 1.0
    assert summary["evaluated_cases"] == 3
    assert result_path.is_file() and predictions_path.is_file()
    result = json.loads(result_path.read_text())
    assert result["metrics"]["overall_hard_accuracy"] == 1.0
    assert result["metrics"]["overall_hard_accuracy_n"] == 3
    assert result["run"]["evaluated_questions"] == 3
    assert result["usage"] == {"input_tokens": 21, "output_tokens": 0}
    assert len(predictions_path.read_text().splitlines()) == 3
    assert str(tmp_path) not in result_path.read_text()

    with pytest.raises(ValidationError, match="already exist"):
        run_benchmark(spec_path, [config], output, client_factory=_PerfectClient)
    limited = run_benchmark(spec_path, [config], output, limit=1, overwrite=True,
                            client_factory=_PerfectClient)
    assert limited["limited"] is True
    assert limited["evaluated_cases"] == 1
    assert not predictions_path.exists()
