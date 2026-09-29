"""Run one benchmark against one or more JevEmbed model configurations."""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

from .. import JevEmbed, __version__
from ..errors import ValidationError
from .format import RESULT_SCHEMA, BenchmarkSpec, load_benchmark
from .metrics import MetricAccumulator


SUMMARY_SCHEMA = "jevembed-benchmark-summary-v1"


def _slug(value):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not slug:
        raise ValidationError("Model ID cannot form a result directory name")
    return slug


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _benchmark_record(spec):
    return {"name": spec.name, "version": spec.version, "split": spec.split,
            "description": spec.description, "cases": len(spec.cases),
            "data": spec.data_files[0] if len(spec.data_files) == 1 else list(spec.data_files),
            "questions": spec.question_count, "manifest_sha256": spec.manifest_sha256,
            "data_sha256": spec.data_sha256}


def _run_model(spec, config, output, *, save_predictions, limit, client_factory):
    started_wall = datetime.now(timezone.utc)
    started = time.perf_counter()
    client = client_factory(config) if client_factory else JevEmbed(config=config)
    cases = spec.cases[:limit] if limit else spec.cases
    accumulator = MetricAccumulator()
    input_tokens = output_tokens = 0
    prediction_path = output / "predictions.jsonl"
    prediction_temporary = prediction_path.with_suffix(".jsonl.tmp")
    prediction_file = None
    if save_predictions:
        output.mkdir(parents=True, exist_ok=True)
        prediction_file = prediction_temporary.open("w", encoding="utf-8")
    try:
        for case in cases:
            request = deepcopy(case.request)
            request["model"] = config.model_id
            response = client.evaluate(request)
            if not isinstance(response, dict) or response.get("model") != config.model_id:
                raise ValidationError(f"Model identity does not match benchmark case {case.record_id}")
            answers = response.get("answers")
            if not isinstance(answers, dict) or set(answers) != set(case.targets):
                raise ValidationError(f"Model answers do not match benchmark case {case.record_id}")
            for question_id, target in case.targets.items():
                accumulator.add(target, answers[question_id],
                                f"case {case.record_id}, question {question_id}")
            usage = response.get("usage", {})
            for key in ("input_tokens", "output_tokens"):
                value = usage.get(key)
                if type(value) is not int or value < 0:
                    raise ValidationError(f"Invalid {key} for benchmark case {case.record_id}")
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            if prediction_file:
                prediction_file.write(json.dumps(
                    {"id": case.record_id, "group": case.group, "model": response.get("model"),
                     "answers": answers, "usage": usage},
                    ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    except Exception:
        if prediction_file:
            prediction_file.close()
        if prediction_temporary.exists():
            prediction_temporary.unlink()
        raise
    if prediction_file:
        prediction_file.close()
        prediction_temporary.replace(prediction_path)
    metrics = accumulator.result()
    if any(value is not None and isinstance(value, float) and not math.isfinite(value)
           for value in metrics.values()):
        raise ValidationError("Benchmark produced a nonfinite metric")
    finished_wall = datetime.now(timezone.utc)
    result = {
        "schema_version": RESULT_SCHEMA,
        "benchmark": _benchmark_record(spec),
        "model": {"id": config.model_id, "config_sha256": config.fingerprint(),
                  "revision": config.revision, "adapter_revision": config.adapter_revision,
                  "projection_revision": config.projection_revision},
        "run": {"jevembed_version": __version__, "started_at": started_wall.isoformat(),
                "finished_at": finished_wall.isoformat(),
                "duration_seconds": time.perf_counter() - started,
                "limited": bool(limit), "case_limit": limit or None,
                "evaluated_cases": len(cases),
                "evaluated_questions": sum(len(case.targets) for case in cases)},
        "task_counts": accumulator.task_counts,
        "metrics": metrics,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "predictions_file": "predictions.jsonl" if save_predictions else None,
        "configuration": {"prompts": asdict(config.prompts), "scoring": asdict(config.scoring),
                          "max_input_tokens": config.max_input_tokens,
                          "overflow_policy": config.overflow_policy,
                          "dtype": config.dtype},
    }
    _write_json(output / "results.json", result)
    if not save_predictions and prediction_path.exists():
        prediction_path.unlink()
    return result


def run_benchmark(benchmark, configs, output_dir, *, save_predictions=False, limit=0,
                  overwrite=False, client_factory=None):
    """Evaluate configs and write one result directory per model plus summary.json."""
    spec = benchmark if isinstance(benchmark, BenchmarkSpec) else load_benchmark(benchmark)
    configs = list(configs)
    if not configs:
        raise ValidationError("At least one model configuration is required")
    if type(limit) is not int or limit < 0:
        raise ValidationError("limit must be a nonnegative integer")
    if type(overwrite) is not bool:
        raise ValidationError("overwrite must be an explicit boolean")
    root = Path(output_dir) / spec.name
    slugs = [_slug(config.model_id) for config in configs]
    if len(set(slugs)) != len(slugs):
        raise ValidationError("Model IDs map to duplicate result directory names")
    existing = [root / "summary.json", *(root / slug / "results.json" for slug in slugs)]
    if not overwrite and any(path.exists() for path in existing):
        raise ValidationError("Benchmark results already exist; pass overwrite=True to replace them")
    models = []
    for config, slug in zip(configs, slugs):
        result = _run_model(spec, config, root / slug, save_predictions=save_predictions,
                            limit=limit, client_factory=client_factory)
        models.append({"model_id": config.model_id, "result_file": f"{slug}/results.json",
                       "main_score": result["metrics"]["overall_hard_accuracy"]})
    cases = spec.cases[:limit] if limit else spec.cases
    summary = {"schema_version": SUMMARY_SCHEMA, "benchmark": _benchmark_record(spec),
               "limited": bool(limit), "case_limit": limit or None,
               "evaluated_cases": len(cases),
               "evaluated_questions": sum(len(case.targets) for case in cases),
               "models": models}
    _write_json(root / "summary.json", summary)
    return summary
