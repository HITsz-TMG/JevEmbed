"""Strict, model-neutral benchmark file format."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

from ..errors import ValidationError
from ..schemas import validate_request
from ..serialization import validate_json


BENCHMARK_SCHEMA = "jevembed-benchmark-v1"
RESULT_SCHEMA = "jevembed-benchmark-result-v1"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class BenchmarkTarget:
    kind: str
    labels: tuple[str, ...]
    mode: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class BenchmarkCase:
    record_id: str
    group: str | None
    request: dict
    answers: dict
    metadata: dict
    targets: dict[str, BenchmarkTarget]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    version: str
    split: str
    description: str
    data_files: tuple[str, ...]
    manifest_sha256: str
    data_sha256: str
    cases: tuple[BenchmarkCase, ...]

    @property
    def question_count(self):
        return sum(len(case.targets) for case in self.cases)

    @property
    def data_file(self):
        """Backward-compatible access for single-file benchmarks."""
        return self.data_files[0] if len(self.data_files) == 1 else None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads(value, source):
    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        value = json.loads(text, object_pairs_hook=_unique_object)
        validate_json(value)
        return value
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
        raise ValidationError(f"{source}: {exc}") from exc


def _number(value, low, high, source):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValidationError(f"{source} must be a finite number in [{low}, {high}]")
    return float(value)


def _target(question, answer, source):
    if not isinstance(answer, dict) or len(answer) != 1:
        raise ValidationError(f"{source} must contain exactly one target field")
    kind = question["type"]
    if kind == "choice":
        labels = tuple(question["criteria"])
    elif kind == "score":
        labels = tuple(str(index) for index in range(len(question["criteria"])))
    else:
        labels = ()
    key, value = next(iter(answer.items()))
    if kind == "noul":
        if key != "noul":
            raise ValidationError(f"{source} requires noul")
        numeric = int(value) if type(value) is bool else value
        return BenchmarkTarget(kind, labels, "noul", (_number(numeric, 0, 1, source),))
    if key == "probabilities":
        if not isinstance(value, dict) or set(value) != set(labels):
            raise ValidationError(f"{source}.probabilities must contain every label exactly once")
        probabilities = tuple(_number(value[label], 0, 1, source) for label in labels)
        total = math.fsum(probabilities)
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValidationError(f"{source}.probabilities must sum to 1")
        return BenchmarkTarget(kind, labels, "distribution",
                               tuple(probability / total for probability in probabilities))
    if kind == "choice" and key == "choice" and isinstance(value, str) and value in labels:
        return BenchmarkTarget(kind, labels, "distribution",
                               tuple(float(label == value) for label in labels))
    if kind == "score" and key == "level" and type(value) is int and 0 <= value < len(labels):
        return BenchmarkTarget(kind, labels, "distribution",
                               tuple(float(index == value) for index in range(len(labels))))
    if kind == "score" and key == "score":
        return BenchmarkTarget(kind, labels, "score",
                               (_number(value, 0, len(labels) - 1, source),))
    raise ValidationError(f"{source} has an invalid target for {kind}")


def _load_cases(paths):
    cases, ids = [], set()
    try:
        for path in paths:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    source = f"{path.name} line {line_number}"
                    record = _loads(line, source)
                    allowed = {"id", "group", "request", "answers", "metadata"}
                    if not isinstance(record, dict) or set(record) - allowed:
                        raise ValidationError(f"{source}: fields are id, optional group, request, answers, and optional metadata")
                    record_id = record.get("id")
                    if not isinstance(record_id, str) or not record_id or record_id in ids:
                        raise ValidationError(f"{source}: id must be a unique nonempty string")
                    group = record.get("group")
                    if group is not None and (not isinstance(group, str) or not group):
                        raise ValidationError(f"{source}: group must be a nonempty string")
                    metadata = record.get("metadata", {})
                    if not isinstance(metadata, dict):
                        raise ValidationError(f"{source}: metadata must be an object")
                    request = record.get("request")
                    if isinstance(request, dict) and "model" in request:
                        raise ValidationError(f"{source}: benchmark requests must omit model")
                    try:
                        validate_request(request, "benchmark-model")
                    except ValidationError as exc:
                        raise ValidationError(f"{source}: invalid request: {exc}") from exc
                    answers = record.get("answers")
                    if not isinstance(answers, dict) or set(answers) != set(request["questions"]):
                        raise ValidationError(f"{source}: answers must match the request question IDs exactly")
                    targets = {question_id: _target(request["questions"][question_id], answers[question_id],
                                                    f"{source}.answers.{question_id}")
                               for question_id in request["questions"]}
                    ids.add(record_id)
                    cases.append(BenchmarkCase(record_id, group, request, answers, metadata, targets))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"Cannot read benchmark data: {exc}") from exc
    if not cases:
        raise ValidationError("Benchmark data is empty")
    return tuple(cases)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError(f"Cannot hash benchmark data: {exc}") from exc
    return digest.hexdigest()


def _data_sha256(paths, names):
    if len(paths) == 1:
        try:
            return _sha256(paths[0].read_bytes())
        except OSError as exc:
            raise ValidationError(f"Cannot hash benchmark data: {exc}") from exc
    digest = hashlib.sha256()
    for name, path in zip(names, paths):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_sha256(path)))
        digest.update(b"\0")
    return digest.hexdigest()


def load_benchmark(manifest_path):
    """Load a benchmark manifest and its model-neutral JSONL cases."""
    path = Path(manifest_path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"Cannot read benchmark manifest: {exc}") from exc
    manifest = _loads(raw, path.name)
    allowed = {"schema_version", "name", "version", "split", "description", "data"}
    required = {"schema_version", "name", "version", "split", "data"}
    if not isinstance(manifest, dict) or set(manifest) - allowed or not required <= set(manifest):
        raise ValidationError("Benchmark manifest fields are schema_version, name, version, split, optional description, and data")
    if manifest["schema_version"] != BENCHMARK_SCHEMA:
        raise ValidationError(f"schema_version must be {BENCHMARK_SCHEMA}")
    name = manifest["name"]
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValidationError("Benchmark name must use letters, numbers, dot, underscore, or hyphen")
    for field in ("version", "split"):
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise ValidationError(f"Benchmark {field} must be a nonempty string")
    description = manifest.get("description", "")
    if not isinstance(description, str):
        raise ValidationError("Benchmark description must be a string")
    data_value = manifest["data"]
    if isinstance(data_value, str):
        data_files = [data_value]
    elif isinstance(data_value, list) and data_value:
        data_files = data_value
    else:
        raise ValidationError("Benchmark data must be a relative JSONL path or a nonempty array of paths")
    if any(not isinstance(data_file, str) or not data_file or Path(data_file).is_absolute()
           for data_file in data_files):
        raise ValidationError("Every benchmark data path must be a relative JSONL path")
    if len(set(data_files)) != len(data_files):
        raise ValidationError("Benchmark data paths must be unique")
    root = path.resolve().parent
    data_paths = []
    for data_file in data_files:
        data_path = (root / data_file).resolve()
        if root != data_path.parent and root not in data_path.parents:
            raise ValidationError("Benchmark data must stay inside the manifest directory")
        if data_path.suffix != ".jsonl" or not data_path.is_file():
            raise ValidationError("Benchmark data must be an existing JSONL file")
        data_paths.append(data_path)
    return BenchmarkSpec(name, manifest["version"], manifest["split"], description, tuple(data_files),
                         _sha256(raw), _data_sha256(data_paths, data_files), _load_cases(data_paths))
