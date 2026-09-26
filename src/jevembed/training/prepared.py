"""Streaming preparation and memory-mapped training data for each process."""

from collections.abc import Sequence
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from ..backends.base import EmbeddingInput
from ..compiler import TaskPlan
from ..errors import ValidationError
from ..serialization import SERIALIZATION_VERSION
from .data import Example, scan_splits


_FORMAT_VERSION = 1
_ROWS_PER_BATCH = 256
_BYTES_PER_BATCH = 8 * 1024 * 1024


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(train_path, eval_path):
    return {"train": _sha256(train_path),
            "validation": _sha256(eval_path) if eval_path is not None else None}


def _cache_spec(config):
    return {"format_version": _FORMAT_VERSION, "model_id": config.model_id,
            "serialization_version": SERIALIZATION_VERSION, "prompts": asdict(config.prompts)}


def _spec_signature(spec):
    encoded = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_METADATA_KEYS = ("train_questions", "validation_questions", "dataset_sha256",
                  "format_version", "cache_signature")


def _with_identity(counts, spec):
    return {**counts, "format_version": _FORMAT_VERSION, "cache_signature": _spec_signature(spec)}


def is_valid_prepared(cache_dir):
    """Check the committed cache marker and required files without loading Arrow."""
    cache_dir = Path(cache_dir)
    try:
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        spec = manifest["spec"]
        if (spec["format_version"] != _FORMAT_VERSION or manifest["format_version"] != _FORMAT_VERSION
                or spec["serialization_version"] != SERIALIZATION_VERSION
                or manifest["cache_signature"] != _spec_signature(spec)):
            return False
        if (type(manifest["train_questions"]) is not int or manifest["train_questions"] < 1
                or type(manifest["validation_questions"]) is not int
                or manifest["validation_questions"] < 0):
            return False
        hashes = manifest["dataset_sha256"]
        if (not isinstance(hashes, dict) or not isinstance(hashes["train"], str)
                or len(hashes["train"]) != 64 or
                (hashes["validation"] is None) != (manifest["validation_questions"] == 0)
                or (hashes["validation"] is not None and
                    (not isinstance(hashes["validation"], str) or len(hashes["validation"]) != 64))):
            return False
        files = ["train.arrow"]
        if manifest["validation_questions"]:
            files.extend(("validation.arrow", "validation_examples.arrow"))
        return all((cache_dir / name).is_file() and (cache_dir / name).stat().st_size > 0
                   for name in files)
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _cache_matches(cache_dir, spec, hashes, has_validation):
    if not is_valid_prepared(cache_dir):
        return None
    try:
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (manifest.get("spec") != spec or manifest.get("dataset_sha256") != hashes
            or bool(manifest.get("validation_questions")) != has_validation):
        return None
    return {key: manifest[key] for key in _METADATA_KEYS}


def validate_data(train_path, eval_path, config):
    """Validate both splits without creating a cache or importing Arrow."""
    return _with_identity(scan_splits(train_path, eval_path, config), _cache_spec(config))


def _row_bytes(value):
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, list):
        return sum(_row_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_row_bytes(item) for item in value.values())
    return 8


class _BoundedWriter:
    """Flush ArrowWriter before its pending Python rows grow beyond the byte budget."""

    def __init__(self, writer):
        self.writer = writer
        self.rows = 0
        self.bytes = 0

    def write(self, row):
        size = _row_bytes(row)
        if self.rows and (self.rows >= _ROWS_PER_BATCH or self.bytes + size > _BYTES_PER_BATCH):
            self.flush()
        self.writer.write(row, writer_batch_size=_ROWS_PER_BATCH + 1)
        self.rows += 1
        self.bytes += size
        if self.rows >= _ROWS_PER_BATCH or self.bytes >= _BYTES_PER_BATCH:
            self.flush()

    def flush(self):
        if self.rows:
            self.writer.write_examples_on_file()
            self.rows = 0
            self.bytes = 0


def prepare_data(train_path, eval_path, config, cache_dir):
    """Validate once, write bounded Arrow batches, and publish a complete cache."""
    from datasets import Features, Sequence as DatasetSequence, Value
    from datasets.arrow_writer import ArrowWriter

    cache_dir = Path(cache_dir)
    spec = _cache_spec(config)
    if cache_dir.is_dir():
        cached = _cache_matches(cache_dir, spec, _source_hashes(train_path, eval_path),
                                eval_path is not None)
        if cached is not None:
            return cached
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}.build-", dir=cache_dir.parent))
    row_features = Features({"texts": DatasetSequence(Value("string")), "path": Value("int64"),
                             "target_mode": Value("int64"), "target": DatasetSequence(Value("float64"))})
    example_features = Features({"record_id": Value("string"), "question_id": Value("string"),
                                 "group": Value("string"), "fingerprint": Value("string"),
                                 "plan_json": Value("string")})
    try:
        with ArrowWriter(path=str(staging / "train.arrow"), features=row_features,
                         writer_batch_size=_ROWS_PER_BATCH) as train_writer:
            bounded_train = _BoundedWriter(train_writer)
            if eval_path is None:
                metadata = scan_splits(train_path, None, config,
                                       on_train=lambda example: bounded_train.write(example.training_row()))
            else:
                with ArrowWriter(path=str(staging / "validation.arrow"), features=row_features,
                                 writer_batch_size=_ROWS_PER_BATCH) as validation_writer, ArrowWriter(
                                     path=str(staging / "validation_examples.arrow"),
                                     features=example_features, writer_batch_size=_ROWS_PER_BATCH) as examples_writer:
                    bounded_validation = _BoundedWriter(validation_writer)
                    bounded_examples = _BoundedWriter(examples_writer)
                    def write_validation(example):
                        bounded_validation.write(example.training_row())
                        bounded_examples.write({"record_id": example.record_id,
                                                "question_id": example.question_id,
                                                "group": example.group,
                                                "fingerprint": example.fingerprint,
                                                "plan_json": json.dumps(asdict(example.plan), ensure_ascii=False)})

                    metadata = scan_splits(
                        train_path, eval_path, config,
                        on_train=lambda example: bounded_train.write(example.training_row()),
                        on_validation=write_validation)
                    bounded_validation.flush()
                    bounded_examples.flush()
                    validation_writer.finalize()
                    examples_writer.finalize()
            bounded_train.flush()
            train_writer.finalize()
        if _source_hashes(train_path, eval_path) != metadata["dataset_sha256"]:
            raise ValidationError("Training data changed while the prepared cache was being built")
        metadata = _with_identity(metadata, spec)
        (staging / "manifest.json").write_text(
            json.dumps({"spec": spec, **metadata}, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8")
        _publish(staging, cache_dir)
        return metadata
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _publish(staging, cache_dir):
    """Keep an old cache intact until the complete replacement is ready."""
    if not cache_dir.exists():
        os.replace(staging, cache_dir)
        return
    backup = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}.old-", dir=cache_dir.parent))
    backup.rmdir()
    os.replace(cache_dir, backup)
    try:
        os.replace(staging, cache_dir)
    except Exception:
        os.replace(backup, cache_dir)
        raise
    shutil.rmtree(backup)


class _LazyValidationExamples(Sequence):
    def __init__(self, rows, metadata, indices=None):
        self._rows = rows
        self._metadata = metadata
        self._indices = range(len(rows)) if indices is None else indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(self._rows, self._metadata, self._indices[index])
        row = self._rows[self._indices[index]]
        info = self._metadata[self._indices[index]]
        plan_data = json.loads(info["plan_json"])
        plan_data["inputs"] = [EmbeddingInput(**item) for item in plan_data["inputs"]]
        return Example(info["record_id"], info["question_id"], info["group"],
                       info["fingerprint"], TaskPlan(**plan_data), row["texts"],
                       row["target_mode"], row["target"])


def open_prepared(cache_dir, expected_metadata=None):
    """Open file-backed Arrow datasets and a lazy view of validation examples."""
    from datasets import Dataset

    cache_dir = Path(cache_dir)
    try:
        if not is_valid_prepared(cache_dir):
            raise ValidationError("Prepared data cache is missing or incomplete")
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        actual_metadata = {key: manifest[key] for key in _METADATA_KEYS}
        if expected_metadata is not None and actual_metadata != expected_metadata:
            raise ValidationError("Prepared data cache identity differs from rank-zero metadata")
        train = Dataset.from_file(str(cache_dir / "train.arrow"), in_memory=False)
        if manifest["validation_questions"]:
            validation = Dataset.from_file(str(cache_dir / "validation.arrow"), in_memory=False)
            details = Dataset.from_file(str(cache_dir / "validation_examples.arrow"), in_memory=False)
            if len(validation) != len(details) or len(validation) != manifest["validation_questions"]:
                raise ValidationError("Prepared validation cache is incomplete")
            examples = _LazyValidationExamples(validation, details)
        else:
            validation, examples = None, _LazyValidationExamples((), ())
        if len(train) != manifest["train_questions"]:
            raise ValidationError("Prepared training cache is incomplete")
        return train, validation, examples
    except ValidationError:
        raise
    except (OSError, KeyError, ValueError) as exc:
        raise ValidationError("Prepared data cache is missing or invalid") from exc
