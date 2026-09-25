"""Evaluate an embedding model on the JevEmbed-Data test split."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import time

import torch
from torch import nn

from jevembed.backends import PairedProjectionBackend, SentenceTransformersBackend
from jevembed.config import ModelConfig
from jevembed.training.data import load_examples
from jevembed.training.evaluation import evaluate


class PrefetchedModel(nn.Module):
    """Present batched backend vectors to the shared supervised evaluator."""

    def __init__(self, examples, vectors):
        super().__init__()
        self.register_parameter("device_anchor", nn.Parameter(torch.empty(0), requires_grad=False))
        self.texts = [text for example in examples for text in example.texts]
        self.vectors = torch.tensor(vectors, dtype=torch.float32)
        self.offset = 0

    def encode(self, texts, **kwargs):
        end = self.offset + len(texts)
        if self.texts[self.offset:end] != texts:
            raise RuntimeError("Encoded vectors are out of order")
        result = self.vectors[self.offset:end]
        self.offset = end
        return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_files(path):
    if path.is_file() and path.suffix == ".jsonl":
        return [path]
    files = sorted(path.glob("test-*.parquet")) if path.is_dir() else [path]
    if not files or any(not file.is_file() or file.suffix != ".parquet" for file in files):
        raise ValueError("--test-data must be a JSONL file, a test Parquet shard, or a directory of test shards")
    return files


def parquet_to_jsonl(files, destination):
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Reading Parquet requires jevembed[train]") from exc
    rows = 0
    with destination.open("w", encoding="utf-8") as output:
        for file in files:
            reader = parquet.ParquetFile(file)
            for batch in reader.iter_batches(batch_size=2048):
                for row in batch.to_pylist():
                    record = {"id": row["id"], "group": row["group"],
                              "request": json.loads(row["request_json"]),
                              "answers": json.loads(row["answers_json"])}
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rows += 1
    return rows


def combine(parts):
    names = ("loss", "choice_accuracy", "score_level_accuracy", "score_mae", "noul_mae",
             "noul_binary_accuracy", "target_distribution_brier", "target_distribution_tvd")
    questions = sum(part["questions"] for part in parts)
    result = {"questions": questions}
    for name in names:
        count = questions if name == "loss" else sum(part[name + "_n"] for part in parts)
        total = sum((part[name] or 0.0) * (part["questions"] if name == "loss" else part[name + "_n"])
                    for part in parts)
        result[name] = total / count if count else None
        if name != "loss":
            result[name + "_n"] = count
    hard_names = ("choice_accuracy", "score_level_accuracy", "noul_binary_accuracy")
    hard_count = sum(result[name + "_n"] for name in hard_names)
    correct = sum((result[name] or 0.0) * result[name + "_n"] for name in hard_names)
    result["overall_hard_accuracy"] = correct / hard_count if hard_count else None
    result["overall_hard_accuracy_n"] = hard_count
    return result


def encode_length_sorted(backend, inputs, batch_size, backend_name):
    """Batch similar-length inputs and restore their original scoring order."""
    indexed = sorted(enumerate(inputs), key=lambda pair: (
        pair[1].role, len(backend.adapter.render(pair[1]))))
    vectors = [None] * len(inputs)
    # Sentence Transformers sorts inside each call; a larger call gives it a
    # wider length window. The paired-projection backend needs explicit batches.
    call_size = min(4096, batch_size * 32) if backend_name == "sentence_transformers" else batch_size
    for start in range(0, len(indexed), call_size):
        group = indexed[start:start + call_size]
        batch = backend.encode([item for _, item in group])
        if len(batch.vectors) != len(group):
            raise RuntimeError("Backend returned the wrong number of embeddings")
        for (index, _), vector in zip(group, batch.vectors):
            vectors[index] = vector
    if any(vector is None for vector in vectors):
        raise RuntimeError("An embedding was not returned")
    return vectors


def score_chunk(backend, examples, config, batch_size):
    inputs = [item for example in examples for item in example.plan.inputs]
    vectors = encode_length_sorted(backend, inputs, batch_size, config.backend)
    model = PrefetchedModel(examples, vectors)
    result = evaluate(model, examples, config)
    if model.offset != len(inputs):
        raise RuntimeError("Some embeddings were not scored")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, help="Local base model snapshot")
    parser.add_argument("--projection-path", type=Path, help="Local paired-projection checkpoint or directory")
    parser.add_argument("--adapter-path", type=Path, help="LoRA adapter to evaluate")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"))
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--overflow-policy", choices=("error", "truncate"))
    parser.add_argument("--encode-batch-size", type=int)
    parser.add_argument("--chunk-questions", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test only; zero uses the full split")
    args = parser.parse_args(argv)
    if args.chunk_questions < 1 or args.limit < 0 or (args.encode_batch_size is not None and args.encode_batch_size < 1):
        parser.error("Batch sizes must be positive and --limit must be nonnegative")
    if args.max_input_tokens is not None and args.max_input_tokens < 1:
        parser.error("--max-input-tokens must be positive")
    if args.projection_path and not args.projection_path.exists():
        parser.error("Projection path does not exist")
    if args.adapter_path and not args.adapter_path.exists():
        parser.error("Adapter path does not exist")
    config = ModelConfig.load(args.config)
    changes = {"device": args.device, "dtype": args.dtype, "cache_capacity": 0}
    if args.model_path:
        changes["model_name_or_path"] = str(args.model_path.resolve())
    if args.projection_path:
        changes["projection_name_or_path"] = str(args.projection_path.resolve())
    if args.adapter_path:
        changes["adapter_name_or_path"] = str(args.adapter_path.resolve())
        changes["adapter_revision"] = None
    if args.model_path or args.projection_path or args.adapter_path:
        changes["local_files_only"] = True
    if args.max_input_tokens is not None:
        changes["max_input_tokens"] = args.max_input_tokens
    if args.overflow_policy:
        changes["overflow_policy"] = args.overflow_policy
    if args.encode_batch_size:
        changes["batch_size"] = args.encode_batch_size
    config = replace(config, **changes)
    files = test_files(args.test_data)
    file_hashes = [sha256(file) for file in files]
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="jevembed-test-") as temporary:
        if files[0].suffix == ".jsonl":
            data_path = files[0]
        else:
            data_path = Path(temporary) / "test.jsonl"
            parquet_to_jsonl(files, data_path)
        examples = load_examples(data_path, config)
    if args.limit:
        examples = examples[:args.limit]
    backend_class = SentenceTransformersBackend if config.backend == "sentence_transformers" else PairedProjectionBackend
    if config.backend not in ("sentence_transformers", "paired_projection"):
        parser.error("This evaluation supports local Sentence Transformers and paired-projection backends")
    backend = backend_class(config=config)
    identity = backend.prepare()
    print(f"Evaluating {len(examples)} test questions with {config.model_id}", flush=True)
    parts = []
    with torch.inference_mode():
        for start in range(0, len(examples), args.chunk_questions):
            chunk = examples[start:start + args.chunk_questions]
            parts.append(score_chunk(backend, chunk, config, config.batch_size))
            done = min(start + len(chunk), len(examples))
            print(f"{done}/{len(examples)} scored; elapsed={time.monotonic()-started:.1f}s", flush=True)
    metrics = combine(parts)
    base_revision = identity.get("resolved_revision", identity.get("resolved_base_revision"))
    adapter_file = Path(config.adapter_name_or_path) / "adapter_model.safetensors" if config.adapter_name_or_path else None
    result = {"dataset": "HIT-TMG/JevEmbed-Data", "split": "test", "questions": len(examples),
              "limited": bool(args.limit), "test_shard_sha256": file_hashes,
              "model_id": config.model_id, "base_revision": base_revision,
              "adapter_sha256": sha256(adapter_file) if adapter_file and adapter_file.is_file() else None,
              "scoring": asdict(config.scoring), "prompts": asdict(config.prompts),
              "max_input_tokens": identity["max_input_tokens"], "overflow_policy": config.overflow_policy,
              "metrics": metrics, "seconds": time.monotonic() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                encoding="utf-8")
    temporary_output.replace(args.output)
    print(json.dumps({"overall_hard_accuracy": metrics["overall_hard_accuracy"],
                      "questions": metrics["questions"], "output": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
