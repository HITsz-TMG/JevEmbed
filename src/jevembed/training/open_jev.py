"""Convert a pinned Open-Jev subset to supervised JevEmbed JSONL."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

from ..config import ModelConfig
from ..errors import ValidationError
from ..serialization import serialize, validate_json
from .data import check_disjoint, load_examples
from .run import write_json

DATASET = "ZefanCai/Open-Jev"
DEFAULT_REVISION = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"
DEFAULT_SUBSET = "release-v2-redistributable"


def convert_record(row, split):
    """Only state, question, kind, and options enter the compiled model inputs."""
    validate_json(row)
    if row.get("split") != split:
        raise ValidationError("Source row split differs from the requested split")
    options, target, kind = row["options"], row["target"], row["kind"]
    if not isinstance(options, list) or not options or any(not isinstance(x, str) or not x for x in options):
        raise ValidationError("Open-Jev options must be nonempty strings")
    if len(set(options)) != len(options) or not isinstance(target, list) or len(target) != len(options):
        raise ValidationError("Duplicate options or target length mismatch")
    if any(type(x) not in (float, int) or not 0 <= x <= 1 for x in target) or abs(sum(target) - 1) > 1e-6:
        raise ValidationError("Expected a probability distribution; independent multilabel targets need another adapter")
    question = {"type": kind, "instructions": row["question"]}
    if kind == "choice":
        # Null descriptions make the inference compiler emit each original option verbatim.
        question["criteria"] = dict.fromkeys(options)
        answer = {"probabilities": dict(zip(options, target))}
    elif kind == "score":
        values = row.get("metadata", {}).get("score_values")
        if values is not None and values != list(range(len(options))):
            raise ValidationError("Non-index Score values are not supported by JevEmbed")
        question["criteria"] = options
        answer = {"probabilities": dict(zip(map(str, range(len(options))), target))}
    elif kind == "noul":
        if set(options) != {"no", "yes"}:
            raise ValidationError("Noul options must be no/yes")
        # Source rows provide no descriptive true/false criteria; use criteria-free Noul.
        answer = {"noul": target[options.index("yes")]}
    else:
        raise ValidationError(f"Unsupported Open-Jev kind: {kind}")
    return {"id": row["id"], "group": row["group_id"],
            "request": {"state": row["state"], "questions": {"decision": question}},
            "answers": {"decision": answer}}


def fingerprint(record):
    return hashlib.sha256(serialize({"state": record["request"]["state"],
                                   "question": record["request"]["questions"]["decision"]}).encode()).hexdigest()


def prepare(source_paths, output, revision, subset, deduplicate=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValidationError("Prepared data output must be new or empty")
    # Validation takes precedence when exact model inputs recur across source groups.
    pools, groups, stats, all_targets = {}, {}, {}, {}
    for split in ("validation", "train"):
        path = Path(source_paths[split])
        records, seen, group_ids, ids = [], set(), set(), set()
        count = Counter()
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                    record = convert_record(row, split)
                except (ValueError, KeyError, TypeError, ValidationError) as exc:
                    raise ValidationError(f"Open-Jev {split} row {line_number}: {exc}") from exc
                count["source_rows"] += 1
                if record["id"] in ids:
                    raise ValidationError("Duplicate source record ID")
                ids.add(record["id"])
                group_ids.add(record["group"])
                key = fingerprint(record)
                target = serialize(record["answers"])
                if key in all_targets and all_targets[key] != target:
                    raise ValidationError("Conflicting labels for identical model inputs")
                all_targets[key] = target
                if key in seen:
                    count["duplicate_inputs"] += 1
                    if deduplicate:
                        count["duplicate_inputs_removed"] += 1
                        continue
                seen.add(key)
                records.append((key, record))
        groups[split] = group_ids
        pools[split] = records
        stats[split] = {**count, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if groups["train"] & groups["validation"]:
        raise ValidationError("Official train/validation group overlap")
    validation_keys = {key for key, _ in pools["validation"]}
    output.mkdir(parents=True, exist_ok=True)
    for split, records in pools.items():
        kept, overlap = [], 0
        for key, record in records:
            if split == "train" and key in validation_keys:
                if not deduplicate:
                    raise ValidationError("Cross-split identical inputs; explicitly use --deduplicate to remove train overlap")
                overlap += 1
            else:
                kept.append(record)
        with (output / f"{split}.jsonl").open("x", encoding="utf-8") as handle:
            for record in kept:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stats[split].update(retained_questions=len(kept), cross_split_inputs_removed=overlap,
                            kinds=dict(Counter(r["request"]["questions"]["decision"]["type"] for r in kept)))
    train = load_examples(output / "train.jsonl", ModelConfig())
    validation = load_examples(output / "validation.jsonl", ModelConfig())
    check_disjoint(train, validation)
    manifest = {"dataset": DATASET, "revision": revision, "subset": subset, "splits": stats,
                "noul_mapping": "criteria-free; preserve question and state; target=P(yes)",
                "choice_mapping": "original option text as label with null description; target distribution preserved",
                "score_mapping": "original ordered option descriptions; target distribution over level indices",
                "excluded_from_model_inputs": ["target", "metadata", "id", "group_id", "source", "split"],
                "deduplicate": deduplicate,
                "split_policy": ("official splits; deduplicate within splits and remove train inputs matching validation"
                                 if deduplicate else "official train/validation rows preserved, including repeated inputs; reject split overlap")}
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Immutable dataset commit SHA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, help="Offline snapshot containing raw/<subset>/*.jsonl.gz")
    parser.add_argument("--deduplicate", action="store_true", help="Optional: changes the official row counts")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("Use an immutable 40-character commit SHA for --revision")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.subset):
        parser.error("Invalid subset name")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be new or empty")
    paths = {}
    for split in ("train", "validation"):
        filename = f"raw/{args.subset}/{split}.jsonl.gz"
        if args.source_dir:
            paths[split] = args.source_dir / filename
        else:
            from huggingface_hub import hf_hub_download
            paths[split] = hf_hub_download(DATASET, filename, repo_type="dataset", revision=args.revision)
    print(json.dumps(prepare(paths, args.output, args.revision, args.subset, args.deduplicate), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
