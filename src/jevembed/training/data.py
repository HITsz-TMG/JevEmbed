"""Strict supervised records compiled through the same path as inference."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from ..compiler import compile_request
from ..errors import ValidationError
from ..prompts import PromptAdapter
from ..schemas import validate_request
from ..serialization import serialize, validate_json

PATHS = ("choice", "score", "noul_with_criteria", "noul_without_criteria")


@dataclass
class Example:
    record_id: str
    question_id: str
    group: str | None
    fingerprint: str
    plan: object
    texts: list[str]
    target_mode: int  # 0: distribution, 1: expected Score, 2: Noul probability
    target: list[float]

    def training_row(self):
        return {"texts": self.texts, "path": PATHS.index(self.plan.path),
                "target_mode": self.target_mode, "target": self.target}


def _number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValidationError(f"Target must be a finite number in [{low}, {high}]")
    return float(value)


def _target(plan, answer):
    if not isinstance(answer, dict) or len(answer) != 1:
        raise ValidationError("Each answer must contain exactly one supervision field")
    key, value = next(iter(answer.items()))
    if plan.kind == "noul":
        if key != "noul":
            raise ValidationError("Noul supervision requires noul")
        return 2, [_number(int(value) if type(value) is bool else value, 0, 1)]
    if key == "probabilities":
        if not isinstance(value, dict) or set(value) != set(plan.labels):
            raise ValidationError("Target probabilities must contain every candidate label exactly once")
        probs = [_number(value[label], 0, 1) for label in plan.labels]
        if not math.isclose(sum(probs), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValidationError("Target probabilities must sum to 1")
        return 0, [p / sum(probs) for p in probs]
    if plan.kind == "choice" and key == "choice" and type(value) is str and value in plan.labels:
        return 0, [float(label == value) for label in plan.labels]
    if plan.kind == "score":
        if key == "level" and type(value) is int and 0 <= value < len(plan.labels):
            return 0, [float(i == value) for i in range(len(plan.labels))]
        if key == "score":
            return 1, [_number(value, 0, len(plan.labels) - 1)]
    raise ValidationError(f"Invalid {plan.kind} target: use choice, level, score, or probabilities as appropriate")


def load_examples(path, config):
    examples, ids, seen = [], set(), {}
    adapter = PromptAdapter(config.prompts)
    for line_number, line in enumerate(_lines(path), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, object_pairs_hook=_unique_keys)
            validate_json(record)
            if not isinstance(record, dict) or set(record) - {"id", "group", "request", "answers"}:
                raise ValidationError("Record fields are id, optional group, request, and answers")
            rid = record.get("id")
            if not isinstance(rid, str) or not rid or rid in ids:
                raise ValidationError("Record IDs must be unique nonempty strings")
            group = record.get("group")
            if group is not None and (not isinstance(group, str) or not group):
                raise ValidationError("group must be a nonempty string")
            request, answers = record.get("request"), record.get("answers")
            validate_request(request, config.model_id)
            if not isinstance(answers, dict) or set(answers) != set(request["questions"]):
                raise ValidationError("answers must match the request question IDs exactly")
            ids.add(rid)
            for plan in compile_request(request, config.prompts):
                content = {"state": request["state"], "question": request["questions"][plan.question_id]}
                fingerprint = hashlib.sha256(serialize(content).encode()).hexdigest()
                mode, target = _target(plan, answers[plan.question_id])
                supervision = (mode, target)
                if fingerprint in seen and seen[fingerprint] != supervision:
                    raise ValidationError("Conflicting labels for identical state/question supervision")
                seen[fingerprint] = supervision
                examples.append(Example(rid, plan.question_id, group, fingerprint, plan,
                                        [adapter.render(item) for item in plan.inputs], mode, target))
        except (ValueError, TypeError, KeyError, ValidationError) as exc:
            raise ValidationError(f"Training data line {line_number}: {exc}") from exc
    if not examples:
        raise ValidationError("Training data is empty")
    return examples


def _lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        yield from handle


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate JSON object key")
        result[key] = value
    return result


def check_disjoint(train, validation):
    for label, getter in [("record IDs", lambda x: x.record_id), ("state/question pairs", lambda x: x.fingerprint),
                          ("groups", lambda x: x.group)]:
        left, right = ({getter(x) for x in rows} - {None} for rows in (train, validation))
        if left & right:
            raise ValidationError(f"Training and validation overlap in {label}; use independent splits")


def check_lengths(model, examples, overflow_policy):
    """Match Transformer.tokenize preprocessing; never truncate without opting in."""
    counts = {"questions": len(examples), "inputs": 0, "overlong_questions": 0, "overlong_inputs": 0,
              "max_original_tokens": 0, "limit": model.max_seq_length, "overflow_policy": overflow_policy}
    for example in examples:
        texts = [t.strip() for t in example.texts]
        if model[0].do_lower_case:
            texts = [t.lower() for t in texts]
        lengths = [len(ids) for ids in model.tokenizer(texts, truncation=False, padding=False)["input_ids"]]
        count = sum(n > model.max_seq_length for n in lengths)
        counts["inputs"] += len(lengths)
        counts["overlong_questions"] += bool(count)
        counts["overlong_inputs"] += count
        counts["max_original_tokens"] = max(counts["max_original_tokens"], max(lengths))
    if counts["overlong_inputs"] and overflow_policy == "error":
        raise ValidationError(f"{counts['overlong_inputs']} training/evaluation inputs exceed "
                              f"{model.max_seq_length} tokens; shorten data or explicitly enable truncation")
    return counts


def check_lengths_distributed(model, examples, overflow_policy):
    """Check each input once across ranks, then report global counts on every rank."""
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return check_lengths(model, examples, overflow_policy)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local = check_lengths(model, examples[rank::world_size], "truncate")
    counts = torch.tensor([local[key] for key in
                           ("questions", "inputs", "overlong_questions", "overlong_inputs", "max_original_tokens")],
                          dtype=torch.long, device=next(model.parameters()).device)
    dist.all_reduce(counts[:4], op=dist.ReduceOp.SUM)
    dist.all_reduce(counts[4:], op=dist.ReduceOp.MAX)
    result = dict(zip(("questions", "inputs", "overlong_questions", "overlong_inputs", "max_original_tokens"),
                      counts.tolist()))
    result.update(limit=model.max_seq_length, overflow_policy=overflow_policy)
    if result["overlong_inputs"] and overflow_policy == "error":
        raise ValidationError(f"{result['overlong_inputs']} training/evaluation inputs exceed "
                              f"{model.max_seq_length} tokens; shorten data or explicitly enable truncation")
    return result
