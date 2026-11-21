"""The published CLM-Qwen3.5 chat-options layout for the paired-head backend."""

import json

from .base import EmbeddingInput
from ..compiler import TaskPlan


def _to_text(value, indent=0):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    pad = " " * indent
    if isinstance(value, dict):
        parts = [f"{pad}{key}:\n{_to_text(item, indent + 2)}" if isinstance(item, (dict, list)) and item
                 else f"{pad}{key}: {_to_text(item)}" for key, item in value.items()]
        return ("\n\n" if indent == 0 else "\n").join(parts)
    if isinstance(value, list):
        parts = [f"{pad}-\n{_to_text(item, indent + 2)}" if isinstance(item, (dict, list)) and item
                 else f"{pad}- {_to_text(item)}" for item in value]
        return "\n".join(parts)
    return json.dumps(value, ensure_ascii=False)


def _option_lines(question):
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice":
        keys = list(criteria)
        lines = [key if criteria[key] in (None, "") else f"{key}: {_to_text(criteria[key])}" for key in keys]
        return keys, lines, lines, "Answer with the name of one option."
    if kind == "score":
        keys = [str(i) for i in range(len(criteria))]
        descriptions = [_to_text(text) for text in criteria]
        lines = [f"{key}: {text}" for key, text in zip(keys, descriptions, strict=True)]
        return keys, lines, lines, "Answer with the number of one level."
    criteria = criteria or {}
    descriptions = []
    for key in ("false", "true"):
        value = criteria.get(key) if isinstance(criteria, dict) else None
        descriptions.append(key if value in (None, "") else f"{key}: {_to_text(value)}")
    return ["false", "true"], descriptions, descriptions, "Answer true or false."


def _chat_message(context, question):
    instructions = _to_text(question.get("instructions")).strip()
    _, lines, _, answer_format = _option_lines(question)
    heading = "Statement" if question["type"] == "noul" else "Question"
    options = "\n".join(f"- {line}" for line in lines)
    body = f"{heading}: {instructions}\n\nOptions:\n{options}\n\n{answer_format}"
    return f"{context}\n\n{body}" if context else body


def render_query(tokenizer, payload, *, max_tokens, overflow_policy):
    """Render and bound the full chat-options turn while retaining its question suffix."""
    context = _to_text(payload["state"]).strip()
    question = payload["question"]

    def token_ids(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def render(current_context):
        message = _chat_message(current_context, question)
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": message}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    text = render(context)
    ids = token_ids(text)
    cap = max_tokens - 1
    if len(ids) <= cap:
        return text, None
    if overflow_policy == "error":
        raise ValueError(f"CLM state/question/options exceed {cap} tokens ({len(ids)})")
    context_ids = token_ids(context)
    keep = max(0, len(context_ids) - (len(ids) - cap) - 16)
    shortened_context = tokenizer.decode(context_ids[:keep])
    shortened = render(shortened_context)
    shortened_ids = token_ids(shortened)
    text = tokenizer.decode(shortened_ids[-cap:])
    return text, {"original_tokens": len(ids), "encoded_tokens": min(cap, len(shortened_ids)),
                  "kept": "context_head_then_tail_if_options_overflow"}


def compile_clm_request(request, prompts):
    plans = []
    for question_id, question in request["questions"].items():
        labels, _, candidates, _ = _option_lines(question)
        payload = json.dumps({"state": request["state"], "question": question},
                             ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        inputs = [EmbeddingInput("query", "", payload)]
        inputs.extend(EmbeddingInput("document", "", text) for text in candidates)
        kind = question["type"]
        legend = {str(i): value for i, value in enumerate(question.get("criteria", []))} if kind == "score" else {}
        path = "clm_noul" if kind == "noul" else kind
        plans.append(TaskPlan(question_id, kind, path, inputs, labels, legend))
    return plans
