"""Strict teacher-output parsing and validation for hard Jev labels."""

import json
import re

from ..compiler import compile_request
from ..config import PromptConfig
from ..errors import ValidationError
from ..serialization import validate_json
from ..training.data import _target


_FENCE = re.compile(r"```(?:json)?[ \t]*\n(.*?)\n```\Z", re.IGNORECASE | re.DOTALL)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValidationError(f"Non-finite JSON number is forbidden: {value}")


def parse_json(text):
    """Parse a full JSON response or one enclosing JSON fence, without extraction."""
    if type(text) is not str or not text.strip():
        raise ValidationError("Teacher response must contain JSON")
    content = text.strip()
    if content.startswith("```"):
        match = _FENCE.fullmatch(content)
        if match is None:
            raise ValidationError("Only one enclosing JSON code fence is allowed")
        content = match.group(1)
    try:
        parsed = json.loads(content, object_pairs_hook=_unique_pairs,
                            parse_constant=_reject_constant)
        validate_json(parsed)
        return parsed
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValidationError(f"Teacher response is not valid JSON: {exc}") from exc


def validate_state(state, config):
    """Check finite JSON and every supported Draft 2020-12 shape constraint."""
    validate_json(state)
    try:
        from jsonschema import Draft202012Validator
        from referencing import Registry
    except ImportError as exc:
        raise ValidationError("JSON Schema validation requires jevembed[synthesis]") from exc

    def no_remote(uri):
        raise ValidationError(f"Remote JSON Schema reference is forbidden: {uri}")

    schema = config["task"]["state_schema"]
    validator = Draft202012Validator(schema, registry=Registry(retrieve=no_remote))
    try:
        error = next(validator.iter_errors(state), None)
    except Exception as exc:
        raise ValidationError(f"State schema reference could not be resolved: {exc}") from exc
    if error is not None:
        path = ".".join(map(str, error.absolute_path)) or "root"
        raise ValidationError(f"State violates task.state_schema at {path}: {error.message}")


def validate_answer(answer, config):
    """Accept only hard Choice, Score, or Noul labels for the fixed question."""
    if type(answer) is not dict or len(answer) != 1:
        raise ValidationError("answer must contain exactly one hard label")
    question = config["task"]["question"]
    kind = question["type"]
    if kind == "choice":
        if set(answer) != {"choice"} or type(answer["choice"]) is not str:
            raise ValidationError("Choice answer must be {choice: candidate label}")
    elif kind == "score":
        if set(answer) != {"level"} or type(answer["level"]) is not int:
            raise ValidationError("Score answer must be {level: integer index}")
    else:
        if set(answer) != {"noul"} or type(answer["noul"]) is not bool:
            raise ValidationError("Noul answer must be {noul: boolean}")
    task = config["task"]
    root_type = task["state_schema"]["type"]
    state = {"string": "example", "object": {}, "array": []}[root_type]
    request = {"state": state, "questions": {task["question_id"]: question}}
    plan = compile_request(request, PromptConfig())[0]
    _target(plan, answer)


def validate_generation(payload, config):
    """Return validated state and answer from one exact teacher result object."""
    validate_json(payload)
    if type(payload) is not dict or set(payload) != {"state", "answer"}:
        raise ValidationError("Teacher result must contain exactly state and answer")
    state, answer = payload["state"], payload["answer"]
    validate_state(state, config)
    validate_answer(answer, config)
    return state, answer
