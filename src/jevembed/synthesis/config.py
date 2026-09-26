"""Strict, versioned configuration for synthesis with a fixed Jev question."""

import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import yaml

from ..errors import ValidationError
from ..schemas import validate_request
from ..serialization import validate_json
from .quotas import quota_labels


_SLUG = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_CASE_TYPES = frozenset({"standard", "negation", "boundary", "hard_negative"})
_PROTECTED_OPTIONS = frozenset({"model", "messages", "max_tokens", "temperature",
                                "stream", "auth", "authorization", "api_key", "api_key_env",
                                "headers", "base_url", "url", "n", "best_of",
                                "max_completion_tokens", "max_new_tokens", "stream_options",
                                "tools", "tool_choice", "functions", "function_call"})
_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            raise ValidationError("Expected a YAML mapping")
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise ValidationError("YAML mapping keys must be strings") from exc
            if duplicate:
                raise ValidationError(f"Duplicate YAML key: {key!r}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _mapping(value, path, allowed, required=()):
    if type(value) is not dict:
        raise ValidationError(f"{path} must be an object")
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown:
        raise ValidationError(f"{path} has unknown keys: {sorted(map(str, unknown))}")
    if missing:
        raise ValidationError(f"{path} is missing: {sorted(missing)}")


def _text(value, path):
    if type(value) is not str or not value.strip():
        raise ValidationError(f"{path} must be a nonempty string")
    return value


def _slug(value, path):
    if type(value) is not str or not _SLUG.fullmatch(value):
        raise ValidationError(f"{path} must be a lowercase slug using letters, digits, '-' or '_'")
    return value


def _positive_number(value, path):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValidationError(f"{path} must be a positive finite number")
    return value


def _nonnegative_number(value, path):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValidationError(f"{path} must be a nonnegative finite number")
    return value


def _integer(value, path, minimum):
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{path} must be an integer >= {minimum}")
    return value


def _meaningful(value):
    if type(value) is str:
        return bool(value.strip())
    if type(value) is dict:
        return bool(value) and all(type(key) is str and key.strip() for key in value) and any(
            _meaningful(item) for item in value.values())
    if type(value) is list:
        return bool(value) and any(_meaningful(item) for item in value)
    return value is not None


def _check_question(question, question_id, root_type):
    _mapping(question, "task.question", {"type", "instructions", "criteria"}, {"type", "instructions"})
    state = {"string": "example", "object": {}, "array": []}[root_type]
    validate_request({"model": "synthesis", "state": state,
                      "questions": {question_id: question}})
    if not _meaningful(question["instructions"]):
        raise ValidationError("task.question.instructions must have meaningful content")
    criteria = question.get("criteria")
    if question["type"] == "choice":
        for label, description in criteria.items():
            if type(label) is not str or not label.strip():
                raise ValidationError("Choice labels must be nonempty strings")
            if description is not None and not _meaningful(description):
                raise ValidationError(f"Choice description for {label!r} is empty")
    elif question["type"] == "score":
        if any(not _meaningful(description) for description in criteria):
            raise ValidationError("Score levels must have nonempty ordered descriptions")
    elif "criteria" in question:
        if any(not _meaningful(description) for description in criteria.values()):
            raise ValidationError("Noul true and false criteria must be nonempty")
    return question


def _check_schema(schema):
    if type(schema) is not dict:
        raise ValidationError("task.state_schema must be a JSON Schema object")
    root_type = schema.get("type")
    if type(root_type) is not str or root_type not in {"string", "object", "array"}:
        raise ValidationError("task.state_schema must declare root type string, object or array")
    if schema.get("$schema", _DRAFT_2020_12) != _DRAFT_2020_12:
        raise ValidationError("task.state_schema must use JSON Schema Draft 2020-12")
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise ValidationError("JSON Schema validation requires jevembed[synthesis]") from exc
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ValidationError(f"task.state_schema is invalid: {exc}") from exc

    references = []
    anchors = {}

    def check_references(value, resource):
        if type(value) is dict:
            if value is not schema and "$id" in value:
                resource = value
            if "$schema" in value and value["$schema"] != _DRAFT_2020_12:
                raise ValidationError("task.state_schema must use JSON Schema Draft 2020-12 throughout")
            for name in ("$anchor", "$dynamicAnchor"):
                if name in value:
                    anchors.setdefault(id(resource), set()).add(value[name])
            for key, item in value.items():
                if key in {"$ref", "$dynamicRef", "$recursiveRef"} and (
                    type(item) is not str or not item.startswith("#")
                ):
                    raise ValidationError("task.state_schema permits only local # references")
                if key == "$ref":
                    references.append((item, resource))
                check_references(item, resource)
        elif type(value) is list:
            for item in value:
                check_references(item, resource)

    check_references(schema, schema)
    for reference, resource in references:
        fragment = unquote(reference[1:])
        if not fragment:
            continue
        if not fragment.startswith("/"):
            if fragment not in anchors.get(id(resource), set()):
                raise ValidationError(f"Unresolved local JSON Schema reference: {reference}")
            continue
        target = resource
        for token in fragment[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if type(target) is dict and token in target:
                target = target[token]
            elif type(target) is list and token.isdecimal() and int(token) < len(target):
                target = target[int(token)]
            else:
                raise ValidationError(f"Unresolved local JSON Schema reference: {reference}")
    return root_type


def _check_request_options(options):
    if type(options) is not dict:
        raise ValidationError("teacher.request_options must be an object")

    def inspect(value):
        if type(value) is dict:
            protected = _PROTECTED_OPTIONS.intersection(key.lower() for key in value)
            if protected:
                raise ValidationError(f"teacher.request_options cannot override {sorted(protected)}")
            for item in value.values():
                inspect(item)
        elif type(value) is list:
            for item in value:
                inspect(item)

    inspect(options)
    return options


def validate_config(raw):
    """Return a normalized plain dict; reject unknown or unsafe settings."""
    validate_json(raw)
    _mapping(raw, "config", {"version", "teacher", "task", "generation"},
             {"version", "teacher", "task"})
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise ValidationError("config.version must be 1")

    teacher = raw["teacher"]
    _mapping(teacher, "teacher", {"base_url", "model", "api_key_env", "timeout", "max_retries",
                                  "temperature", "max_tokens", "request_options"},
             {"base_url", "model"})
    base_url = _text(teacher["base_url"], "teacher.base_url").rstrip("/")
    try:
        parsed = urlsplit(base_url)
        valid_host = bool(parsed.hostname) and parsed.port != 0
    except ValueError:
        valid_host = False
        parsed = None
    if parsed is None or parsed.scheme not in {"http", "https"} or not valid_host or (
        parsed.username or parsed.password or parsed.query or parsed.fragment
        or not parsed.path.endswith("/v1")
    ):
        raise ValidationError("teacher.base_url must be an http(s) /v1 root without credentials or query")
    api_key_env = teacher.get("api_key_env")
    if api_key_env is not None and (type(api_key_env) is not str or not _ENV_NAME.fullmatch(api_key_env)):
        raise ValidationError("teacher.api_key_env must name an environment variable")
    options = _check_request_options(teacher.get("request_options", {}))
    normalized_teacher = {
        "base_url": base_url,
        "model": _text(teacher["model"], "teacher.model"),
        "api_key_env": api_key_env,
        "timeout": _positive_number(teacher.get("timeout", 120), "teacher.timeout"),
        "max_retries": _integer(teacher.get("max_retries", 2), "teacher.max_retries", 0),
        "temperature": _nonnegative_number(teacher.get("temperature", 0.7), "teacher.temperature"),
        "max_tokens": _integer(teacher.get("max_tokens", 2048), "teacher.max_tokens", 1),
        "request_options": options,
    }

    task = raw["task"]
    _mapping(task, "task", {"id", "domain", "question_id", "question", "state_schema",
                            "state_description", "labeling_guidance"},
             {"id", "domain", "question", "state_schema", "state_description"})
    schema = task["state_schema"]
    root_type = _check_schema(schema)
    question_id = _text(task.get("question_id", "decision"), "task.question_id")
    normalized_task = {
        "id": _slug(task["id"], "task.id"),
        "domain": _text(task["domain"], "task.domain"),
        "question_id": question_id,
        "question": _check_question(task["question"], question_id, root_type),
        "state_schema": schema,
        "state_description": _text(task["state_description"], "task.state_description"),
        "labeling_guidance": task.get("labeling_guidance", ""),
    }
    if type(normalized_task["labeling_guidance"]) is not str:
        raise ValidationError("task.labeling_guidance must be a string")

    generation = raw.get("generation", {})
    _mapping(generation, "generation", {"count", "concurrency", "max_attempts_per_sample",
                                        "verify_labels", "case_types", "seed", "label_quotas"})
    case_types = generation.get("case_types", ["standard", "boundary", "hard_negative"])
    if type(case_types) is not list or not case_types or any(type(item) is not str for item in case_types):
        raise ValidationError("generation.case_types must be a nonempty list")
    if len(case_types) != len(set(case_types)) or set(case_types) - _CASE_TYPES:
        raise ValidationError(f"generation.case_types may contain each of {sorted(_CASE_TYPES)} once")
    verify_labels = generation.get("verify_labels", True)
    if type(verify_labels) is not bool:
        raise ValidationError("generation.verify_labels must be a boolean")
    normalized_generation = {
        "count": _integer(generation.get("count", 10), "generation.count", 1),
        "concurrency": _integer(generation.get("concurrency", 2), "generation.concurrency", 1),
        "max_attempts_per_sample": _integer(generation.get("max_attempts_per_sample", 3),
                                             "generation.max_attempts_per_sample", 1),
        "verify_labels": verify_labels,
        "case_types": case_types,
        "seed": _integer(generation.get("seed", 42), "generation.seed", 0),
    }
    quotas = generation.get("label_quotas")
    labels = quota_labels(normalized_task["question"])
    if quotas == "balanced":
        count = normalized_generation["count"]
        if count < len(labels):
            raise ValidationError("generation.count must cover every label for balanced quotas")
        base, remainder = divmod(count, len(labels))
        quotas = {label: base + (index < remainder) for index, label in enumerate(labels)}
    elif quotas is not None:
        if type(quotas) is not dict or set(quotas) != set(labels):
            raise ValidationError("generation.label_quotas must specify every label exactly once")
        quotas = {label: _integer(quotas[label], f"generation.label_quotas.{label}", 0)
                  for label in labels}
        if sum(quotas.values()) != normalized_generation["count"]:
            raise ValidationError("generation.label_quotas must sum to generation.count")
    normalized_generation["label_quotas"] = quotas
    return {"version": 1, "teacher": normalized_teacher, "task": normalized_task,
            "generation": normalized_generation}


def load_config(path):
    """Load YAML without duplicate keys, then apply the same strict validation."""
    try:
        raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"Cannot read synthesis config: {exc}") from exc
    return validate_config(raw)
