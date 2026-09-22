import json
import math

from .errors import ValidationError

SERIALIZATION_VERSION = "compact-sorted-unicode-v1"


def validate_json(value):
    def visit(v, ancestors):
        if v is None or type(v) in (str, bool, int):
            return
        if type(v) is float and math.isfinite(v):
            return
        if type(v) not in (dict, list) or id(v) in ancestors:
            raise ValidationError("Expected finite, acyclic JSON with string object keys")
        ancestors.add(id(v))
        if isinstance(v, dict):
            if any(type(k) is not str for k in v):
                raise ValidationError("JSON object keys must be strings")
            values = v.values()
        else:
            values = v
        for item in values:
            visit(item, ancestors)
        ancestors.remove(id(v))
    try:
        visit(value, set())
    except RecursionError as exc:
        raise ValidationError("JSON is too deeply nested") from exc


def serialize(value):
    validate_json(value)
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
