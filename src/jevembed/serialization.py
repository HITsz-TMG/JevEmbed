import json
import math

from .errors import ValidationError

SERIALIZATION_VERSION = "configurable-text-v2"


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


def serialize_for_mode(value, mode):
    if mode == "compact_json":
        return serialize(value)
    if mode != "prose":
        raise ValidationError(f"Unknown serialization mode: {mode}")
    validate_json(value)

    def render(item, indent=0):
        if item is None:
            return ""
        if isinstance(item, str):
            return item
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, (int, float)):
            return str(item)
        padding = " " * indent
        if isinstance(item, dict):
            lines = []
            for key, entry in item.items():
                if isinstance(entry, (dict, list)) and entry:
                    lines.append(f"{padding}{key}:\n{render(entry, indent + 2)}")
                else:
                    lines.append(f"{padding}{key}: {render(entry)}")
            return ("\n\n" if indent == 0 else "\n").join(lines)
        lines = []
        for entry in item:
            if isinstance(entry, (dict, list)) and entry:
                lines.append(f"{padding}-\n{render(entry, indent + 2)}")
            else:
                lines.append(f"{padding}- {render(entry)}")
        return "\n".join(lines)

    return render(value)
