from .errors import ValidationError
from .serialization import validate_json


def _content(value, path):
    if type(value) not in (str, dict, list):
        raise ValidationError(f"{path}: expected string, object or array")


def validate_request(request, default_model=None):
    validate_json(request)
    if not isinstance(request, dict) or set(request) - {"model", "state", "questions"}:
        raise ValidationError("Request must contain only model, state, questions")
    model = request.get("model", default_model)
    if not isinstance(model, str) or not model:
        raise ValidationError("model is required")
    if "state" not in request:
        raise ValidationError("state is required")
    _content(request["state"], "state")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValidationError("questions must be a nonempty object")
    for qid, q in questions.items():
        if not isinstance(q, dict) or set(q) - {"type", "instructions", "criteria"}:
            raise ValidationError(f"questions.{qid}: invalid question fields")
        _content(q.get("instructions"), f"questions.{qid}.instructions")
        kind, criteria = q.get("type"), q.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ValidationError(f"{qid}: Choice requires 1..255 candidates")
            for value in criteria.values():
                if value is not None:
                    _content(value, f"{qid}.criteria")
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValidationError(f"{qid}: Score requires 2..10 ordered levels")
            for value in criteria:
                _content(value, f"{qid}.criteria")
        elif kind == "noul":
            if "criteria" in q:
                if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
                    raise ValidationError(f"{qid}: omit Noul criteria or supply both true and false")
                for value in criteria.values():
                    _content(value, f"{qid}.criteria")
        else:
            raise ValidationError(f"{qid}: unknown question type {kind!r}")
    return model
