import math
from typing import Protocol

from .errors import BackendError


class ConfidenceEstimator(Protocol):
    def __call__(self, probabilities: list[float]) -> float: ...


def normalize_vectors(vectors, count, expected_dimension=None):
    if len(vectors) != count:
        raise BackendError(f"Expected {count} vectors, received {len(vectors)}")
    result, dimension = [], expected_dimension
    for vector in vectors:
        try:
            values = [float(v) for v in vector]
        except (TypeError, ValueError) as exc:
            raise BackendError("Invalid vector") from exc
        if not values or not all(math.isfinite(v) for v in values):
            raise BackendError("Empty or nonfinite vector")
        dimension = dimension or len(values)
        if len(values) != dimension:
            raise BackendError("Embedding dimensions differ")
        scale = max(map(abs, values))
        if scale == 0:
            raise BackendError("Zero embedding vector")
        scaled = [v / scale for v in values]
        norm = math.sqrt(math.fsum(v*v for v in scaled))
        result.append([v/norm for v in scaled])
    return result


def softmax(similarities, temperature):
    maximum = max(similarities)
    # Subtract before dividing: even subnormal positive temperatures are safe.
    values = [math.exp((s - maximum) / temperature) for s in similarities]
    total = math.fsum(values)
    return [v / total for v in values]


def normalized_entropy(probabilities):
    if len(probabilities) == 1:
        return 1.0
    entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0)
    return min(1.0, max(0.0, 1 - entropy / math.log(len(probabilities))))


def top_margin(probabilities):
    if len(probabilities) < 2:
        return 1.0
    top = max(probabilities)
    return min(1.0, max(0.0, top - (math.fsum(probabilities) - top) / (len(probabilities) - 1)))


def decision_answer(plan, probabilities, expected_score=None, *, tolerance=1e-3):
    """Map a native head's candidate probabilities to the shared answer schema."""
    if isinstance(probabilities, dict):
        values = [float(probabilities.get(label, float("nan"))) for label in plan.labels]
    else:
        values = [float(value) for value in probabilities]
    if (len(values) != len(plan.labels) or
        not all(math.isfinite(value) and 0 <= value <= 1 for value in values) or
        not math.isclose(math.fsum(values), 1.0, abs_tol=tolerance)):
        raise BackendError("Decision head returned invalid probabilities")
    total = math.fsum(values)
    values = [value / total for value in values]
    distribution = dict(zip(plan.labels, values, strict=True))
    if plan.kind == "noul":
        if "true" not in distribution:
            raise BackendError("Noul decision output has no true probability")
        answer = {"type": "noul", "noul": distribution["true"]}
    else:
        answer = {"type": plan.kind, "probabilities": distribution, "confidence": max(values)}
        if plan.kind == "choice":
            answer["choice"] = plan.labels[max(range(len(values)), key=values.__getitem__)]
        else:
            score = (math.fsum(i * probability for i, probability in enumerate(values))
                     if expected_score is None else float(expected_score))
            if not math.isfinite(score):
                raise BackendError("Decision head returned a non-finite expected score")
            answer.update(score=score, legend=plan.legend)
    return answer, distribution


def decision_response(config, plans, predictions, input_tokens, *, tolerance=1e-3):
    """Return JevEmbed's standard response and diagnostics for native heads."""
    if type(input_tokens) is not int or input_tokens < 0:
        raise BackendError("Decision backend returned invalid input token usage")
    answers, diagnostics = {}, {}
    for plan in plans:
        try:
            prediction = predictions[plan.question_id]
            probabilities = getattr(prediction, "probabilities", prediction)
            expected_score = getattr(prediction, "expected_level", None)
            answer, distribution = decision_answer(
                plan, probabilities, expected_score, tolerance=tolerance)
        except BackendError as exc:
            raise BackendError(f"Invalid decision output for {plan.question_id}: {exc}") from exc
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise BackendError(f"Invalid decision output for {plan.question_id}: {exc}") from exc
        answers[plan.question_id] = answer
        diagnostics[plan.question_id] = {"path": plan.path, "probabilities": distribution}
    response = {"model": config.model_id, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
    return response, diagnostics


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def score_plan(plan, vectors, config, confidence_estimator=None):
    query = vectors[0]
    similarities = [min(1.0, max(-1.0, math.fsum(a*b for a, b in zip(query, vector)))) for vector in vectors[1:]]
    if plan.kind == "noul":
        if plan.path == "clm_noul":
            parameters = config.noul_without_criteria
            value = similarities[1] - similarities[0]
        else:
            parameters = getattr(config, plan.path)
            value = similarities[0] if len(similarities) == 1 else similarities[0] - similarities[1]
        answer = {"type": "noul", "noul": sigmoid(parameters.slope * value + parameters.intercept)}
    else:
        probabilities = softmax(similarities, getattr(config, f"{plan.kind}_temperature"))
        estimator = confidence_estimator or (top_margin if config.confidence == "top_margin" else normalized_entropy)
        confidence = float(estimator(probabilities))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise BackendError("ConfidenceEstimator must return a finite value in [0, 1]")
        answer = {"type": plan.kind, "probabilities": dict(zip(plan.labels, probabilities)), "confidence": confidence}
        if plan.kind == "choice":
            answer["choice"] = plan.labels[max(range(len(probabilities)), key=probabilities.__getitem__)]
        else:
            answer.update(score=math.fsum(i*p for i, p in enumerate(probabilities)), legend=plan.legend)
    return answer, {"path": plan.path, "similarities": similarities}
