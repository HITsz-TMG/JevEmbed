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


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def score_plan(plan, vectors, config, confidence_estimator=normalized_entropy):
    query = vectors[0]
    similarities = [min(1.0, max(-1.0, math.fsum(a*b for a, b in zip(query, vector)))) for vector in vectors[1:]]
    if plan.kind == "noul":
        parameters = getattr(config, plan.path)
        value = similarities[0] if plan.path == "noul_without_criteria" else similarities[0] - similarities[1]
        answer = {"type": "noul", "noul": sigmoid(parameters.slope * value + parameters.intercept)}
    else:
        probabilities = softmax(similarities, getattr(config, f"{plan.kind}_temperature"))
        confidence = float(confidence_estimator(probabilities))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise BackendError("ConfidenceEstimator must return a finite value in [0, 1]")
        answer = {"type": plan.kind, "probabilities": dict(zip(plan.labels, probabilities)), "confidence": confidence}
        if plan.kind == "choice":
            answer["choice"] = plan.labels[max(range(len(probabilities)), key=probabilities.__getitem__)]
        else:
            answer.update(score=math.fsum(i*p for i, p in enumerate(probabilities)), legend=plan.legend)
    return answer, {"path": plan.path, "similarities": similarities}
