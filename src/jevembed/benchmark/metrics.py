"""Metrics shared by benchmark runs."""

import math

from ..errors import ValidationError


def _finite(value, source):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValidationError(f"Invalid model output at {source}")
    return float(value)


def _hard_index(values):
    indices = [index for index, value in enumerate(values) if value == 1.0]
    return indices[0] if len(indices) == 1 and all(value in (0.0, 1.0) for value in values) else None


class MetricAccumulator:
    def __init__(self):
        self.totals = {}
        self.task_counts = {"choice": 0, "score": 0, "noul": 0}

    def _add(self, name, value):
        total, count = self.totals.get(name, (0.0, 0))
        self.totals[name] = (total + float(value), count + 1)

    def add(self, target, answer, source):
        if not isinstance(answer, dict) or answer.get("type") != target.kind:
            raise ValidationError(f"Invalid model answer type at {source}")
        self.task_counts[target.kind] += 1
        if target.kind == "noul":
            prediction = _finite(answer.get("noul"), f"{source}.noul")
            if not 0 <= prediction <= 1:
                raise ValidationError(f"Noul output is outside [0, 1] at {source}")
            gold = target.values[0]
            error = abs(prediction - gold)
            self._add("noul_mae", error)
            self._add("target_distribution_brier", 2 * error * error)
            self._add("target_distribution_tvd", error)
            if gold in (0.0, 1.0):
                correct = (prediction >= 0.5) == bool(gold)
                self._add("noul_binary_accuracy", correct)
                self._add("overall_hard_accuracy", correct)
            return

        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(target.labels):
            raise ValidationError(f"Invalid probabilities at {source}")
        predicted = tuple(_finite(probabilities[label], f"{source}.probabilities.{label}")
                          for label in target.labels)
        if any(value < 0 or value > 1 for value in predicted) or not math.isclose(
                math.fsum(predicted), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValidationError(f"Probabilities must be in [0, 1] and sum to 1 at {source}")

        if target.mode == "distribution":
            self._add("target_distribution_brier",
                      math.fsum((actual - expected) ** 2
                                for actual, expected in zip(predicted, target.values)))
            self._add("target_distribution_tvd",
                      math.fsum(abs(actual - expected)
                                for actual, expected in zip(predicted, target.values)) / 2)
            hard = _hard_index(target.values)
            if hard is not None:
                selected = max(range(len(predicted)), key=predicted.__getitem__)
                correct = selected == hard
                name = "choice_accuracy" if target.kind == "choice" else "score_level_accuracy"
                self._add(name, correct)
                self._add("overall_hard_accuracy", correct)

        if target.kind == "score":
            prediction = _finite(answer.get("score"), f"{source}.score")
            gold = (target.values[0] if target.mode == "score" else
                    math.fsum(index * probability for index, probability in enumerate(target.values)))
            self._add("score_mae", abs(prediction - gold))

    def result(self):
        names = ("overall_hard_accuracy", "choice_accuracy", "score_level_accuracy",
                 "noul_binary_accuracy", "score_mae", "noul_mae",
                 "target_distribution_brier", "target_distribution_tvd")
        metrics = {}
        for name in names:
            total, count = self.totals.get(name, (0.0, 0))
            metrics[name] = total / count if count else None
            metrics[name + "_n"] = count
        return metrics
