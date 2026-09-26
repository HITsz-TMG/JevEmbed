"""Canonical label order and hard answers for synthesis quotas."""


def quota_labels(question):
    """Return the complete ordered label vocabulary for one fixed question."""
    kind = question["type"]
    if kind == "choice":
        return list(question["criteria"])
    if kind == "score":
        return [str(index) for index in range(len(question["criteria"]))]
    return ["true", "false"]


def quota_answers(config):
    """Return ordered ``(hard answer, target count)`` pairs, or None if unconstrained."""
    quotas = config["generation"]["label_quotas"]
    if quotas is None:
        return None
    question = config["task"]["question"]
    kind = question["type"]
    result = []
    for label in quota_labels(question):
        if kind == "choice":
            answer = {"choice": label}
        elif kind == "score":
            answer = {"level": int(label)}
        else:
            answer = {"noul": label == "true"}
        result.append((answer, quotas[label]))
    return result
