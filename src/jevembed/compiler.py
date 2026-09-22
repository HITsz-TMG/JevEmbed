from dataclasses import dataclass

from .backends.base import EmbeddingInput
from .serialization import serialize


@dataclass
class TaskPlan:
    question_id: str
    kind: str
    path: str
    inputs: list[EmbeddingInput]
    labels: list[str]
    legend: dict


def compile_request(request, prompts):
    state = serialize(request["state"])
    plans = []
    for qid, question in request["questions"].items():
        kind = question["type"]
        instruction = serialize(question["instructions"])
        criteria = question.get("criteria")
        labels, legend = [], {}
        query = EmbeddingInput("query", instruction, state)
        if kind == "noul" and "criteria" not in question:
            inputs = [EmbeddingInput("query", prompts.similarity_instruction, state),
                      EmbeddingInput("query", prompts.similarity_instruction, instruction)]
            path = "noul_similarity"
        else:
            path = "noul_criteria" if kind == "noul" else kind
            if kind == "choice":
                labels = list(criteria)
                texts = [name if value is None else f"{name}: {serialize(value)}" for name, value in criteria.items()]
            elif kind == "score":
                labels = [str(i) for i in range(len(criteria))]
                texts = [serialize(value) for value in criteria]
                legend = dict(zip(labels, criteria))
            else:
                labels = ["true", "false"]
                texts = [serialize(criteria[label]) for label in labels]
            inputs = [query] + [EmbeddingInput("document", "", text) for text in texts]
        plans.append(TaskPlan(qid, kind, path, inputs, labels, legend))
    return plans
