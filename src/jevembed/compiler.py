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
        if kind == "noul":
            if "criteria" in question:
                labels = ["true", "false"]
                inputs = [EmbeddingInput("query", prompts.similarity_instruction, f"{instruction}\n{state}")] + [
                    EmbeddingInput("query", prompts.similarity_instruction, f"{label}: {serialize(criteria[label])}")
                    for label in labels
                ]
                path = "noul_with_criteria"
            else:
                inputs = [EmbeddingInput("query", prompts.similarity_instruction, instruction),
                          EmbeddingInput("query", prompts.similarity_instruction, state)]
                path = "noul_without_criteria"
        else:
            path = kind
            if kind == "choice":
                labels = list(criteria)
                texts = [name if value is None else f"{name}: {serialize(value)}" for name, value in criteria.items()]
            elif kind == "score":
                labels = [str(i) for i in range(len(criteria))]
                texts = [serialize(value) for value in criteria]
                legend = dict(zip(labels, criteria))
            inputs = [EmbeddingInput("query", instruction, state)] + [EmbeddingInput("document", "", text) for text in texts]
        plans.append(TaskPlan(qid, kind, path, inputs, labels, legend))
    return plans
