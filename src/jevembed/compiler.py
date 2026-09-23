from dataclasses import dataclass

from .backends.base import EmbeddingInput
from .serialization import serialize


_NOUL_FALLBACK = {
    "true": "The answer to the question is yes.",
    "false": "The answer to the question is no.",
}


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
        if kind == "noul" and prompts.noul_format == "retrieval":
            state_query = EmbeddingInput("query", prompts.similarity_instruction, state)
            if "criteria" in question:
                labels = ["true", "false"]
                inputs = [state_query] + [EmbeddingInput("query", prompts.similarity_instruction,
                            f"{instruction}\n{serialize(criteria[label])}") for label in labels]
                path = "noul_criteria"
            else:
                inputs = [EmbeddingInput("query", prompts.similarity_instruction, instruction), state_query]
                path = "noul_similarity"
        elif kind == "noul" and "criteria" not in question and prompts.noul_format == "legacy":
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
                source = _NOUL_FALLBACK if criteria is None else criteria
                texts = [serialize(source[label]) for label in labels]
            inputs = [query] + [EmbeddingInput("document", "", text) for text in texts]
        plans.append(TaskPlan(qid, kind, path, inputs, labels, legend))
    return plans
