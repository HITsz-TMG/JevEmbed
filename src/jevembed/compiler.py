from dataclasses import dataclass

from .backends.base import EmbeddingInput
from .serialization import serialize_for_mode


@dataclass
class TaskPlan:
    question_id: str
    kind: str
    path: str
    inputs: list[EmbeddingInput]
    labels: list[str]
    legend: dict


def compile_request(request, prompts):
    def formatted(value):
        text = serialize_for_mode(value, prompts.serialization_mode)
        return text.strip() if prompts.strip_rendered else text

    state = formatted(request["state"])
    plans = []
    for qid, question in request["questions"].items():
        kind = question["type"]
        instruction = formatted(question["instructions"])
        criteria = question.get("criteria")
        labels, legend = [], {}
        if kind == "noul":
            has_criteria = "criteria" in question
            if prompts.noul_input_mapping == "binary_candidates":
                labels = ["true", "false"]
                inputs = [EmbeddingInput("query", instruction, state)]
                for label in labels:
                    if has_criteria and criteria[label] not in (None, ""):
                        description = formatted(criteria[label])
                    else:
                        fallback = getattr(prompts, f"noul_{label}_fallback")
                        description = fallback.format(instruction=instruction) if instruction else label
                    text = prompts.noul_candidate_template.format(label=label, text=description)
                    inputs.append(EmbeddingInput("document", "", text))
                path = "noul_with_criteria" if has_criteria else "noul_without_criteria"
            elif has_criteria:
                labels = ["true", "false"]
                inputs = [EmbeddingInput("query", prompts.similarity_instruction, f"{instruction}\n{state}")] + [
                    EmbeddingInput("query", prompts.similarity_instruction,
                                   prompts.noul_candidate_template.format(label=label, text=formatted(criteria[label])))
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
                texts = [name if value is None or (prompts.choice_empty_uses_label and value == "") else
                         prompts.choice_candidate_template.format(label=name, text=formatted(value))
                         for name, value in criteria.items()]
            elif kind == "score":
                labels = [str(i) for i in range(len(criteria))]
                texts = [formatted(value) for value in criteria]
                legend = dict(zip(labels, criteria))
            inputs = [EmbeddingInput("query", instruction, state)] + [EmbeddingInput("document", "", text) for text in texts]
        plans.append(TaskPlan(qid, kind, path, inputs, labels, legend))
    return plans
