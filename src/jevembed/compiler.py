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
                if prompts.noul_candidate_order == "request" and has_criteria:
                    labels = list(criteria)
                else:
                    labels = (["true", "false"] if prompts.noul_candidate_order == "true_false"
                              else ["false", "true"])
                inputs = [EmbeddingInput("query", instruction, state)]
                for label in labels:
                    value = criteria[label] if has_criteria else None
                    if has_criteria and value is None and prompts.noul_null_uses_text:
                        description = formatted(value)
                    elif has_criteria and value not in (None, ""):
                        description = formatted(value)
                    else:
                        fallback = getattr(prompts, f"noul_{label}_fallback")
                        description = fallback.format(instruction=instruction) if instruction else label
                    if has_criteria and criteria[label] == "" and prompts.noul_empty_uses_label:
                        text = label
                    elif (has_criteria and prompts.noul_matching_label_uses_label
                          and description == label):
                        text = label
                    else:
                        text = prompts.noul_candidate_template.format(label=label, text=description)
                    inputs.append(EmbeddingInput("document", "", text))
                path = "noul_with_criteria" if has_criteria else "noul_without_criteria"
            elif has_criteria:
                labels = (["true", "false"] if prompts.noul_candidate_order == "true_false"
                          else ["false", "true"])
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
                texts = []
                for name, value in criteria.items():
                    if value is None:
                        text = name
                    elif value == "" and prompts.choice_empty_uses_label:
                        text = name
                    else:
                        text = prompts.choice_candidate_template.format(label=name, text=formatted(value))
                    texts.append(text)
            elif kind == "score":
                labels = [str(i) for i in range(len(criteria))]
                texts = [prompts.score_candidate_template.format(label=label, max_label=labels[-1], text=formatted(value))
                         for label, value in zip(labels, criteria, strict=True)]
                legend = dict(zip(labels, criteria))
            inputs = [EmbeddingInput("query", instruction, state)] + [EmbeddingInput("document", "", text) for text in texts]
        plans.append(TaskPlan(qid, kind, path, inputs, labels, legend))
    return plans
