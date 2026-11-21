"""Bridge JevEmbed requests to Jeva's released set-aware decision head."""

import importlib.metadata
import json
import math
import threading

from .base import EmbeddingInput
from ..errors import BackendError, ValidationError


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def compile_jeva_request(request, prompts):
    from ..compiler import TaskPlan

    plans = []
    for qid, question in request["questions"].items():
        kind, criteria = question["type"], question.get("criteria")
        if kind == "choice":
            labels = list(criteria)
            texts = [label if criteria[label] in (None, "") else f"{label}: {_text(criteria[label])}" for label in labels]
        elif kind == "score":
            labels = [str(i) for i in range(len(criteria))]
            texts = [_text(value) for value in criteria]
        else:
            labels = ["false", "true"]
            texts = [label if criteria is None or criteria[label] == "" else _text(criteria[label]) for label in labels]
        instruction = _text(question["instructions"])
        if not instruction.strip() or any(not text.strip() for text in texts):
            raise ValidationError("Jeva requires nonempty instructions and candidate texts")
        inputs = [EmbeddingInput("query", instruction, _text(request["state"]))]
        inputs.extend(EmbeddingInput("document", "", value) for value in texts)
        legend = dict(zip(labels, criteria)) if kind == "score" else {}
        plans.append(TaskPlan(qid, kind, "native_jeva", inputs, labels, legend))
    return plans


class JevaDecisionBackend:
    def __init__(self, *, config, model=None):
        self.config = config
        if config.backend != "jeva_decide":
            raise ValidationError("JevaDecisionBackend requires backend=jeva_decide")
        self._model = model
        self._lock = threading.RLock()
        self.metadata = {}

    def prepare(self):
        with self._lock:
            if self._model is None:
                self._load()
            return self.metadata

    def _load(self):
        cfg = self.config
        try:
            import torch
            from jeva import Jeva

            device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
            options = dict(device=device, revision=cfg.revision, local_files_only=cfg.local_files_only,
                           trust_remote_code=False, cache="hybrid" if cfg.cache_capacity else "none",
                           option_cache_size=max(1, cfg.cache_capacity), option_chunk_size=cfg.batch_size,
                           truncation=cfg.overflow_policy)
            if cfg.max_input_tokens != "model_default":
                options.update(max_state_length=cfg.max_input_tokens, max_option_length=min(1024, cfg.max_input_tokens))
            if cfg.dtype != "auto":
                options["dtype"] = cfg.dtype
            model = Jeva.from_pretrained(cfg.model_name_or_path, **options)
            self.metadata = {
                "model_name_or_path": cfg.model_name_or_path, "requested_revision": cfg.revision,
                "checkpoint_dir": str(model.checkpoint_dir), "device": device,
                "decision_head": "native matrix stack", "versions": {
                    name: importlib.metadata.version(name) for name in ("jeva-model", "torch", "transformers")}}
        except ImportError as exc:
            raise BackendError("Install jevembed[jeva] in a dedicated Python 3.12 environment") from exc
        except Exception as exc:
            raise BackendError(f"Cannot load native Jeva checkpoint {cfg.model_name_or_path}: {exc}") from exc
        self._model = model

    def clear_cache(self):
        with self._lock:
            if self._model is not None:
                self._model.clear_cache()

    def evaluate(self, request, plans):
        with self._lock:
            self.prepare()
            answers, diagnostics, tokens = {}, {}, 0
            for plan in plans:
                try:
                    native = self._model.predict(
                        state=request["state"], question=plan.inputs[0].instruction,
                        decision_type=plan.kind, options=[item.text for item in plan.inputs[1:]],
                        allow_null=False, positive_index=1)
                except ValueError as exc:
                    raise ValidationError(str(exc)) from exc
                except Exception as exc:
                    raise BackendError(f"Native Jeva decision failed: {exc}") from exc
                probabilities = [float(value) for value in native.candidate_probabilities]
                if (len(probabilities) != len(plan.labels) or
                    not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities) or
                    not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)):
                    raise BackendError("Jeva returned invalid candidate probabilities")
                distribution = dict(zip(plan.labels, probabilities))
                if plan.kind == "noul":
                    answer = {"type": "noul", "noul": distribution["true"]}
                else:
                    answer = {"type": plan.kind, "probabilities": distribution,
                              "confidence": max(probabilities)}
                    if plan.kind == "choice":
                        answer["choice"] = plan.labels[max(range(len(probabilities)), key=probabilities.__getitem__)]
                    else:
                        answer.update(score=math.fsum(i * p for i, p in enumerate(probabilities)), legend=plan.legend)
                if type(native.usage.input_tokens) is not int or native.usage.input_tokens < 0:
                    raise BackendError("Jeva returned invalid input token usage")
                tokens += native.usage.input_tokens
                answers[plan.question_id] = answer
                diagnostics[plan.question_id] = {"path": "native_jeva", "probabilities": distribution,
                                                 "truncated_tokens": native.usage.truncated_tokens}
            response = {"model": self.config.model_id, "answers": answers,
                        "usage": {"input_tokens": tokens, "output_tokens": 0}}
            return response, diagnostics
