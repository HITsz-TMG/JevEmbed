"""Bridge JevEmbed requests to Jeva's released set-aware decision head."""

import importlib.metadata
import threading
from dataclasses import replace

from ..compiler import compile_request
from ..errors import BackendError, ValidationError
from ..scoring import decision_response
from .runtime import resolve_device


def compile_jeva_request(request, prompts):
    plans = compile_request(request, prompts)
    for plan, (question_id, question) in zip(plans, request["questions"].items(), strict=True):
        plan.path = "native_jeva"
        instruction = plan.inputs[0].instruction
        if plan.kind == "noul" and "criteria" in question:
            for index, (label, item) in enumerate(zip(plan.labels, plan.inputs[1:], strict=True), start=1):
                if question["criteria"][label] is None:
                    plan.inputs[index] = replace(item, text="null")
        texts = [item.text for item in plan.inputs[1:]]
        if not instruction.strip() or any(not text.strip() for text in texts):
            raise ValidationError("Jeva requires nonempty instructions and candidate texts")
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

            device = resolve_device(cfg, torch)
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
            predictions, truncations, tokens = {}, {}, 0
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
                if type(native.usage.input_tokens) is not int or native.usage.input_tokens < 0:
                    raise BackendError("Jeva returned invalid input token usage")
                tokens += native.usage.input_tokens
                predictions[plan.question_id] = native.candidate_probabilities
                truncations[plan.question_id] = native.usage.truncated_tokens
            response, diagnostics = decision_response(self.config, plans, predictions, tokens, tolerance=1e-5)
            for question_id, truncated_tokens in truncations.items():
                diagnostics[question_id]["truncated_tokens"] = truncated_tokens
            return response, diagnostics
