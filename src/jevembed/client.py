from dataclasses import asdict, dataclass, field, replace
from contextlib import nullcontext
import logging
import hashlib
import json
import math
import threading
import time

from .backends import (ClefGemmaDecisionBackend, GemmaTypedDecisionBackend, HTTPEmbeddingBackend,
                       JevaDecisionBackend, KananaDecideBackend,
                       PairedProjectionBackend, SentenceTransformersBackend)
from .backends.embeddinggemma_decision import compile_clef_request, compile_gemma_request
from .backends.kanana_decide import compile_kanana_request
from .backends.clm_qwen35 import compile_clm_request
from .backends.jeva_decide import compile_jeva_request
from .cache import EmbeddingCache
from .compiler import compile_request
from .config import ModelConfig
from .errors import BackendError, ValidationError
from .prompts import PromptAdapter
from .schemas import validate_request
from .scoring import normalize_vectors, score_plan
from .serialization import SERIALIZATION_VERSION


@dataclass
class _Model:
    config: ModelConfig
    backend: object
    cache: EmbeddingCache
    lock: object = field(default_factory=threading.RLock)
    dimension: int | None = None
    cache_generation: int = 0


class JevEmbed:
    def __init__(self, backend=None, model=None, config=None, *, confidence_estimator=None):
        self._models, self._aliases = {}, {}
        self.default_model = model or (config.model_id if config else None)
        self.confidence_estimator = confidence_estimator
        if backend is not None or config is not None:
            config = config or getattr(backend, "config", None) or ModelConfig(model_id=model or "custom", backend="custom")
            if model and model != config.model_id:
                config = replace(config, model_id=model)
            self.register(config, backend=backend)

    def register(self, config, *, backend=None, aliases=()):
        names = (config.model_id, *config.aliases, *aliases)
        if len(set(names)) != len(names) or any(name in self._aliases for name in names):
            raise ValidationError("Model IDs/aliases must be unique; create a new client to replace a model")
        if backend is None:
            if config.backend == "sentence_transformers":
                backend = SentenceTransformersBackend(config=config)
            elif config.backend == "paired_projection":
                backend = PairedProjectionBackend(config=config)
            elif config.backend == "kanana_decide":
                backend = KananaDecideBackend(config=config)
            elif config.backend == "jeva_decide":
                backend = JevaDecisionBackend(config=config)
            elif config.backend == "gemma_typed_decision":
                backend = GemmaTypedDecisionBackend(config=config)
            elif config.backend == "clef_gemma_decision":
                backend = ClefGemmaDecisionBackend(config=config)
            elif config.backend == "http":
                backend = HTTPEmbeddingBackend(config)
            else:
                raise ValidationError("custom backend must be supplied")
        backend_config = getattr(backend, "config", None)
        if backend_config is not None:
            supplied, registered = asdict(backend_config), asdict(config)
            for key in ("model_id", "scoring", "cache_capacity", "aliases"):
                supplied.pop(key)
                registered.pop(key)
            if supplied != registered:
                raise ValidationError("Backend and registered encoding configuration must match")
        self._models[config.model_id] = _Model(config, backend, EmbeddingCache(config.cache_capacity))
        self._aliases.update({name: config.model_id for name in names})
        self.default_model = self.default_model or config.model_id
        return self

    def models(self):
        return [{"id": model, "aliases": [a for a, m in self._aliases.items() if m == model and a != model]}
                for model in self._models]

    def _prepare(self, request):
        name = validate_request(request, self.default_model)
        if name not in self._aliases:
            raise ValidationError(f"Unknown model: {name}")
        entry = self._models[self._aliases[name]]
        if entry.config.backend == "kanana_decide":
            plans = compile_kanana_request(request, entry.config.prompts)
        elif entry.config.backend == "jeva_decide":
            plans = compile_jeva_request(request, entry.config.prompts)
        elif entry.config.backend == "gemma_typed_decision":
            plans = compile_gemma_request(request, entry.config.prompts)
        elif entry.config.backend == "clef_gemma_decision":
            plans = compile_clef_request(request, entry.config.prompts)
        elif entry.config.prompts.version == "clm-qwen35-chat-options-v1":
            plans = compile_clm_request(request, entry.config.prompts)
        else:
            plans = compile_request(request, entry.config.prompts)
        return entry, plans

    def _explain(self, entry, plans):
        adapter = PromptAdapter(entry.config.prompts)
        diagnostics = adapter.diagnostics()
        if entry.config.prompts.version == "clm-qwen35-chat-options-v1":
            diagnostics.update(renderer="qwen3.5_chat_options", state_and_candidates_encoded_separately=True,
                               answer_position_pooling=True)
        if entry.config.backend in ("jeva_decide", "gemma_typed_decision", "clef_gemma_decision"):
            if entry.config.backend == "gemma_typed_decision":
                renderer = "separate_state_question_option_pooling"
            elif entry.config.backend == "clef_gemma_decision":
                renderer = "single_sequence_question_option_span_pooling"
            else:
                renderer = "native_jeva_decision_api"
            return {"model": entry.config.model_id, "config": asdict(entry.config),
                    "serialization_version": SERIALIZATION_VERSION,
                    "prompt_diagnostics": {**diagnostics, "renderer": renderer,
                                           "vector_similarity_scoring": False},
                    "tasks": [{"question_id": p.question_id, "path": p.path, "labels": p.labels,
                               "candidate_count": len(p.labels), "native_decision_head": True}
                              for p in plans]}
        if entry.config.backend == "kanana_decide":
            diagnostics.update(instruction_preserved=True, renderer="checkpoint_native",
                               query_format="state/question JSON payload; native chat template applied at inference")
        return {"model": entry.config.model_id, "config": asdict(entry.config),
                "serialization_version": SERIALIZATION_VERSION, "prompt_diagnostics": diagnostics,
                "tasks": [{"question_id": p.question_id, "path": p.path, "labels": p.labels,
                           "inputs": [{**asdict(i), "rendered": adapter.render(i)} for i in p.inputs]} for p in plans]}

    def explain(self, request):
        return self._explain(*self._prepare(request))

    def evaluate(self, request):
        return self._evaluate(request, include_trace=False)

    def system_one(self, *, state, questions, model=None):
        request = {"state": state, "questions": questions}
        if model is not None:
            request["model"] = model
        return self.evaluate(request)

    def clear_cache(self, model=None):
        entries = list(self._models.values()) if model is None else [self._models[self._aliases[model]]]
        for entry in entries:
            with entry.lock:
                entry.cache.clear()
                entry.cache_generation += 1
            if hasattr(entry.backend, "clear_cache"):
                entry.backend.clear_cache()

    def evaluate_with_trace(self, request):
        return self._evaluate(request, include_trace=True)

    def _evaluate(self, request, *, include_trace):
        started = time.perf_counter()
        entry, plans = self._prepare(request)
        cfg = entry.config
        if cfg.backend in ("jeva_decide", "gemma_typed_decision", "clef_gemma_decision"):
            return self._evaluate_native_decision(request, entry, plans, include_trace, started)
        unique = list(dict.fromkeys(item for plan in plans for item in plan.inputs))
        vectors, sources, batches = {}, [], []
        truncations = {}
        usage, hits = 0, 0
        # Built-in backends protect their own mutable state. Local encode calls
        # serialize inside SentenceTransformersBackend; HTTP calls may overlap.
        # Unknown custom backends retain whole-request serialization.
        built_in = isinstance(entry.backend, (HTTPEmbeddingBackend, JevaDecisionBackend, GemmaTypedDecisionBackend,
                                             ClefGemmaDecisionBackend, KananaDecideBackend,
                                             PairedProjectionBackend, SentenceTransformersBackend))
        with (nullcontext() if built_in else entry.lock):
            provenance = entry.backend.prepare() if hasattr(entry.backend, "prepare") else {}
            identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
            prefix = (cfg.fingerprint(), SERIALIZATION_VERSION, identity)
            missing = []
            with entry.lock:
                cache_generation = entry.cache_generation
                for item in unique:
                    cached = entry.cache.get((prefix, item))
                    if cached is None:
                        missing.append(item)
                    else:
                        vectors[item] = cached["vector"]
                        if cached["truncation"]:
                            truncations[item] = cached["truncation"]
                        hits += 1
            # Cache writes are committed only after the entire request has succeeded.
            for start in range(0, len(missing), cfg.batch_size):
                items = missing[start:start+cfg.batch_size]
                try:
                    batch = entry.backend.encode(items)
                except (BackendError, ValidationError):
                    raise
                except Exception as exc:
                    raise BackendError(f"Backend encode failed: {exc}") from exc
                with entry.lock:
                    normalized = normalize_vectors(batch.vectors, len(items), entry.dimension or cfg.expected_dimension)
                    if normalized:
                        entry.dimension = len(normalized[0])
                token_count = batch.input_tokens
                source = batch.usage_source
                if token_count is None:
                    if cfg.usage_mode == "strict":
                        raise BackendError("Backend did not return exact input token usage")
                    adapter = PromptAdapter(cfg.prompts)
                    token_count = sum(math.ceil(len(adapter.render(i).encode("utf-8"))/4) for i in items)
                    source = "estimated_utf8_bytes_div4"
                    logging.getLogger(__name__).warning("Token usage estimated as ceil(UTF-8 bytes / 4)")
                if type(token_count) is not int or token_count < 0:
                    raise BackendError("Backend input_tokens must be a nonnegative integer")
                usage += token_count
                if include_trace:
                    sources.append(source)
                    batches.append({"inputs": [asdict(item) for item in items], "metadata": batch.metadata})
                for record in batch.metadata.get("truncated", []):
                    truncations[items[record["index"]]] = {k: v for k, v in record.items() if k != "index"}
                vectors.update(zip(items, normalized))
            answers, diagnostics = {}, {}
            scoring = entry.backend.scoring_config() if hasattr(entry.backend, "scoring_config") else cfg.scoring
            for plan in plans:
                answer, diagnostic = score_plan(
                    plan, [vectors[item] for item in plan.inputs], scoring, self.confidence_estimator)
                answers[plan.question_id] = answer
                if include_trace:
                    diagnostics[plan.question_id] = diagnostic
            with entry.lock:
                # A concurrent clear_cache() must not be undone by an in-flight request.
                if entry.cache_generation == cache_generation:
                    for item in missing:
                        entry.cache.put((prefix, item), {"vector": vectors[item], "truncation": truncations.get(item)})
        response = {"model": cfg.model_id, "answers": answers,
                    "usage": {"input_tokens": usage, "output_tokens": 0}}
        if not include_trace:
            return response
        trace = self._explain(entry, plans)
        trace.update(questions=diagnostics, cache_hits=hits, encoded_inputs=len(missing),
                     usage_sources=sorted(set(sources)) or ["cache"], batches=batches,
                     backend_metadata=getattr(entry.backend, "metadata", {}),
                     truncated_inputs=[{"input": asdict(item), **info} for item, info in truncations.items()],
                     elapsed_seconds=time.perf_counter()-started)
        return {"response": response, "trace": trace}

    def _evaluate_native_decision(self, request, entry, plans, include_trace, started):
        provenance = entry.backend.prepare()
        response, questions = entry.backend.evaluate(request, plans)
        if not include_trace:
            return response
        trace = self._explain(entry, plans)
        trace.update(questions=questions, cache_owner="native_decision_model",
                     usage_sources=["native_decision_backend"], batches=[],
                     backend_metadata=provenance,
                     elapsed_seconds=time.perf_counter() - started)
        return {"response": response, "trace": trace}
