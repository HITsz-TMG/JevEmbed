from dataclasses import asdict, dataclass, field, replace
import logging
import hashlib
import json
import math
import threading
import time

from .backends import HTTPEmbeddingBackend, SentenceTransformersBackend
from .cache import EmbeddingCache
from .compiler import compile_request
from .config import ModelConfig
from .errors import BackendError, ValidationError
from .prompts import PromptAdapter
from .schemas import validate_request
from .scoring import normalize_vectors, normalized_entropy, score_plan
from .serialization import SERIALIZATION_VERSION


@dataclass
class _Model:
    config: ModelConfig
    backend: object
    cache: EmbeddingCache
    lock: object = field(default_factory=threading.RLock)
    dimension: int | None = None


class JevEmbed:
    def __init__(self, backend=None, model=None, config=None, *, confidence_estimator=normalized_entropy):
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
        return entry, compile_request(request, entry.config.prompts)

    def _explain(self, entry, plans):
        adapter = PromptAdapter(entry.config.prompts)
        return {"model": entry.config.model_id, "config": asdict(entry.config),
                "serialization_version": SERIALIZATION_VERSION, "prompt_diagnostics": adapter.diagnostics(),
                "tasks": [{"question_id": p.question_id, "path": p.path, "labels": p.labels,
                           "inputs": [{**asdict(i), "rendered": adapter.render(i)} for i in p.inputs]} for p in plans]}

    def explain(self, request):
        return self._explain(*self._prepare(request))

    def evaluate(self, request):
        return self.evaluate_with_trace(request)["response"]

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

    def evaluate_with_trace(self, request):
        started = time.perf_counter()
        entry, plans = self._prepare(request)
        cfg = entry.config
        unique = list(dict.fromkeys(item for plan in plans for item in plan.inputs))
        vectors, sources, batches = {}, [], []
        truncations = {}
        usage, hits = 0, 0
        with entry.lock:
            provenance = entry.backend.prepare() if hasattr(entry.backend, "prepare") else {}
            identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
            prefix = (cfg.fingerprint(), SERIALIZATION_VERSION, identity)
            missing = []
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
                sources.append(source)
                batches.append({"inputs": [asdict(item) for item in items], "metadata": batch.metadata})
                for record in batch.metadata.get("truncated", []):
                    truncations[items[record["index"]]] = {k: v for k, v in record.items() if k != "index"}
                vectors.update(zip(items, normalized))
            answers, diagnostics = {}, {}
            for plan in plans:
                answers[plan.question_id], diagnostics[plan.question_id] = score_plan(
                    plan, [vectors[item] for item in plan.inputs], cfg.scoring, self.confidence_estimator)
            for item in missing:
                entry.cache.put((prefix, item), {"vector": vectors[item], "truncation": truncations.get(item)})
        response = {"model": cfg.model_id, "answers": answers,
                    "usage": {"input_tokens": usage, "output_tokens": 0}}
        trace = self._explain(entry, plans)
        trace.update(questions=diagnostics, cache_hits=hits, encoded_inputs=len(missing),
                     usage_sources=sorted(set(sources)) or ["cache"], batches=batches,
                     backend_metadata=getattr(entry.backend, "metadata", {}),
                     truncated_inputs=[{"input": asdict(item), **info} for item, info in truncations.items()],
                     elapsed_seconds=time.perf_counter()-started)
        return {"response": response, "trace": trace}
