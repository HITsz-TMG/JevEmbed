"""Kanana's native state/candidate encoder, with JevEmbed scoring and caching."""

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import threading
from dataclasses import replace

from .base import EmbeddingBatch, EmbeddingInput
from ..config import ModelConfig
from ..errors import BackendError, ValidationError


def _compact(value):
    # Kanana preserves JSON field order, including structured instructions.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def compile_kanana_request(request, prompts):
    """Keep the original question available to the model's native query renderer.

    This is deliberately independent of Transformers so --explain stays offline.
    Query text is a JSON transport payload, not a prompt sent to the encoder.
    """
    from ..compiler import compile_request

    plans = compile_request(request, prompts)
    for plan in plans:
        qid = plan.question_id
        if not qid:
            raise ValidationError("Kanana question IDs must be nonempty")
        if any(not label for label in plan.labels):
            raise ValidationError("Kanana candidate labels must be nonempty")
        texts = [item.text for item in plan.inputs[1:]]
        if any(not text.strip() for text in texts):
            raise ValidationError("Kanana candidate descriptions must not render empty")
        if len(set(texts)) != len(texts):
            raise ValidationError("Kanana candidates render to identical text")
        payload = _compact({"state": request["state"], "question": request["questions"][qid]})
        plan.inputs[0] = EmbeddingInput("query", "", payload)
    return plans


class KananaDecideBackend:
    """Load one pinned snapshot, including its processor and native renderer.

    The upstream embed_messages method performs last-token pooling and L2
    normalization, one input at a time. JevEmbed owns the embedding cache.
    """

    def __init__(self, model_name_or_path=None, *, config=None, **options):
        self.config = config or ModelConfig(backend="kanana_decide",
                                           model_name_or_path=model_name_or_path or "", **options)
        if config is not None and (model_name_or_path is not None or options):
            self.config = replace(config, model_name_or_path=model_name_or_path or config.model_name_or_path, **options)
        if self.config.backend != "kanana_decide":
            raise ValidationError("KananaDecideBackend requires backend=kanana_decide")
        if not self.config.trust_remote_code:
            raise ValidationError("Kanana requires explicit trust_remote_code=true")
        if self.config.code_revision and self.config.code_revision != self.config.revision:
            raise ValidationError("Kanana code_revision must match the snapshot revision")
        self._lock = threading.RLock()
        self._model = self._reference = self._processor = None
        self._limit = None
        self.metadata = {}

    def prepare(self):
        with self._lock:
            self._load()
            return self.metadata

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise BackendError("Install jevembed[kanana] in a separate environment; see docs/compatibility.md#decision-model-runtimes") from exc
        cfg = self.config
        device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise BackendError("CUDA was requested but is unavailable")
        dtype = cfg.dtype
        if dtype == "auto":
            dtype = "bfloat16" if device.startswith("cuda") and torch.cuda.is_bf16_supported() else "float32"
        if device == "cpu" and dtype != "float32":
            raise ValidationError("CPU backend requires dtype=float32 or auto")
        root = Path(cfg.model_name_or_path)
        try:
            if not root.is_dir():
                root = Path(snapshot_download(
                    cfg.model_name_or_path, revision=cfg.revision, local_files_only=cfg.local_files_only,
                    allow_patterns=["*.json", "*.py", "*.jinja", "*.safetensors"]))
            root = root.resolve()
            kwargs = {"trust_remote_code": cfg.trust_remote_code, "local_files_only": True,
                      "dtype": getattr(torch, dtype)}
            if cfg.attention_implementation:
                kwargs["attn_implementation"] = cfg.attention_implementation
            model = AutoModel.from_pretrained(str(root), **kwargs).eval().to(device)
            native_cfg = model.config
            if native_cfg.model_type != "kanana_v_decide":
                raise BackendError("Expected a kanana_v_decide checkpoint")
            dimension = int(native_cfg.embedding_dimension)
            if cfg.expected_dimension and dimension != cfg.expected_dimension:
                raise BackendError(f"Expected dimension {cfg.expected_dimension}, got {dimension}")
            limit = native_cfg.max_length if cfg.max_input_tokens == "model_default" else cfg.max_input_tokens
            if limit > native_cfg.max_length:
                raise ValidationError(f"Configured token limit exceeds model limit {native_cfg.max_length}")
            # The native processor reloads from _name_or_path; pointing at the
            # snapshot prevents a second, unpinned Hub load of its tokenizer.
            processor = model.processor
            module = model.__class__.__module__.rsplit(".", 1)[0]
            reference = importlib.import_module(f"{module}.decision_kanana_v_decide")
            files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(root.iterdir()) if path.suffix in (".py", ".json", ".jinja")}
            weights = {path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                       for path in sorted(root.glob("*.safetensors"))}
            resolved = root.name if root.parent.name == "snapshots" else getattr(native_cfg, "_commit_hash", None)
            self.metadata = {"model_name_or_path": cfg.model_name_or_path, "requested_revision": cfg.revision,
                             "resolved_revision": resolved, "files_sha256": files, "weights": weights,
                             "trust_remote_code": cfg.trust_remote_code, "device": device, "dtype": dtype,
                             "dimension": dimension, "pooling": "last_token_l2", "max_input_tokens": limit,
                             "native_decision_temperature": float(native_cfg.decision_temperature),
                             "renderer": native_cfg.renderer,
                             "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")}}
        except (BackendError, ValidationError):
            raise
        except Exception as exc:
            raise BackendError(f"Cannot load Kanana from {cfg.model_name_or_path}; install jevembed[kanana] "
                               f"in a compatible environment: {exc}") from exc
        self._model, self._reference, self._processor, self._limit = model, reference, processor, limit

    def _messages(self, item):
        ref, processor = self._reference, self._processor
        tokenizer, limit = processor.tokenizer, self._limit
        truncate = self.config.overflow_policy == "truncate"
        # Native message_ids enforces an upper bound below one million. This
        # ceiling is only for measuring the original input before shortening.
        measurement_limit = 999_999
        if item.role == "query":
            wire = json.loads(item.text)
            payload, _, candidates = ref.build_pairs(wire["state"], {"q": wire["question"]})["q"]
            ref.validate_candidates(candidates, tokenizer, processor, limit, truncate=truncate)
            def render(maximum, shorten):
                return ref.query_messages(payload, tokenizer, processor, maximum, truncate=shorten,
                                          truncate_question_candidates=shorten)
            kept = "state_head_and_tail"
        elif item.role == "document":
            def render(maximum, shorten):
                return ref.document_messages(item.text, tokenizer, processor, maximum, truncate=shorten)
            kept = "head"
        else:
            raise ValidationError(f"Unknown embedding role: {item.role}")
        original = render(measurement_limit, False)
        original_tokens = len(ref.message_ids(original, tokenizer, processor, measurement_limit))
        if original_tokens <= limit:
            return original, None
        if not truncate:
            raise ValidationError(f"Input exceeds {limit} tokens ({original_tokens}); truncation is disabled")
        messages = render(limit, True)
        encoded_tokens = len(ref.message_ids(messages, tokenizer, processor, limit))
        return messages, {"original_tokens": original_tokens, "encoded_tokens": encoded_tokens, "kept": kept}

    def encode(self, items):
        with self._lock:
            self._load()
            messages, truncated = [], []
            try:
                for index, item in enumerate(items):
                    prompt, record = self._messages(item)
                    messages.append(prompt)
                    if record:
                        truncated.append({"index": index, **record})
            except (ValueError, KeyError, TypeError) as exc:
                raise ValidationError(f"Invalid Kanana input: {exc}") from exc
            try:
                vectors, tokens = self._model.embed_messages(messages)
                return EmbeddingBatch(vectors.cpu().tolist(), tokens, "tokenizer", {"truncated": truncated})
            except Exception as exc:
                raise BackendError(f"Kanana encoding failed: {exc}") from exc
