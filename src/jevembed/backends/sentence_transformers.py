from dataclasses import replace
import hashlib
import importlib.metadata
from pathlib import Path
import threading

from .base import EmbeddingBatch
from ..config import ModelConfig
from ..errors import BackendError, ValidationError
from ..prompts import PromptAdapter


class SentenceTransformersBackend:
    """Lazy SentenceTransformer loader. Never changes the environment or retries with trust enabled."""

    def __init__(self, model_name_or_path=None, *, config=None, **options):
        self.config = config or ModelConfig(model_name_or_path=model_name_or_path or "", **options)
        if config is not None and (model_name_or_path is not None or options):
            self.config = replace(config, model_name_or_path=model_name_or_path or config.model_name_or_path, **options)
        self.adapter = PromptAdapter(self.config.prompts)
        self._model = None
        self._lock = threading.RLock()
        self.metadata = {}

    def prepare(self):
        with self._lock:
            self._load()
            return self.metadata

    def _load(self):
        if self._model is not None:
            return
        cfg = self.config
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise BackendError("Install jevembed[local] in a compatible environment") from exc
        device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
        dtype = cfg.dtype
        if dtype == "auto":
            dtype = "bfloat16" if device.startswith("cuda") and torch.cuda.is_bf16_supported() else "float32"
        if device == "cpu" and dtype != "float32":
            raise ValidationError("CPU backend requires dtype=float32 or auto")
        model_kwargs = {"torch_dtype": getattr(torch, dtype)}
        if cfg.attention_implementation:
            model_kwargs["attn_implementation"] = cfg.attention_implementation
        if cfg.code_revision:
            model_kwargs["code_revision"] = cfg.code_revision
        try:
            model = SentenceTransformer(cfg.model_name_or_path, revision=cfg.revision,
                                        trust_remote_code=cfg.trust_remote_code, device=device,
                                        local_files_only=cfg.local_files_only, model_kwargs=model_kwargs)
        except Exception as exc:
            raise BackendError(f"Cannot load {cfg.model_id} from {cfg.model_name_or_path} "
                               f"(trust_remote_code={cfg.trust_remote_code}); check this model's "
                               f"config and Transformers dependencies: {exc}") from exc
        pooling = [m.get_config_dict() for m in model if m.__class__.__name__ == "Pooling"]
        if any(not p.get("include_prompt", True) for p in pooling):
            raise BackendError("include_prompt=False requires a dedicated prompt-aware adapter")
        first = model[0]
        if first.__class__.__name__ != "Transformer":
            raise BackendError("This model requires a dedicated tokenizer/length adapter")
        limit = int(model.max_seq_length)
        if isinstance(cfg.max_input_tokens, int):
            if cfg.max_input_tokens > limit:
                raise ValidationError(f"Configured token limit exceeds model limit {limit}")
            limit = cfg.max_input_tokens
        model.max_seq_length = limit
        dimension = model.get_sentence_embedding_dimension()
        if cfg.expected_dimension and dimension != cfg.expected_dimension:
            raise BackendError(f"Expected dimension {cfg.expected_dimension}, got {dimension}")
        model_config = first.auto_model.config
        resolved_revision = getattr(model_config, "_commit_hash", None)
        root = Path(cfg.model_name_or_path)
        local_files = {}
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file() and (path.suffix in (".json", ".py") or path.name.endswith(".safetensors")) and ".cache" not in path.parts:
                    stat = path.stat()
                    local_files[str(path.relative_to(root))] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                    if path.suffix in (".json", ".py"):
                        local_files[str(path.relative_to(root))]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            metadata_file = root / ".cache/huggingface/download/config.json.metadata"
            if metadata_file.is_file():
                resolved_revision = metadata_file.read_text().splitlines()[0]
        tokenizer = model.tokenizer
        self.metadata = {"model_name_or_path": cfg.model_name_or_path, "requested_revision": cfg.revision,
                         "resolved_revision": resolved_revision, "code_revision": cfg.code_revision,
                         "revision_status": "resolved" if resolved_revision else "local_snapshot_commit_unknown",
                         "local_files": local_files, "trust_remote_code": cfg.trust_remote_code,
                         "device": str(model.device), "dtype": dtype, "dimension": dimension,
                         "pooling": pooling, "padding_side": tokenizer.padding_side,
                         "tokenizer": tokenizer.__class__.__name__, "max_input_tokens": limit,
                         "attention_implementation": getattr(model_config, "_attn_implementation", None),
                         "is_causal": getattr(model_config, "is_causal", None),
                         "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "sentence-transformers")}}
        self._model = model

    def encode(self, items):
        with self._lock:
            self._load()
            model, cfg = self._model, self.config
            texts = [self.adapter.render(item) for item in items]
            # Mirror the repository Transformer.tokenize preprocessing for exact counts.
            prepared = [text.strip() for text in texts]
            if model[0].do_lower_case:
                prepared = [text.lower() for text in prepared]
            lengths = [len(ids) for ids in model.tokenizer(prepared, truncation=False, padding=False)["input_ids"]]
            truncated = [{"index": i, "original_tokens": n, "encoded_tokens": model.max_seq_length}
                         for i, n in enumerate(lengths) if n > model.max_seq_length]
            if truncated and cfg.overflow_policy == "error":
                raise ValidationError(f"Input exceeds {model.max_seq_length} tokens: {truncated}")
            try:
                vectors = model.encode(texts, prompt="", batch_size=cfg.batch_size,
                                       normalize_embeddings=cfg.normalize_embeddings,
                                       convert_to_numpy=True, show_progress_bar=False).tolist()
            except Exception as exc:
                raise BackendError(f"Encoding failed for {cfg.model_id}: {exc}") from exc
            return EmbeddingBatch(vectors, sum(min(n, model.max_seq_length) for n in lengths),
                                  "tokenizer", {**self.metadata, "truncated": truncated})
