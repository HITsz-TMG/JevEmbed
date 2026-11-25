"""A configurable text encoder with separate query and document projection heads."""

from dataclasses import replace
import json
import hashlib
import importlib.metadata
import math
from pathlib import Path
import threading

from .base import EmbeddingBatch
from ..config import ModelConfig
from ..config import LogisticConfig
from ..errors import BackendError, ValidationError
from ..prompts import PromptAdapter
from .clm_qwen35 import render_query
from .runtime import resolve_device_dtype


def _make_head(nn, *, hidden, width, depth, projection, activation, layernorm, residual):
    if depth < 2 or min(hidden, width, projection) < 1:
        raise BackendError("Invalid projection-head dimensions or depth")
    try:
        activation_class = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
    except KeyError as exc:
        raise BackendError(f"Unsupported projection activation: {activation}") from exc

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList((nn.LayerNorm(width) if layernorm else nn.Identity())
                                       for _ in range(depth - 2))
            self.out = nn.Linear(width, projection)
            self.act = activation_class()
            self.residual = residual

        def forward(self, x):
            x = self.act(self.inp(x))
            for linear, norm in zip(self.hidden, self.norms):
                transformed = self.act(norm(linear(x)))
                x = x + transformed if self.residual else transformed
            return self.out(x)

    return Head()


class PairedProjectionBackend:
    """Encode role-tagged text with a base model and a checkpointed head pair.

    The checkpoint supplies ``cfg``, ``state_head``, and ``action_head``. Prompt
    layout, pooling, truncation, and scoring remain external configuration.
    """

    def __init__(self, model_name_or_path=None, *, config=None, **options):
        self.config = config or ModelConfig(backend="paired_projection", model_name_or_path=model_name_or_path or "", **options)
        if config is not None and (model_name_or_path is not None or options):
            self.config = replace(config, model_name_or_path=model_name_or_path or config.model_name_or_path, **options)
        if self.config.backend != "paired_projection":
            raise ValidationError("PairedProjectionBackend requires backend=paired_projection")
        self.adapter = PromptAdapter(self.config.prompts)
        self._lock = threading.RLock()
        self._model = None
        self._tokenizer = None
        self._heads = None
        self._device = None
        self._limit = None
        self._scale = None
        self.metadata = {}

    def _checkpoint_path(self):
        cfg = self.config
        root = Path(cfg.projection_name_or_path)
        if root.is_file():
            return root
        if root.is_dir():
            path = root / cfg.projection_filename
            if not path.is_file():
                raise BackendError(f"Projection checkpoint file not found: {path}")
            return path
        try:
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download(repo_id=cfg.projection_name_or_path,
                                        filename=cfg.projection_filename,
                                        revision=cfg.projection_revision,
                                        local_files_only=cfg.local_files_only))
        except Exception as exc:
            raise BackendError(f"Cannot load projection checkpoint {cfg.projection_name_or_path}: {exc}") from exc

    def prepare(self):
        with self._lock:
            self._load()
            return self.metadata

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise BackendError("Install jevembed[local] for paired-projection inference") from exc
        cfg = self.config
        device, dtype, torch_dtype = resolve_device_dtype(cfg, torch)
        checkpoint_path = self._checkpoint_path()
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            head_cfg = dict(checkpoint["cfg"])
            hidden = int(head_cfg["hidden_size"])
            projection_size = checkpoint.get("projection_dim", head_cfg.get("projection_dim", cfg.expected_dimension))
            if projection_size is None:
                raise BackendError("Projection dimension is missing from the checkpoint and configuration")
            projection = int(projection_size)
            head_kwargs = dict(hidden=hidden, width=int(head_cfg["width"]), depth=int(head_cfg["depth"]),
                               projection=projection, activation=head_cfg.get("activation", "gelu"),
                               layernorm=bool(head_cfg.get("layernorm", False)),
                               residual=bool(head_cfg.get("residual", False)))
            heads = {role: _make_head(torch.nn, **head_kwargs) for role in ("query", "document")}
            heads["query"].load_state_dict(checkpoint["state_head"], strict=True)
            heads["document"].load_state_dict(checkpoint["action_head"], strict=True)
            if cfg.expected_dimension and cfg.expected_dimension != projection:
                raise BackendError(f"Expected dimension {cfg.expected_dimension}, got {projection}")
            scale = float(torch.as_tensor(checkpoint["logit_scale"]).float().exp().clamp(max=100.0))
            if not math.isfinite(scale) or scale <= 0:
                raise BackendError("Projection checkpoint has an invalid logit scale")
            if cfg.prompts.version == "clm-qwen35-chat-options-v1":
                if abs(cfg.scoring.choice_temperature - 1.0 / scale) > 1e-4:
                    raise BackendError("CLM scoring temperature must match the pinned head logit_scale")
            for head in heads.values():
                head.eval().to(device=device, dtype=torch.float32)
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            raise BackendError(f"Invalid paired-projection checkpoint: {exc}") from exc
        model_kwargs = {"revision": cfg.revision, "trust_remote_code": cfg.trust_remote_code,
                        "local_files_only": cfg.local_files_only, "torch_dtype": torch_dtype}
        if cfg.code_revision:
            model_kwargs["code_revision"] = cfg.code_revision
        if cfg.attention_implementation:
            model_kwargs["attn_implementation"] = cfg.attention_implementation
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, revision=cfg.revision,
                                                       trust_remote_code=cfg.trust_remote_code,
                                                       local_files_only=cfg.local_files_only)
            model = AutoModel.from_pretrained(cfg.model_name_or_path, **model_kwargs).eval()
            if getattr(model.config, "model_type", None) == "qwen3_5":
                model = model.language_model
            model = model.to(device)
            model.config.use_cache = False
            if int(model.config.hidden_size) != hidden:
                raise BackendError(f"Base model hidden size {model.config.hidden_size} differs from checkpoint {hidden}")
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is None:
                    raise BackendError("Tokenizer has neither a pad token nor an EOS token")
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"Cannot load base encoder {cfg.model_name_or_path}: {exc}") from exc
        model_limit = getattr(model.config, "max_position_embeddings", None)
        if cfg.max_input_tokens == "model_default":
            if not isinstance(model_limit, int) or model_limit < 1:
                raise ValidationError("Base model has no usable position limit; set max_input_tokens")
            limit = model_limit
        else:
            limit = cfg.max_input_tokens
            if isinstance(model_limit, int) and limit > model_limit:
                raise ValidationError(f"Configured token limit exceeds model limit {model_limit}")
        checkpoint_hash = hashlib.sha256()
        with checkpoint_path.open("rb") as checkpoint_file:
            for block in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b""):
                checkpoint_hash.update(block)
        checkpoint_digest = checkpoint_hash.hexdigest()
        self.metadata = {
            "base_model": cfg.model_name_or_path, "base_revision": cfg.revision,
            "resolved_base_revision": getattr(model.config, "_commit_hash", None),
            "projection": cfg.projection_name_or_path, "projection_revision": cfg.projection_revision,
            "projection_sha256": checkpoint_digest, "projection_architecture": head_kwargs,
            "checkpoint_scale": scale, "projection_dimension": projection,
            "device": device, "dtype": dtype, "pooling": cfg.pooling,
            "max_input_tokens": limit, "tokenizer": tokenizer.__class__.__name__,
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        }
        self._model, self._tokenizer, self._heads = model, tokenizer, heads
        self._device, self._limit = device, limit
        self._scale = scale

    def scoring_config(self):
        if self.config.prompts.version != "clm-qwen35-chat-options-v1":
            return self.config.scoring
        return replace(self.config.scoring, choice_temperature=1 / self._scale,
                       score_temperature=1 / self._scale,
                       noul_with_criteria=LogisticConfig(self._scale, 0.0),
                       noul_without_criteria=LogisticConfig(self._scale, 0.0))

    def encode(self, items):
        with self._lock:
            self._load()
            import torch
            import torch.nn.functional as functional

            cfg, tokenizer = self.config, self._tokenizer
            texts, special_truncations = [], {}
            for index, item in enumerate(items):
                if item.role == "query" and cfg.prompts.version == "clm-qwen35-chat-options-v1":
                    try:
                        payload = json.loads(item.text)
                        rendered, truncation = render_query(
                            tokenizer, payload, max_tokens=self._limit, overflow_policy=cfg.overflow_policy)
                    except Exception as exc:
                        raise BackendError(f"Cannot render CLM-Qwen3.5 chat-options input: {exc}") from exc
                    texts.append(rendered)
                    if truncation:
                        special_truncations[index] = truncation
                else:
                    texts.append(self.adapter.render(item))
            encoded = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
            token_lists, truncated = [], []
            for index, (item, ids) in enumerate(zip(items, encoded)):
                if item.role not in self._heads:
                    raise ValidationError(f"Unknown embedding role: {item.role}")
                if not ids:
                    ids = tokenizer(" ", add_special_tokens=False)["input_ids"]
                original = len(ids)
                if index in special_truncations:
                    truncated.append({"index": index, "original_tokens": special_truncations[index]["original_tokens"],
                                      "encoded_tokens": original, "kept": special_truncations[index]["kept"]})
                elif original > self._limit:
                    if cfg.overflow_policy == "error":
                        raise ValidationError(f"Input {index} exceeds {self._limit} tokens ({original})")
                    side = cfg.query_truncation_side if item.role == "query" else cfg.document_truncation_side
                    ids = ids[-self._limit:] if side == "left" else ids[:self._limit]
                    truncated.append({"index": index, "original_tokens": original,
                                      "encoded_tokens": len(ids), "kept": "tail" if side == "left" else "head"})
                token_lists.append({"input_ids": ids})
            batch = tokenizer.pad(token_lists, padding=True, return_tensors="pt")
            batch = {key: value.to(self._device) for key, value in batch.items()}
            with torch.inference_mode():
                hidden = self._model(**batch, use_cache=False).last_hidden_state
                mask = batch["attention_mask"]
                positions = torch.arange(len(items), device=hidden.device)
                if cfg.pooling == "last_token":
                    last = mask.shape[1] - 1 - mask.flip(dims=[1]).long().argmax(dim=1)
                    pooled = hidden[positions, last]
                elif cfg.pooling == "cls":
                    pooled = hidden[positions, mask.long().argmax(dim=1)]
                else:
                    pooled = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)
                pooled = functional.normalize(pooled.float(), dim=-1)
                projected = [functional.normalize(self._heads[item.role](pooled[index]), dim=-1)
                             for index, item in enumerate(items)]
                vectors = torch.stack(projected).cpu().tolist()
            return EmbeddingBatch(vectors, sum(map(len, (entry["input_ids"] for entry in token_lists))),
                                  "tokenizer", {"truncated": truncated})
