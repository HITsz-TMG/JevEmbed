"""Native decision adapters for two published EmbeddingGemma checkpoints."""

from dataclasses import dataclass
from contextlib import nullcontext
import importlib.metadata
import json
import math
from pathlib import Path
import threading

from ..compiler import compile_request
from ..errors import BackendError, ValidationError
from ..scoring import decision_response
from .runtime import resolve_device_dtype


def compile_gemma_request(request, prompts):
    for question in request["questions"].values():
        criteria = question.get("criteria") or {}
        if question["type"] == "noul" and any(value not in (None, "") for value in criteria.values()):
            raise ValidationError("EmbeddingGemma Typed Decision does not support custom Noul criteria")
    plans = compile_request(request, prompts)
    for plan in plans:
        plan.path = "native_gemma_typed_decision"
    return plans


def compile_clef_request(request, prompts):
    plans = compile_request(request, prompts)
    for plan in plans:
        plan.path = "native_clef_gemma_decision"
    return plans


def _to_text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class GemmaTypedDecisionBackend:
    """Run the upstream schema-conditioned state/question/option decision head."""

    def __init__(self, *, config, model=None):
        self.config = config
        if config.backend != "gemma_typed_decision":
            raise ValidationError("GemmaTypedDecisionBackend requires backend=gemma_typed_decision")
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
            from transformers import AutoModel, AutoTokenizer
            import huggingface_hub
            import transformers
        except ImportError as exc:
            raise BackendError("Install jevembed[gemma-decision] in its dedicated Python environment") from exc
        device, dtype, torch_dtype = resolve_device_dtype(cfg, torch, reject_float16=True)
        try:
            root = Path(cfg.model_name_or_path)
            if not root.is_dir():
                root = Path(huggingface_hub.snapshot_download(
                    repo_id=cfg.model_name_or_path, revision=cfg.revision,
                    local_files_only=cfg.local_files_only))
            model = AutoModel.from_pretrained(
                str(root), trust_remote_code=True, local_files_only=True, dtype=torch_dtype,
                attn_implementation=cfg.attention_implementation or "sdpa",
            ).eval().to(device)
            if not callable(getattr(model, "decide", None)) or not callable(getattr(model, "_build_inputs", None)):
                raise BackendError("Checkpoint does not expose the reviewed GemmaDecision API")
            model._tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True,
                                                              trust_remote_code=False)
            dimension = int(model.config.embed_dim)
            if cfg.expected_dimension and dimension != cfg.expected_dimension:
                raise BackendError(f"Expected dimension {cfg.expected_dimension}, got {dimension}")
            max_length = int(model.config.max_seq_length or 8192)
            if cfg.max_input_tokens != "model_default" and cfg.max_input_tokens > max_length:
                raise ValidationError(f"Configured token limit exceeds model limit {max_length}")
            if cfg.max_input_tokens != "model_default":
                model.config.max_seq_length = cfg.max_input_tokens
            self.metadata = {
                "model_name_or_path": cfg.model_name_or_path, "requested_revision": cfg.revision,
                "resolved_revision": getattr(model.config, "_commit_hash", None),
                "trust_remote_code": True, "device": device, "dtype": str(dtype).removeprefix("torch."),
                "dimension": dimension, "pooling": "upstream normalized mean pooling",
                "max_input_tokens": cfg.max_input_tokens if cfg.max_input_tokens != "model_default" else max_length,
                "decision_head": "schema-conditioned typed-decision head",
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                             "huggingface_hub": huggingface_hub.__version__},
            }
        except (BackendError, ValidationError):
            raise
        except Exception as exc:
            raise BackendError(f"Cannot load EmbeddingGemma Typed Decision checkpoint: {exc}") from exc
        self._model = model

    def evaluate(self, request, plans):
        with self._lock:
            self.prepare()
            model = self._model
            try:
                inputs, _, texts = model._build_inputs([("", request["state"], request["questions"], None)])
                tokenizer = model.get_tokenizer()
                rendered = [model.config.prompt + text for text in texts]
                original = tokenizer(rendered, padding=False, truncation=False)["input_ids"]
                limit = int(model.config.max_seq_length or 8192)
                if self.config.overflow_policy == "error" and any(len(tokens) > limit for tokens in original):
                    raise ValidationError(f"EmbeddingGemma inputs exceed the {limit}-token model limit")
                input_tokens = int(inputs["attention_mask"].sum().item())
                predictions = model.decide(request["state"], request["questions"], workflow="")
            except ValidationError:
                raise
            except Exception as exc:
                raise BackendError(f"EmbeddingGemma Typed Decision inference failed: {exc}") from exc
            return decision_response(self.config, plans, predictions, input_tokens)


@dataclass(frozen=True)
class _ClefAnswer:
    probabilities: dict[str, float]
    expected_level: float | None


class ClefGemmaDecisionBackend:
    """Local implementation of qnaug's pinned option-span EmbeddingGemma head."""

    def __init__(self, *, config, encoder=None, tokenizer=None, head=None, checkpoint=None):
        self.config = config
        if config.backend != "clef_gemma_decision":
            raise ValidationError("ClefGemmaDecisionBackend requires backend=clef_gemma_decision")
        self._encoder, self._tokenizer = encoder, tokenizer
        self._head, self._checkpoint = head, checkpoint
        self._torch = None
        self._device = None
        self._lock = threading.RLock()
        self.metadata = {}

    def prepare(self):
        with self._lock:
            if self._encoder is None:
                self._load()
            return self.metadata

    def _load(self):
        cfg = self.config
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise BackendError("Install jevembed[gemma-decision] in its dedicated Python environment") from exc
        device, dtype, torch_dtype = resolve_device_dtype(cfg, torch, reject_float16=True)
        root = Path(cfg.model_name_or_path)
        if not root.is_dir():
            try:
                from huggingface_hub import snapshot_download
                root = Path(snapshot_download(repo_id=cfg.model_name_or_path, revision=cfg.revision,
                                              local_files_only=cfg.local_files_only))
            except Exception as exc:
                raise BackendError(f"Cannot download Clef decision checkpoint: {exc}") from exc
        encoder_root = root / "encoder"
        try:
            if not (encoder_root / "config.json").is_file():
                raise BackendError(f"Missing encoder/config.json under {root}")
            checkpoint = torch.load(root / "head.pt", map_location="cpu", weights_only=True)
            if cfg.expected_dimension and int(checkpoint["d_model"]) != cfg.expected_dimension:
                raise BackendError("Decision head dimension does not match expected_dimension")
            encoder = AutoModel.from_pretrained(str(encoder_root), local_files_only=True, dtype=torch_dtype,
                                                attn_implementation=cfg.attention_implementation or "sdpa").eval().to(device)
            tokenizer = AutoTokenizer.from_pretrained(str(encoder_root), local_files_only=True)
            hidden = int(checkpoint["d_model"])

            class DecisionHead(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.norm = torch.nn.LayerNorm(hidden)
                    self.w_opt = torch.nn.Linear(hidden, 512)
                    self.w_q = torch.nn.Linear(hidden, 512)
                    self.w_int = torch.nn.Linear(hidden, 512, bias=False)
                    self.out = torch.nn.Sequential(torch.nn.GELU(), torch.nn.Dropout(0.1),
                                                   torch.nn.Linear(512, 1))

                def forward(self, opt_vecs, q_vec):
                    opts, question = self.norm(opt_vecs), self.norm(q_vec).unsqueeze(0)
                    joint = self.w_opt(opts) + self.w_q(question) + self.w_int(opts * question)
                    return self.out(joint).squeeze(-1)

            head = DecisionHead()
            head.load_state_dict(checkpoint["head"], strict=True)
            head.eval().to(device=device, dtype=torch.float32)
            max_length = int(checkpoint.get("config", {}).get("max_len", 1024))
            if cfg.max_input_tokens != "model_default" and cfg.max_input_tokens > max_length:
                raise ValidationError(f"Configured token limit exceeds checkpoint max_len {max_length}")
        except (BackendError, ValidationError):
            raise
        except Exception as exc:
            raise BackendError(f"Cannot load Clef EmbeddingGemma decision checkpoint: {exc}") from exc
        self._encoder, self._tokenizer, self._head, self._checkpoint = encoder, tokenizer, head, checkpoint
        self._torch, self._device = torch, device
        self._limit = cfg.max_input_tokens if cfg.max_input_tokens != "model_default" else max_length
        self.metadata = {"model_name_or_path": cfg.model_name_or_path, "requested_revision": cfg.revision,
                         "device": device, "dtype": str(dtype).removeprefix("torch."),
                         "dimension": hidden, "max_input_tokens": self._limit,
                         "renderer": "state plus typed question and option token spans",
                         "remote_code_executed": False,
                         "versions": {"torch": torch.__version__,
                                      "transformers": importlib.metadata.version("transformers")}}

    def _ids(self, text):
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def _special_wrap(self):
        full = self._tokenizer("hello world")["input_ids"]
        core = self._ids("hello world")
        for start in range(len(full) - len(core) + 1):
            if full[start:start + len(core)] == core:
                return full[:start], full[start + len(core):]
        return [], []

    def _build_sequence(self, state, plans):
        tokenizer = self._tokenizer
        prefix = self._ids(self._checkpoint["prefix"])
        lead, trail = self._special_wrap()
        mark = self._ids("\n- ")
        state_ids = self._ids(_to_text(state))
        question_parts = []
        for plan in plans:
            question = plan.inputs[0].instruction
            qtype = plan.kind
            header = {"noul": "\nstatement (true or false): ",
                      "score": "\nrating question: ", "choice": "\nquestion: "}[qtype]
            qids = self._ids(header + question)
            options = [self._ids(item.text) for item in plan.inputs[1:]]
            question_parts.append((qids, options))
        questions_size = sum(len(qids) + sum(len(mark) + len(option) for option in options)
                             for qids, options in question_parts)
        budget = self._limit - len(lead) - len(trail) - len(prefix) - questions_size
        if budget < 16:
            raise ValidationError("Question and candidate text exceed the Clef checkpoint token budget")
        if len(state_ids) > budget and self.config.overflow_policy == "error":
            raise ValidationError(f"State and questions exceed the {self._limit}-token Clef checkpoint limit")
        kept_state = state_ids[:budget]
        ids = list(lead) + prefix + kept_state
        spans = []
        for qids, options in question_parts:
            q_start = len(ids)
            ids.extend(qids)
            q_span = (q_start, len(ids))
            option_spans = []
            for option in options:
                ids.extend(mark)
                start = len(ids)
                ids.extend(option)
                option_spans.append((start, len(ids)))
            spans.append((q_span, option_spans))
        ids.extend(trail)
        attention = [1] * len(ids)
        truncation = None if len(state_ids) <= budget else {
            "original_tokens": len(ids) + len(state_ids) - budget,
            "encoded_tokens": len(ids), "kept": "state_prefix_then_all_questions_and_options"}
        return ids, attention, spans, truncation

    def evaluate(self, request, plans):
        with self._lock:
            self.prepare()
            torch = self._torch
            ids, attention, spans, truncation = self._build_sequence(request["state"], plans)
            input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
            attention_mask = torch.tensor([attention], dtype=torch.long, device=self._device)
            use_bf16 = self._device.startswith("cuda") and str(next(self._encoder.parameters()).dtype) == "torch.bfloat16"
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
            with torch.inference_mode(), autocast:
                output = self._encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = (output.last_hidden_state if getattr(output, "last_hidden_state", None) is not None else output[0])[0].float()
            predictions = {}
            temperatures = self._checkpoint.get("temps", {})
            for plan, (question_span, option_spans) in zip(plans, spans, strict=True):
                question_vector = hidden[question_span[0]:question_span[1]].mean(0)
                option_vectors = torch.stack([hidden[start:end].mean(0) for start, end in option_spans])
                logits = self._head(option_vectors, question_vector)
                temperature = float(temperatures.get(f"typed|{plan.kind}",
                                                     temperatures.get(f"extra|{plan.kind}", 1.0)))
                if not math.isfinite(temperature) or temperature <= 0:
                    raise BackendError("Clef checkpoint contains an invalid calibration temperature")
                probabilities = torch.softmax(logits.float() / temperature, dim=-1).cpu().tolist()
                expected = (math.fsum(i * probability for i, probability in enumerate(probabilities))
                            if plan.kind == "score" else None)
                predictions[plan.question_id] = _ClefAnswer(probabilities, expected)
            result, diagnostics = decision_response(self.config, plans, predictions, len(ids))
            if truncation:
                diagnostics[plans[0].question_id]["truncated_state"] = truncation
            return result, diagnostics


__all__ = ["compile_gemma_request", "compile_clef_request",
           "GemmaTypedDecisionBackend", "ClefGemmaDecisionBackend"]
