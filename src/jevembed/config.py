from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from string import Formatter

from .errors import ValidationError


@dataclass(frozen=True)
class PromptConfig:
    query_template: str = "Instruct: {instruction}\nQuery: {text}"
    document_template: str = "{text}"
    similarity_instruction: str = "Retrieve semantically similar text."
    serialization_mode: str = "compact_json"
    choice_candidate_template: str = "{label}: {text}"
    choice_empty_uses_label: bool = False
    noul_input_mapping: str = "retrieval_queries"
    noul_candidate_template: str = "{label}: {text}"
    noul_true_fallback: str | None = None
    noul_false_fallback: str | None = None
    strip_rendered: bool = False
    version: str = "v1"

    def __post_init__(self):
        if any(not isinstance(value, str) for value in (self.query_template, self.document_template, self.similarity_instruction, self.version)):
            raise ValidationError("Prompt settings must be strings")
        if self.serialization_mode not in ("compact_json", "prose"):
            raise ValidationError("serialization_mode must be compact_json or prose")
        if self.noul_input_mapping not in ("retrieval_queries", "binary_candidates"):
            raise ValidationError("noul_input_mapping must be retrieval_queries or binary_candidates")
        for name in ("strip_rendered", "choice_empty_uses_label"):
            if type(getattr(self, name)) is not bool:
                raise ValidationError(f"{name} must be an explicit boolean")
        for template, allowed, required in (
            (self.query_template, {"instruction", "text"}, {"text"}),
            (self.document_template, {"instruction", "text"}, {"text"}),
            (self.choice_candidate_template, {"label", "text"}, set()),
            (self.noul_candidate_template, {"label", "text"}, set()),
        ):
            try:
                fields = {f for _, f, _, _ in Formatter().parse(template) if f is not None}
                if fields - allowed or not required <= fields:
                    raise ValueError(f"template requires {sorted(required)} and supports only {sorted(allowed)}")
                template.format(instruction="", text="", label="")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValidationError(f"Invalid prompt template: {exc}") from exc
        for fallback in (self.noul_true_fallback, self.noul_false_fallback):
            if fallback is None:
                continue
            if not isinstance(fallback, str):
                raise ValidationError("Noul fallback templates must be strings")
            try:
                fields = {f for _, f, _, _ in Formatter().parse(fallback) if f is not None}
                if fields - {"instruction"}:
                    raise ValueError("Noul fallback templates support only {instruction}")
                fallback.format(instruction="")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValidationError(f"Invalid Noul fallback template: {exc}") from exc
        if self.noul_input_mapping == "binary_candidates" and (not self.noul_true_fallback or not self.noul_false_fallback):
            raise ValidationError("binary_candidates requires both Noul fallback templates")


@dataclass(frozen=True)
class LogisticConfig:
    slope: float = 10.0
    intercept: float = 0.0

    def __post_init__(self):
        for v in (self.slope, self.intercept):
            if type(v) not in (int, float) or not math.isfinite(v):
                raise ValidationError("Logistic parameters must be finite numbers")


@dataclass(frozen=True)
class ScoringConfig:
    choice_temperature: float = 0.1
    score_temperature: float = 0.1
    noul_with_criteria: LogisticConfig = field(default_factory=LogisticConfig)
    noul_without_criteria: LogisticConfig = field(default_factory=LogisticConfig)
    confidence: str = "normalized_entropy"
    calibration_status: str = "uncalibrated"
    calibration_record: dict = field(default_factory=dict)

    def __post_init__(self):
        for v in (self.choice_temperature, self.score_temperature):
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                raise ValidationError("Temperatures must be positive and finite")
        if self.confidence not in ("normalized_entropy", "top_margin"):
            raise ValidationError("confidence must be normalized_entropy or top_margin")


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "custom"
    backend: str = "sentence_transformers"
    model_name_or_path: str = ""
    revision: str | None = None
    code_revision: str | None = None
    adapter_name_or_path: str | None = None
    adapter_revision: str | None = None
    projection_name_or_path: str | None = None
    projection_filename: str | None = None
    projection_revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 16
    pooling: str = "model_default"
    query_truncation_side: str = "right"
    document_truncation_side: str = "right"
    normalize_embeddings: bool = True
    max_input_tokens: int | str = "model_default"
    overflow_policy: str = "error"
    attention_implementation: str | None = None
    expected_dimension: int | None = None
    prompts: PromptConfig = field(default_factory=PromptConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    cache_capacity: int = 4096
    usage_mode: str = "strict"
    base_url: str = "http://127.0.0.1:8001/v1"
    api_key_env: str | None = None
    timeout: float = 60.0
    max_retries: int = 2
    template_owner: str = "client"
    server_enforces_length: bool = False
    aliases: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("adapter_name_or_path", "adapter_revision", "projection_name_or_path", "projection_filename", "projection_revision"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValidationError(f"{name} must be a nonempty string or null")
        if self.adapter_revision and not self.adapter_name_or_path:
            raise ValidationError("adapter_revision requires adapter_name_or_path")
        if self.adapter_name_or_path and self.backend != "sentence_transformers":
            raise ValidationError("LoRA adapters require the sentence_transformers backend")
        if self.backend == "paired_projection":
            if not self.model_name_or_path or not self.projection_name_or_path or not self.projection_filename:
                raise ValidationError("paired_projection requires model_name_or_path, projection_name_or_path and projection_filename")
            if self.pooling == "model_default":
                raise ValidationError("paired_projection requires an explicit pooling mode")
        elif any((self.projection_name_or_path, self.projection_filename, self.projection_revision)):
            raise ValidationError("Projection checkpoint settings require the paired_projection backend")
        for name in ("trust_remote_code", "local_files_only", "normalize_embeddings", "server_enforces_length"):
            if type(getattr(self, name)) is not bool:
                raise ValidationError(f"{name} must be an explicit boolean")
        for name, minimum in (("batch_size", 1), ("cache_capacity", 0), ("max_retries", 0)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValidationError(f"Invalid {name}")
        if self.max_input_tokens != "model_default" and (type(self.max_input_tokens) is not int or self.max_input_tokens < 1):
            raise ValidationError("max_input_tokens must be model_default or a positive integer")
        for name, choices in {"overflow_policy": ("error", "truncate"), "usage_mode": ("strict", "estimate"),
                              "backend": ("sentence_transformers", "http", "paired_projection", "custom"),
                              "dtype": ("auto", "float32", "float16", "bfloat16"),
                              "pooling": ("model_default", "last_token", "mean", "cls"),
                              "query_truncation_side": ("left", "right"),
                              "document_truncation_side": ("left", "right"),
                              "template_owner": ("client", "server")}.items():
            if getattr(self, name) not in choices:
                raise ValidationError(f"Invalid {name}: {getattr(self, name)!r}")
        if self.backend != "paired_projection" and self.pooling != "model_default":
            raise ValidationError("Explicit pooling currently requires the paired_projection backend")
        if self.backend in ("sentence_transformers", "http") and (
            self.query_truncation_side == "left" or self.document_truncation_side == "left"
        ):
            raise ValidationError("Left truncation requires the paired_projection or custom backend")
        if self.projection_filename and Path(self.projection_filename).name != self.projection_filename:
            raise ValidationError("projection_filename must be a filename, not a path")
        if not self.model_id or not isinstance(self.model_id, str):
            raise ValidationError("model_id must be a nonempty string")
        if self.expected_dimension is not None and (type(self.expected_dimension) is not int or self.expected_dimension <= 0):
            raise ValidationError("expected_dimension must be a positive integer")
        if not isinstance(self.aliases, (tuple, list)) or any(not isinstance(a, str) or not a for a in self.aliases):
            raise ValidationError("aliases must be a list of nonempty strings")
        if not isinstance(self.timeout, (int, float)) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValidationError("timeout must be positive and finite")

    @classmethod
    def from_dict(cls, data):
        try:
            if not isinstance(data, dict):
                raise ValueError("model config must be a mapping")
            data = dict(data)
            prompts = dict(data.get("prompts", {}))
            old_noul_format = prompts.pop("noul_format", "retrieval")
            if old_noul_format != "retrieval":
                raise ValueError("noul_format only supports retrieval; legacy and unified Noul mappings are no longer supported")
            data["prompts"] = PromptConfig(**prompts)
            scoring = dict(data.get("scoring", {}))
            for old, new in (("noul_criteria", "noul_with_criteria"),
                             ("noul_similarity", "noul_without_criteria")):
                if old in scoring:
                    if new in scoring:
                        raise ValueError(f"Use either {old} or {new}, not both")
                    scoring[new] = scoring.pop(old)
            for key in ("noul_with_criteria", "noul_without_criteria"):
                if key in scoring:
                    scoring[key] = LogisticConfig(**scoring[key])
            data["scoring"] = ScoringConfig(**scoring)
            aliases = data.get("aliases", ())
            if not isinstance(aliases, (tuple, list)):
                raise ValueError("aliases must be an array")
            data["aliases"] = tuple(aliases)
            return cls(**data)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Invalid model config: {exc}") from exc

    @classmethod
    def load(cls, path):
        import yaml
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
