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
    version: str = "v1"
    noul_format: str = "legacy"

    def __post_init__(self):
        if any(not isinstance(value, str) for value in (self.query_template, self.document_template, self.similarity_instruction, self.version)):
            raise ValidationError("Prompt settings must be strings")
        if self.noul_format not in ("legacy", "unified", "retrieval"):
            raise ValidationError("noul_format must be legacy, unified, or retrieval")
        for template in (self.query_template, self.document_template):
            try:
                fields = {f for _, f, _, _ in Formatter().parse(template) if f is not None}
                if fields - {"instruction", "text"} or "text" not in fields:
                    raise ValueError("templates require {text}; only {instruction}/{text} are supported")
                template.format(instruction="", text="")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValidationError(f"Invalid prompt template: {exc}") from exc


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
    noul_criteria: LogisticConfig = field(default_factory=LogisticConfig)
    noul_similarity: LogisticConfig = field(default_factory=LogisticConfig)
    confidence: str = "normalized_entropy"
    calibration_status: str = "uncalibrated"
    calibration_record: dict = field(default_factory=dict)

    def __post_init__(self):
        for v in (self.choice_temperature, self.score_temperature):
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                raise ValidationError("Temperatures must be positive and finite")
        if self.confidence != "normalized_entropy":
            raise ValidationError("Use a Python ConfidenceEstimator for custom confidence")


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "custom"
    backend: str = "sentence_transformers"
    model_name_or_path: str = ""
    revision: str | None = None
    code_revision: str | None = None
    adapter_name_or_path: str | None = None
    adapter_revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 16
    pooling: str = "model_default"
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
        for name in ("adapter_name_or_path", "adapter_revision"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValidationError(f"{name} must be a nonempty string or null")
        if self.adapter_revision and not self.adapter_name_or_path:
            raise ValidationError("adapter_revision requires adapter_name_or_path")
        if self.adapter_name_or_path and self.backend != "sentence_transformers":
            raise ValidationError("LoRA adapters require the sentence_transformers backend")
        for name in ("trust_remote_code", "local_files_only", "normalize_embeddings", "server_enforces_length"):
            if type(getattr(self, name)) is not bool:
                raise ValidationError(f"{name} must be an explicit boolean")
        for name, minimum in (("batch_size", 1), ("cache_capacity", 0), ("max_retries", 0)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValidationError(f"Invalid {name}")
        if self.max_input_tokens != "model_default" and (type(self.max_input_tokens) is not int or self.max_input_tokens < 1):
            raise ValidationError("max_input_tokens must be model_default or a positive integer")
        for name, choices in {"overflow_policy": ("error", "truncate"), "usage_mode": ("strict", "estimate"),
                              "backend": ("sentence_transformers", "http", "custom"),
                              "dtype": ("auto", "float32", "float16", "bfloat16"),
                              "pooling": ("model_default",), "template_owner": ("client", "server")}.items():
            if getattr(self, name) not in choices:
                raise ValidationError(f"Invalid {name}: {getattr(self, name)!r}")
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
            data["prompts"] = PromptConfig(**data.get("prompts", {}))
            scoring = dict(data.get("scoring", {}))
            for key in ("noul_criteria", "noul_similarity"):
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
