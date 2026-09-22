from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingInput:
    role: str
    instruction: str
    text: str


@dataclass
class EmbeddingBatch:
    vectors: list
    input_tokens: int | None
    usage_source: str = "backend"
    metadata: dict = field(default_factory=dict)


class EmbeddingBackend(Protocol):
    def encode(self, items: list[EmbeddingInput]) -> EmbeddingBatch: ...
