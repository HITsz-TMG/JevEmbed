import hashlib
import json
from pathlib import Path

import pytest

from jevembed import JevEmbed, ModelConfig
from jevembed.backends import EmbeddingBatch

ROOT = Path(__file__).resolve().parents[1]


class FixedBackend:
    def __init__(self, mapping=None):
        self.mapping, self.calls = mapping or {}, []

    def encode(self, items):
        self.calls.append(items)
        vectors = []
        for item in items:
            vector = self.mapping.get(item.text)
            if vector is None:
                digest = hashlib.sha256(repr(item).encode()).digest()
                vector = [v-127 for v in digest[:4]]
            vectors.append(vector)
        return EmbeddingBatch(vectors, len(items)*7, "fixed_test_tokens")


@pytest.fixture
def backend():
    return FixedBackend()


@pytest.fixture
def client(backend):
    return JevEmbed(backend=backend, model="test")


@pytest.fixture
def fixtures():
    return {name: json.loads((ROOT / f"tests/fixtures/official_{name}.request.json").read_text())
            for name in ("choice_exchange", "score_safari", "noul_escalation")}


def request(kind="choice", criteria=None):
    q = {"type": kind, "instructions": "instruction"}
    if criteria is not None:
        q["criteria"] = criteria
    elif kind == "choice":
        q["criteria"] = {"a": "alpha", "b": "beta"}
    elif kind == "score":
        q["criteria"] = ["low", "high"]
    return {"state": "state", "questions": {"q": q}}
