"""The test evaluator must restore question input order after length batching."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from jevembed.backends.base import EmbeddingInput


spec = importlib.util.spec_from_file_location(
    "testset_evaluation", Path(__file__).resolve().parents[2] / "scripts/evaluate_jevembed_data.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


@pytest.mark.parametrize("backend_name", ["sentence_transformers", "paired_projection"])
def test_length_sorted_encoding_restores_original_order(backend_name):
    class Backend:
        adapter = SimpleNamespace(render=lambda item: item.text)

        def __init__(self):
            self.calls = []

        def encode(self, items):
            self.calls.append([(item.role, item.text) for item in items])
            return SimpleNamespace(vectors=[[item.role, item.text] for item in items])

    inputs = [EmbeddingInput("query", "", "long-query"),
              EmbeddingInput("document", "", "long-document"),
              EmbeddingInput("query", "", "q"),
              EmbeddingInput("document", "", "d")]
    backend = Backend()
    vectors = evaluation.encode_length_sorted(backend, inputs, 2, backend_name)

    assert vectors == [[item.role, item.text] for item in inputs]
    assert [item for call in backend.calls for item in call] == [
        ("document", "d"), ("document", "long-document"),
        ("query", "q"), ("query", "long-query")]
