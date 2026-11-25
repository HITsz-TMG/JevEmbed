from dataclasses import replace
import sys
from types import SimpleNamespace, ModuleType

import pytest

from jevembed import JevEmbed, ModelConfig, BackendError, ValidationError
from jevembed.backends import SentenceTransformersBackend, EmbeddingInput


@pytest.fixture
def fake_runtime(monkeypatch):
    state = {"load_calls": [], "encode_calls": []}
    class Transformer:
        do_lower_case = False
        auto_model = SimpleNamespace(config=SimpleNamespace(_commit_hash="commit123", _attn_implementation="sdpa"))
    class Pooling:
        def get_config_dict(self):
            return {"include_prompt": state.get("include_prompt", True), "pooling_mode_mean_tokens": True}
    class Tokenizer:
        padding_side = "right"
        def __call__(self, texts, **kwargs):
            return {"input_ids": [list(range(len(text))) for text in texts]}
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            state["load_calls"].append((args, kwargs))
            if state.get("fail"):
                raise RuntimeError("requires trust_remote_code")
            self.max_seq_length = 100
            self.device = "cpu"
            self.tokenizer = Tokenizer()
            self.modules = [Transformer(), Pooling()]
        def __iter__(self):
            return iter(self.modules)
        def __getitem__(self, i):
            return self.modules[i]
        def get_sentence_embedding_dimension(self):
            return 2
        def encode(self, texts, **kwargs):
            state["encode_calls"].append((texts, kwargs))
            return SimpleNamespace(tolist=lambda: [[1, 0] for _ in texts])
    fake = ModuleType("sentence_transformers")
    fake.SentenceTransformer = SentenceTransformer
    torch = ModuleType("torch")
    torch.float32 = "float32"
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr("jevembed.backends.sentence_transformers.importlib.metadata.version", lambda name: "test-version")
    return state


def test_loader_parameters_prompt_once_usage_and_dimension(fake_runtime):
    cfg = ModelConfig(model_name_or_path="/missing/local", trust_remote_code=True, local_files_only=True,
                      revision="commit123", code_revision="code456", expected_dimension=2)
    b = SentenceTransformersBackend(config=cfg)
    assert fake_runtime["load_calls"] == []
    batch = b.encode([EmbeddingInput("query", "original", "state")])
    args, kwargs = fake_runtime["load_calls"][0]
    assert kwargs["trust_remote_code"] is True and kwargs["local_files_only"] is True
    assert kwargs["revision"] == "commit123" and kwargs["model_kwargs"]["code_revision"] == "code456"
    assert kwargs["device"] == "cpu" and kwargs["model_kwargs"]["torch_dtype"] == "float32"
    texts, kwargs = fake_runtime["encode_calls"][0]
    assert texts == ["Instruct: original\nQuery: state"] and kwargs["prompt"] == ""
    assert batch.input_tokens == len(texts[0]) and batch.metadata["dimension"] == 2


def test_loading_failure_never_enables_trust(fake_runtime):
    fake_runtime["fail"] = True
    with pytest.raises(BackendError, match="trust_remote_code=False"):
        SentenceTransformersBackend("broken").encode([EmbeddingInput("document", "", "a")])
    assert len(fake_runtime["load_calls"]) == 1
    assert fake_runtime["load_calls"][0][1]["trust_remote_code"] is False


def test_adapter_loading_and_provenance(fake_runtime, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sys.modules["sentence_transformers"].SentenceTransformer, "load_adapter",
                        lambda self, *args, **kwargs: calls.append((args, kwargs)), raising=False)
    (tmp_path / "adapter_config.json").write_text('{"r":8}')
    (tmp_path / "adapter_model.safetensors").write_bytes(b"mock adapter")
    cfg = ModelConfig(model_name_or_path="fake", adapter_name_or_path=str(tmp_path),
                      adapter_revision="commit", local_files_only=True)
    metadata = SentenceTransformersBackend(config=cfg).prepare()
    assert calls == [((str(tmp_path),), {"revision":"commit", "is_trainable":False,
                                      "adapter_kwargs":{"local_files_only":True}})]
    assert set(metadata["adapter"]["files_sha256"]) == {"adapter_config.json", "adapter_model.safetensors"}


def test_missing_dependency(fake_runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(BackendError, match="compatible environment"):
        SentenceTransformersBackend("missing").encode([EmbeddingInput("document", "", "a")])


def test_overflow_and_explicit_truncation(fake_runtime):
    cfg = ModelConfig(model_name_or_path="fake", max_input_tokens=3)
    with pytest.raises(ValidationError, match="exceeds"):
        SentenceTransformersBackend(config=cfg).encode([EmbeddingInput("document", "", "12345")])
    assert fake_runtime["encode_calls"] == []
    b = SentenceTransformersBackend(config=replace(cfg, overflow_policy="truncate"))
    batch = b.encode([EmbeddingInput("document", "", "12345")])
    assert batch.input_tokens == 3
    assert batch.metadata["truncated"] == [{"index": 0, "original_tokens": 5, "encoded_tokens": 3}]


def test_excluded_prompt_and_dimension_rejected(fake_runtime):
    with pytest.raises(BackendError, match="dimension"):
        SentenceTransformersBackend(config=ModelConfig(expected_dimension=7)).encode([])
    fake_runtime["include_prompt"] = False
    with pytest.raises(BackendError, match="dedicated"):
        SentenceTransformersBackend("fake").encode([])


def test_cached_truncation_and_direct_backend_constructor(fake_runtime):
    backend = SentenceTransformersBackend("fake", max_input_tokens=3, overflow_policy="truncate")
    c = JevEmbed(backend=backend, model="local")
    r = {"state": "long state", "questions": {"q": {"type": "noul", "instructions": "long question"}}}
    first = c.evaluate_with_trace(r)
    cached = c.evaluate_with_trace(r)
    assert first["trace"]["truncated_inputs"] == cached["trace"]["truncated_inputs"]
    assert len(cached["trace"]["truncated_inputs"]) == 2
    assert cached["response"]["usage"]["input_tokens"] == 0
    assert len(fake_runtime["encode_calls"]) == 1


def test_loaded_revision_separates_cache(fake_runtime):
    backend = SentenceTransformersBackend("fake")
    c = JevEmbed(backend=backend, model="local")
    r = {"state": "state", "questions": {"q": {"type": "noul", "instructions": "question"}}}
    c.evaluate(r)
    backend.metadata["resolved_revision"] = "different"
    assert c.evaluate(r)["usage"]["input_tokens"] > 0
