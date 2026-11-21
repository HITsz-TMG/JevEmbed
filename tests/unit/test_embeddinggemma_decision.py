from types import SimpleNamespace

import pytest

from conftest import ROOT
from jevembed import JevEmbed, ModelConfig
from jevembed.backends.embeddinggemma_decision import (
    ClefGemmaDecisionBackend,
    GemmaTypedDecisionBackend,
    compile_typed_decision_request,
)
from jevembed.errors import BackendError, ValidationError


def sample_request(model="embeddinggemma-2-ft-typed-decision"):
    return {
        "model": model,
        "state": {"ticket": "duplicate charge", "impact": "account remains usable"},
        "questions": {
            "team": {"type": "choice", "instructions": "Which team?",
                     "criteria": {"billing": "payments and refunds", "technical": "bugs and outages"}},
            "urgent": {"type": "noul", "instructions": "Needs immediate review?"},
            "severity": {"type": "score", "instructions": "How severe?", "criteria": ["low", "medium", "high"]},
        },
    }


class FakeMask:
    def sum(self):
        return SimpleNamespace(item=lambda: 37)


class FakeGemmaDecisionModel:
    config = SimpleNamespace(embed_dim=768, max_seq_length=8192, prompt="", _commit_hash="fake-revision")

    def _build_inputs(self, examples):
        return {"attention_mask": FakeMask()}, [], ["short test state"]

    def get_tokenizer(self):
        return lambda texts, **kwargs: {"input_ids": [list(range(len(text.split()))) for text in texts]}

    def decide(self, state, questions, workflow=""):
        return {
            "team": SimpleNamespace(probabilities={"billing": 0.8, "technical": 0.2}),
            "urgent": SimpleNamespace(probabilities={"false": 0.25, "true": 0.75}),
            "severity": SimpleNamespace(probabilities={"0": 0.1, "1": 0.6, "2": 0.3}, expected_level=1.2),
        }


def test_gemma_native_decisions_map_to_jev_answers_and_keep_native_head():
    config = ModelConfig.load(ROOT / "configs/embeddinggemma-2-ft-typed-decision.yaml")
    backend = GemmaTypedDecisionBackend(config=config, model=FakeGemmaDecisionModel())
    client = JevEmbed(config=config, backend=backend)
    result = client.evaluate_with_trace(sample_request())
    response = result["response"]

    assert response["usage"]["input_tokens"] == 37
    assert response["answers"]["team"]["choice"] == "billing"
    assert response["answers"]["team"]["confidence"] == 0.8
    assert response["answers"]["urgent"]["noul"] == 0.75
    assert response["answers"]["severity"]["score"] == 1.2
    assert result["trace"]["prompt_diagnostics"]["renderer"] == "separate_state_question_option_pooling"
    assert result["trace"]["prompt_diagnostics"]["vector_similarity_scoring"] is False


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=True):
        ids = list(range(2, 2 + len(text.split())))
        return {"input_ids": ([0] + ids + [1]) if add_special_tokens else ids}


class FakeEncoder:
    def __init__(self, torch):
        self.torch = torch
        self.parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter((self.parameter,))

    def __call__(self, input_ids, attention_mask):
        hidden = self.torch.ones((input_ids.shape[0], input_ids.shape[1], 768), device=input_ids.device)
        return SimpleNamespace(last_hidden_state=hidden)


class FakeHead:
    def __init__(self, torch):
        self.torch = torch

    def __call__(self, option_vectors, question_vector):
        return self.torch.arange(option_vectors.shape[0], dtype=self.torch.float32, device=option_vectors.device)


def test_clef_packed_option_spans_respect_labels_and_upstream_temperatures():
    torch = pytest.importorskip("torch")

    config = ModelConfig.load(ROOT / "configs/qnaug-embeddinggemma2-decision.yaml")
    backend = ClefGemmaDecisionBackend(config=config, encoder=FakeEncoder(torch), tokenizer=FakeTokenizer(),
                                       head=FakeHead(torch), checkpoint={"prefix": "", "temps": {"typed|choice": 1.0}})
    backend._torch, backend._device, backend._limit = torch, "cpu", 1024
    client = JevEmbed(config=config, backend=backend)
    request = sample_request("qnaug-embeddinggemma2-decision")
    result = client.evaluate_with_trace(request)
    response = result["response"]

    assert response["usage"]["input_tokens"] > 0
    assert response["answers"]["team"]["choice"] == "technical"
    assert response["answers"]["urgent"]["noul"] == pytest.approx(0.7310586, rel=1e-5)
    assert response["answers"]["severity"]["score"] > 1.0
    assert result["trace"]["prompt_diagnostics"]["renderer"] == "single_sequence_question_option_span_pooling"


def test_clef_checkpoint_loads_with_weights_only(tmp_path):
    torch = pytest.importorskip("torch")

    path = tmp_path / "head.pt"
    torch.save({"d_model": 768, "head": {"weight": torch.ones(2)}, "config": {"max_len": 1024}}, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["d_model"] == 768
    assert checkpoint["head"]["weight"].shape == (2,)


def test_native_probability_schema_is_checked():
    config = ModelConfig.load(ROOT / "configs/embeddinggemma-2-ft-typed-decision.yaml")
    backend = GemmaTypedDecisionBackend(config=config, model=FakeGemmaDecisionModel())
    request = sample_request()
    plans = compile_typed_decision_request(request, config.backend)
    bad = FakeGemmaDecisionModel()
    bad.decide = lambda *args, **kwargs: {"team": SimpleNamespace(probabilities={"billing": float("nan"), "technical": 0.0})}
    backend._model = bad
    with pytest.raises(BackendError, match="invalid probabilities"):
        backend.evaluate(request, plans)


def test_gemma_loader_uses_same_pinned_snapshot_for_model_and_tokenizer(monkeypatch, tmp_path):
    import sys
    import types

    config = ModelConfig.load(ROOT / "configs/embeddinggemma-2-ft-typed-decision.yaml")
    calls = {"snapshot": [], "model": [], "tokenizer": []}

    class FakeModel(FakeGemmaDecisionModel):
        def eval(self):
            return self

        def to(self, device):
            calls["device"] = device
            return self

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "5.19.0"
    fake_transformers.AutoModel = types.SimpleNamespace(from_pretrained=lambda path, **kwargs: (
        calls["model"].append((path, kwargs)) or FakeModel()))
    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path, **kwargs: (
        calls["tokenizer"].append((path, kwargs)) or object()))
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.__version__ = "1.31.0"
    fake_hub.snapshot_download = lambda **kwargs: (calls["snapshot"].append(kwargs) or str(tmp_path))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    backend = GemmaTypedDecisionBackend(config=config)
    metadata = backend.prepare()
    assert calls["snapshot"] == [{"repo_id": config.model_name_or_path, "revision": config.revision,
                                  "local_files_only": config.local_files_only}]
    assert calls["model"][0][0] == str(tmp_path)
    assert calls["model"][0][1]["trust_remote_code"] is True
    assert calls["model"][0][1]["local_files_only"] is True
    assert calls["tokenizer"][0] == (str(tmp_path), {"local_files_only": True, "trust_remote_code": False})
    assert backend._model._tokenizer is not None
    assert metadata["dimension"] == 768


def test_gemma_typed_decision_rejects_custom_noul_criteria():
    request = sample_request()
    request["questions"]["urgent"]["criteria"] = {"false": "No", "true": "Yes"}
    with pytest.raises(ValidationError, match="custom Noul"):
        compile_typed_decision_request(request, "gemma_typed_decision")


def test_configs_pin_code_and_disable_remote_code_for_local_clef_adapter():
    gemma = ModelConfig.load(ROOT / "configs/embeddinggemma-2-ft-typed-decision.yaml")
    clef = ModelConfig.load(ROOT / "configs/qnaug-embeddinggemma2-decision.yaml")
    assert gemma.revision == gemma.code_revision and gemma.trust_remote_code
    assert not clef.trust_remote_code
    assert gemma.revision == "3d0069b4c56d7a7f505feb53bee90694b2c07845"
    assert len(clef.revision) == 40


def test_qnaug_example_uses_its_dedicated_model_and_covers_typed_decisions():
    import json

    request = json.loads((ROOT / "examples/qnaug_embeddinggemma_decide.json").read_text(encoding="utf-8"))
    config = ModelConfig.load(ROOT / "configs/qnaug-embeddinggemma2-decision.yaml")
    assert request["model"] == config.model_id
    result = JevEmbed(config=config).explain(request)
    assert {task["path"] for task in result["tasks"]} == {
        "native_clef_gemma_decision",
    }
    assert {task["question_id"] for task in result["tasks"]} == {"team", "urgent", "severity"}
