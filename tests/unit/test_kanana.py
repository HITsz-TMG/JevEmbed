"""Kanana configuration and native decision-input compilation tests."""

from dataclasses import replace
import json
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

from conftest import FixedBackend, ROOT
from jevembed import BackendError, JevEmbed, ModelConfig, ValidationError
from jevembed.backends import EmbeddingInput, KananaDecideBackend
from jevembed.backends.kanana_decide import compile_kanana_request


def kanana_config(**changes):
    return replace(ModelConfig.load(ROOT / "configs/kanana-2-v-decide-3b-2611.yaml"), **changes)


def test_shipped_kanana_config_and_explain_are_offline():
    config = ModelConfig.load(ROOT / "configs/kanana-2-v-decide-3b-2611.yaml")
    assert config.backend == "kanana_decide"
    assert config.model_name_or_path == "kakaocorp/kanana-2-v-decide-3b-2611"
    assert config.expected_dimension == 2560
    assert config.revision == "12820d654ed9789a0f7f48daeee1703c3483c464"

    client = JevEmbed(config=config)
    request = {"model": config.aliases[0], "state": {"z": "한글", "a": [True, None]},
               "questions": {"route": {"type": "choice", "instructions": {"z": 1, "a": 2},
                                       "criteria": {"empty": "", "missing": None, "json": {"z": 3, "a": 4}}}}}
    trace = client.explain(request)
    assert trace["model"] == config.model_id
    assert client._models[config.model_id].backend._model is None
    inputs = trace["tasks"][0]["inputs"]
    assert json.loads(inputs[0]["text"]) == {"state": request["state"], "question": request["questions"]["route"]}
    assert [item["text"] for item in inputs[1:]] == ["empty", "missing", 'json: {"z":3,"a":4}']


@pytest.mark.parametrize("fixture_name", ["choice_exchange", "score_safari", "noul_escalation"])
def test_native_input_compilation_preserves_state_question_and_candidate_order(fixture_name):
    request = json.loads((ROOT / f"tests/fixtures/official_{fixture_name}.request.json").read_text(encoding="utf-8"))
    plans = compile_kanana_request(request, kanana_config().prompts)
    for plan, (qid, question) in zip(plans, request["questions"].items()):
        assert plan.question_id == qid
        assert plan.inputs[0].role == "query"
        payload = json.loads(plan.inputs[0].text)
        assert payload == {"state": request["state"], "question": question}
        assert all(item.role == "document" for item in plan.inputs[1:])
        if question["type"] == "choice":
            assert plan.labels == list(question["criteria"])
        elif question["type"] == "score":
            assert plan.labels == [str(i) for i in range(len(question["criteria"]))]
        else:
            assert plan.labels == ["true", "false"]


def test_kanana_scoring_uses_native_temperature_and_reuses_framework_cache():
    config = kanana_config(expected_dimension=2)
    class NativeVectors(FixedBackend):
        def encode(self, items):
            from jevembed.backends import EmbeddingBatch
            self.calls.append(items)
            return EmbeddingBatch([[0, 1] if item.text.startswith("rest:") or item.text.startswith("false:")
                                   else [1, 0] for item in items], len(items) * 7)
    backend = NativeVectors()
    client = JevEmbed(config=config, backend=backend)
    request = {"state": "state", "questions": {"q": {"type": "choice", "instructions": "pick",
              "criteria": {"best": "preferred", "rest": "other"}}}}
    first = client.evaluate(request)
    second = client.evaluate(request)
    assert first["answers"]["q"]["choice"] == "best"
    expected = 1 / (1 + math.exp(-1 / config.scoring.choice_temperature))
    assert first["answers"]["q"]["probabilities"]["best"] == pytest.approx(expected)
    assert first["answers"]["q"]["confidence"] == pytest.approx(2 * expected - 1)
    assert second["usage"]["input_tokens"] == 0
    assert len(backend.calls) == 1
    request["questions"]["n"] = {"type": "noul", "instructions": "is true?"}
    assert client.evaluate(request)["answers"]["n"]["noul"] == pytest.approx(expected)
    request["questions"]["n"]["criteria"] = {"true": "yes", "false": "no"}
    assert client.evaluate(request)["answers"]["n"]["noul"] == pytest.approx(expected)
    # Candidates can be reused after the candidate set changes, while the query
    # must be recomputed because the original option schema is part of it.
    request["questions"] = {"q": {**request["questions"]["q"], "criteria": {"best": "preferred"}}}
    changed = client.evaluate_with_trace(request)
    assert changed["trace"]["cache_hits"] == 1
    assert changed["trace"]["encoded_inputs"] == 1


def test_kanana_requires_trusted_custom_code():
    config = ModelConfig.load(ROOT / "configs/kanana-2-v-decide-3b-2611.yaml")
    with pytest.raises(ValidationError, match="trust_remote_code=true"):
        KananaDecideBackend(config=replace(config, trust_remote_code=False))
    with pytest.raises(ValidationError, match="snapshot revision"):
        KananaDecideBackend(config=replace(config, code_revision="other"))


@pytest.mark.parametrize("criteria", [{"": "empty label"}, {"a": "a", "b": ""}])
def test_empty_labels_rejected_or_empty_descriptions_use_label(criteria):
    request = {"state": "state", "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": criteria}}}
    if "" in criteria:
        with pytest.raises(ValidationError, match="nonempty"):
            compile_kanana_request(request, kanana_config().prompts)
    else:
        assert compile_kanana_request(request, kanana_config().prompts)[0].inputs[-1].text == "b"


@pytest.fixture
def fake_kanana_runtime(monkeypatch, tmp_path):
    state = {"load": [], "snapshots": [], "encoded": []}
    torch = ModuleType("torch")
    torch.float32 = "float32"
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    reference = SimpleNamespace()
    reference.message_ids = lambda messages, *args: list(range(messages["tokens"]))
    reference.validate_candidates = lambda *args, **kwargs: None
    def build_pairs(original_state, questions):
        return {"q": ({"state": original_state, "question": questions["q"]}, ["a"], ["a"])}
    reference.build_pairs = build_pairs
    def query_messages(payload, tokenizer, processor, maximum, **kwargs):
        state["payload"] = payload
        state["query_kwargs"] = kwargs
        return {"tokens": min(12, maximum) if kwargs.get("truncate") else 12, "kind": "query"}
    reference.query_messages = query_messages
    reference.document_messages = lambda text, tokenizer, processor, maximum, **kwargs: {
        "tokens": min(9, maximum) if kwargs.get("truncate") else 9, "kind": "document"}
    monkeypatch.setitem(sys.modules, "kanana_fixture.decision_kanana_v_decide", reference)

    class Model:
        __module__ = "kanana_fixture.modeling_kanana_v_decide"
        config = SimpleNamespace(model_type="kanana_v_decide", embedding_dimension=2, max_length=20480,
                                 decision_temperature=0.056139365295102696, renderer="kanana-v-decide")
        processor = SimpleNamespace(tokenizer=object())
        def eval(self):
            return self
        def to(self, device):
            state["device"] = device
            return self
        def embed_messages(self, messages):
            state["encoded"].append(messages)
            return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [[1, 0] for _ in messages])), sum(m["tokens"] for m in messages)

    def load(path, **kwargs):
        state["load"].append((path, kwargs))
        return Model()
    transformers = ModuleType("transformers")
    transformers.AutoModel = SimpleNamespace(from_pretrained=load)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    hub = ModuleType("huggingface_hub")
    def snapshot(repo, **kwargs):
        state["snapshots"].append((repo, kwargs))
        return str(tmp_path)
    hub.snapshot_download = snapshot
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr("jevembed.backends.kanana_decide.importlib.metadata.version", lambda name: "test-version")
    (tmp_path / "config.json").write_text("{}")
    state["reference"] = reference
    return state


def test_snapshot_loading_is_pinned_and_processor_is_local(fake_kanana_runtime):
    state = fake_kanana_runtime
    backend = KananaDecideBackend(config=kanana_config(expected_dimension=2, local_files_only=True))
    assert state["load"] == []
    metadata = backend.prepare()
    _, kwargs = state["snapshots"][0]
    assert kwargs["revision"] == kanana_config().revision
    assert kwargs["local_files_only"] is True
    assert state["load"][0][1] == {"trust_remote_code": True, "local_files_only": True,
                                  "dtype": "float32", "attn_implementation": "sdpa"}
    assert metadata["dimension"] == 2
    assert "config.json" in metadata["files_sha256"]


def test_native_renderer_forward_usage_and_truncation(fake_kanana_runtime):
    state = fake_kanana_runtime
    backend = KananaDecideBackend(config=kanana_config(expected_dimension=2, max_input_tokens=10))
    request = {"state": {"z": 1}, "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {"a": None}}}}
    plan = compile_kanana_request(request, kanana_config().prompts)[0]
    batch = backend.encode(plan.inputs)
    assert state["payload"] == {"state": request["state"], "question": request["questions"]["q"]}
    assert batch.vectors == [[1, 0], [1, 0]]
    assert batch.input_tokens == 19
    assert state["query_kwargs"] == {"truncate": True, "truncate_question_candidates": True}
    assert batch.metadata["truncated"] == [{"index": 0, "original_tokens": 12, "encoded_tokens": 10,
                                            "kept": "state_head_and_tail"}]
    strict = KananaDecideBackend(config=kanana_config(expected_dimension=2, max_input_tokens=10, overflow_policy="error"))
    with pytest.raises(ValidationError, match="exceeds 10"):
        strict.encode(plan.inputs)
    assert len(state["encoded"]) == 1
    with pytest.raises(ValidationError, match="Unknown embedding role"):
        backend.encode([EmbeddingInput("invalid", "", "state")])


def test_bad_dimension_or_unsupported_limit_fails_before_encoding(fake_kanana_runtime):
    with pytest.raises(BackendError, match="dimension"):
        KananaDecideBackend(config=kanana_config()).prepare()
    with pytest.raises(ValidationError, match="exceeds model limit"):
        KananaDecideBackend(config=kanana_config(expected_dimension=2, max_input_tokens=20481)).prepare()
