from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest

from conftest import FixedBackend, ROOT, request
from jevembed import JevEmbed, ModelConfig, PromptConfig, ScoringConfig, LogisticConfig, ValidationError, BackendError
from jevembed.backends import EmbeddingBatch
from jevembed.scoring import normalize_vectors, normalized_entropy, sigmoid, softmax
from jevembed.serialization import serialize


def test_official_compilation(client, fixtures):
    choice, score, noul = [fixtures[n] for n in ("choice_exchange", "score_safari", "noul_escalation")]
    for r in (choice, score, noul):
        r["model"] = "test"
    c = client.explain(choice)["tasks"][0]["inputs"]
    assert c[0]["rendered"] == "Instruct: Which team should handle this?\nQuery: My running shoes arrived in the wrong size. Can I swap them for a size 10?"
    assert [i["rendered"] for i in c[1:]] == ["returns: Exchanges, wrong or damaged items", "shipping: Delivery status, delays, lost packages", "billing: Charges, invoices, payment problems"]
    s = client.explain(score)["tasks"][0]["inputs"]
    assert s[0]["rendered"] == "Instruct: How severe is the reported issue?\nQuery: The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari."
    assert [i["rendered"] for i in s[1:]] == score["questions"]["bug_severity"]["criteria"]
    n = client.explain(noul)["tasks"]
    assert [i["rendered"] for i in n[0]["inputs"]] == [
        "Instruct: Retrieve semantically similar text.\nQuery: I have asked three times now. Can I please just talk to a real person?",
        "Instruct: Retrieve semantically similar text.\nQuery: Is the customer asking for a human agent?"]
    assert all(i["role"] == "query" for i in n[0]["inputs"])
    assert [i["rendered"] for i in n[1]["inputs"]] == [
        "Instruct: Has the customer contacted support about this before?\nQuery: I have asked three times now. Can I please just talk to a real person?",
        "Mentions a prior attempt, ticket, or that they have asked before", "No sign of any previous contact"]


def test_official_reference():
    data = json.loads((ROOT / "tests/fixtures/official_score_safari.reference.json").read_text())
    answer = data["answers"]["bug_severity"]
    assert sum(int(k)*p for k, p in answer["probabilities"].items()) == pytest.approx(answer["score"])


def test_explain_never_encodes(client, backend):
    client.explain(request())
    assert backend.calls == []


def test_question_id_and_unrelated_question_independence(client, backend):
    r = request()
    answer = client.evaluate(r)["answers"]["q"]
    renamed = {**r, "questions": {"new": r["questions"]["q"]}}
    assert client.evaluate(renamed)["answers"]["new"] == answer
    assert len(backend.calls) == 1
    renamed["questions"]["unrelated"] = {"type": "score", "instructions": "other", "criteria": ["first", "last"]}
    assert client.evaluate(renamed)["answers"]["new"] == answer


def test_structures_unicode_null_and_single(client):
    r = request(criteria={"独有": None})
    r["state"] = {"z": [True, None, {"ü": 1}], "a": "保留"}
    r["questions"]["q"]["instructions"] = ["原始", {"规则": False}]
    trace = client.explain(r)
    assert trace["tasks"][0]["inputs"][0]["text"] == '{"a":"保留","z":[true,null,{"ü":1}]}'
    assert trace["tasks"][0]["inputs"][1]["text"] == "独有"
    a = client.evaluate(r)["answers"]["q"]
    assert a["probabilities"] == {"独有": 1} and a["confidence"] == 1
    r["questions"]["q"]["criteria"] = {"a": {"z": 3, "b": [None]}}
    assert client.explain(r)["tasks"][0]["inputs"][1]["text"] == 'a: {"b":[null],"z":3}'


def test_choice_reorder_and_tie():
    b = FixedBackend({"state": [1, 0], "a": [1, 0], "b": [1, 0]})
    c = JevEmbed(backend=b, model="test")
    a = c.evaluate(request(criteria={"b": None, "a": None}))["answers"]["q"]
    assert a["choice"] == "b"  # first in request wins exact ties
    a2 = c.evaluate(request(criteria={"a": None, "b": None}))["answers"]["q"]
    assert a2["choice"] == "a" and a2["probabilities"] == a["probabilities"]


def test_choice_limit(client):
    assert len(client.evaluate(request(criteria={str(i): None for i in range(255)}))["answers"]["q"]["probabilities"]) == 255
    with pytest.raises(ValidationError):
        client.evaluate(request(criteria={str(i): None for i in range(256)}))


def test_score_expectation_and_legend(client):
    levels = ["normal", {"nested": [1, False, None]}, ["high", 3]]
    answer = client.evaluate(request("score", levels))["answers"]["q"]
    assert answer["legend"] == dict(zip(["0", "1", "2"], levels))
    assert answer["score"] == pytest.approx(answer["probabilities"]["1"] + 2*answer["probabilities"]["2"])


def test_noul_parameters_and_order():
    cfg = ModelConfig(model_id="test", backend="custom", scoring=ScoringConfig(
        noul_criteria=LogisticConfig(2, -1), noul_similarity=LogisticConfig(3, -2)))
    c = JevEmbed(config=cfg, backend=FixedBackend({"state": [1, 0], "yes": [1, 0], "no": [0, 1], "instruction": [1, 0]}))
    r = request("noul", {"false": "no", "true": "yes"})
    assert c.evaluate(r)["answers"]["q"] == {"type": "noul", "noul": sigmoid(1)}
    assert [i["text"] for i in c.explain(r)["tasks"][0]["inputs"]][1:] == ["yes", "no"]
    assert c.evaluate(request("noul"))["answers"]["q"]["noul"] == sigmoid(1)


@pytest.mark.parametrize("kind,criteria", [("choice", {}), ("choice", []), ("choice", {"a": 3}),
    ("score", []), ("score", ["a"]), ("score", ["a"]*11), ("score", [True, "x"]),
    ("noul", {}), ("noul", {"true": "a"}), ("noul", {"true": "a", "false": None}), ("unknown", [])])
def test_invalid_questions(client, kind, criteria):
    with pytest.raises(ValidationError):
        client.evaluate(request(kind, criteria))


@pytest.mark.parametrize("value", [None, 1, True, float("nan"), {"x": float("inf")}, {1: "x"}, (1, 2), {"x": object()}])
def test_invalid_state(client, value):
    with pytest.raises(ValidationError):
        client.evaluate({**request(), "state": value})


def test_null_noul_unknown_model_fields_and_cycles(client):
    r = request("noul")
    r["questions"]["q"]["criteria"] = None
    with pytest.raises(ValidationError):
        client.evaluate(r)
    for r in [{**request(), "model": "jev-latest"}, {**request(), "extra": 3}, {"state": "x", "questions": {}}]:
        with pytest.raises(ValidationError):
            client.evaluate(r)
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValidationError):
        serialize(cycle)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "1"])
def test_invalid_temperature(value):
    with pytest.raises(ValidationError):
        ScoringConfig(choice_temperature=value)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), True, "1"])
def test_invalid_logistic(value):
    with pytest.raises(ValidationError):
        LogisticConfig(slope=value)


def test_stable_math():
    assert softmax([1, -1], 1e-320) == [1, 0]
    assert normalized_entropy([.5, .5]) == pytest.approx(0)
    assert normalized_entropy([1, 0]) == 1
    assert normalized_entropy([1]) == 1
    assert sigmoid(-1000) == 0 and sigmoid(1000) == 1
    assert normalize_vectors([[1e308, 1e308]], 1)[0] == pytest.approx([2**-.5]*2)


@pytest.mark.parametrize("vectors,count", [([], 1), ([[]], 1), ([[0, 0]], 1), ([[float("nan")]], 1),
    ([[float("inf")]], 1), ([[1], [1, 2]], 2), ([[1], [1]], 1)])
def test_bad_vectors(vectors, count):
    with pytest.raises(BackendError):
        normalize_vectors(vectors, count)


def test_cache_usage_capacity_clear_and_concurrency(client, backend):
    r = request()
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(client.evaluate, [r]*4))
    assert sorted(a["usage"]["input_tokens"] for a in results) == [0, 0, 0, 21]
    assert len(backend.calls) == 1
    client.clear_cache()
    assert client.evaluate(r)["usage"]["input_tokens"] == 21
    tiny = JevEmbed(config=ModelConfig(model_id="tiny", backend="custom", cache_capacity=1), backend=FixedBackend())
    tiny.evaluate(r)
    assert tiny.evaluate(r)["usage"]["input_tokens"] == 14


def test_alias_model_and_config_isolation():
    c = JevEmbed()
    for i in range(3):
        c.register(ModelConfig(model_id=str(i), backend="custom", revision=str(i),
                               prompts=PromptConfig(query_template=f"v{i} {{instruction}} {{text}}")), backend=FixedBackend(), aliases=[f"alias{i}"])
    for i in range(3):
        result = c.evaluate({**request(), "model": f"alias{i}"})
        assert result["model"] == str(i) and result["usage"]["input_tokens"] == 21
    with pytest.raises(ValidationError):
        c.register(ModelConfig(model_id="0"), backend=FixedBackend())


def test_failure_does_not_commit_partial_cache():
    class Broken(FixedBackend):
        def encode(self, items):
            batch = super().encode(items)
            if len(self.calls) == 2:
                raise RuntimeError("broken")
            return batch
    b = Broken()
    c = JevEmbed(config=ModelConfig(model_id="test", backend="custom", batch_size=1), backend=b)
    with pytest.raises(BackendError):
        c.evaluate(request())
    assert c.evaluate(request())["usage"]["input_tokens"] == 21


def test_missing_usage():
    class NoUsage(FixedBackend):
        def encode(self, items):
            return EmbeddingBatch(super().encode(items).vectors, None)
    with pytest.raises(BackendError):
        JevEmbed(backend=NoUsage(), model="test").evaluate(request())
    c = JevEmbed(config=ModelConfig(model_id="test", backend="custom", usage_mode="estimate"), backend=NoUsage())
    result = c.evaluate_with_trace(request())
    assert result["trace"]["usage_sources"] == ["estimated_utf8_bytes_div4"]


def test_default_configs_and_dropped_instruction():
    assert ModelConfig().trust_remote_code is False
    for name, trust in [("kalm-embedding-v2.5", True), ("qwen3-embedding-0.6b", False),
                        ("multilingual-e5-large-instruct", False), ("qwen3-embedding-4b", False)]:
        cfg = ModelConfig.load(ROOT / f"configs/{name}.yaml")
        assert cfg.trust_remote_code is trust and cfg.local_files_only is False
        assert cfg.model_name_or_path in cfg.aliases
    with pytest.raises(ValidationError):
        ModelConfig.from_dict({"trust_remote_code": "false"})
    with pytest.raises(ValidationError):
        PromptConfig(query_template="{unknown}")
    c = JevEmbed(config=ModelConfig(model_id="test", backend="custom", prompts=PromptConfig(query_template="query: {text}")), backend=FixedBackend())
    assert c.explain(request())["prompt_diagnostics"]["instruction_preserved"] is False


def test_mixed_request_and_noul_query_dedup(client, backend, fixtures):
    questions = {}
    for r in fixtures.values():
        questions.update(r["questions"])
    questions["second_similarity"] = {"type": "noul", "instructions": "other"}
    r = {"state": "shared", "questions": questions}
    result = client.evaluate(r)
    assert set(result["answers"]) == set(questions)
    inputs = [i for call in backend.calls for i in call]
    assert len([i for i in inputs if i.text == "shared" and i.instruction == "Retrieve semantically similar text."]) == 1
