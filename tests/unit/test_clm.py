import json
from dataclasses import replace

import pytest

from conftest import FixedBackend, ROOT
from jevembed import JevEmbed, ModelConfig
from jevembed.scoring import sigmoid, top_margin
from jevembed.serialization import serialize_for_mode


def clm_client(backend=None):
    config = ModelConfig.load(ROOT / "configs/clm-v0.1-8b.yaml")
    return JevEmbed(config=config, backend=backend or FixedBackend())


def test_clm_prompt_mapping_matches_reference_schema():
    client = clm_client()
    cases = (
        ("choice_exchange", [
            "My running shoes arrived in the wrong size. Can I swap them for a size 10?\n\nWhich team should handle this?",
            "Exchanges, wrong or damaged items",
            "Delivery status, delays, lost packages",
            "Charges, invoices, payment problems",
        ], ["query", "document", "document", "document"]),
        ("score_safari", [
            "The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari.\n\nHow severe is the reported issue?",
            "Cosmetic; no impact to functionality",
            "Broken or degraded feature, but workaround exists",
            "Blocking issue; no workaround exists",
        ], ["query", "document", "document", "document"]),
    )
    for name, expected, roles in cases:
        request = json.loads((ROOT / f"tests/fixtures/official_{name}.request.json").read_text())
        request["model"] = "clm-v0.1-8b"
        inputs = client.explain(request)["tasks"][0]["inputs"]
        assert [item["rendered"] for item in inputs] == expected
        assert [item["role"] for item in inputs] == roles

    request = json.loads((ROOT / "tests/fixtures/official_noul_escalation.request.json").read_text())
    request["model"] = "clm-v0.1-8b"
    plans = client.explain(request)["tasks"]
    assert [item["rendered"] for item in plans[0]["inputs"]] == [
        "I have asked three times now. Can I please just talk to a real person?\n\nIs the customer asking for a human agent?",
        "true: Yes. This is true: Is the customer asking for a human agent?",
        "false: No. This is false: Is the customer asking for a human agent?",
    ]
    assert [item["rendered"] for item in plans[1]["inputs"]] == [
        "I have asked three times now. Can I please just talk to a real person?\n\nHas the customer contacted support about this before?",
        "true: Mentions a prior attempt, ticket, or that they have asked before",
        "false: No sign of any previous contact",
    ]
    assert all([item["role"] for item in plan["inputs"]] == ["query", "document", "document"]
               for plan in plans)


def test_clm_prose_serialization_preserves_field_order():
    value = {"state": "Ready", "details": {"a": 1, "b": ["x", "y"]}, "ok": True}
    assert serialize_for_mode(value, "prose") == (
        "state: Ready\n\ndetails:\n  a: 1\n  b:\n    - x\n    - y\n\nok: true")


def test_clm_reuses_binary_scoring_and_top_margin():
    config = ModelConfig.load(ROOT / "configs/clm-v0.1-8b.yaml")
    config = replace(config, backend="custom", projection_name_or_path=None,
                     projection_filename=None, projection_revision=None, pooling="model_default",
                     expected_dimension=None)
    backend = FixedBackend({"state": [1, 0], "true: Yes. This is true: instruction": [1, 0],
                            "false: No. This is false: instruction": [0, 1],
                            "true: yes": [1, 0], "false: no": [0, 1],
                            "alpha": [1, 0], "beta": [0, 1]})
    client = JevEmbed(config=config, backend=backend)
    without = {"state": "state", "questions": {"q": {"type": "noul", "instructions": "instruction"}}}
    with_criteria = {"state": "state", "questions": {"q": {"type": "noul", "instructions": "instruction",
                       "criteria": {"true": "yes", "false": "no"}}}}
    assert client.evaluate(without)["answers"]["q"]["noul"] == pytest.approx(sigmoid(100))
    assert client.evaluate(with_criteria)["answers"]["q"]["noul"] == pytest.approx(sigmoid(100))
    choice = {"state": "state", "questions": {"q": {"type": "choice", "instructions": "instruction",
                   "criteria": {"a": "alpha", "b": "beta"}}}}
    answer = client.evaluate(choice)["answers"]["q"]
    assert answer["choice"] == "a"
    assert answer["confidence"] == pytest.approx(top_margin(list(answer["probabilities"].values())))
