"""Fixed-question synthesis accepts only schema-valid states and hard labels."""

import json
from pathlib import Path

import pytest

from jevembed import ValidationError
from jevembed.synthesis import (load_config, parse_json, validate_answer,
                                validate_config, validate_generation, validate_state)
from jevembed.synthesis.quotas import quota_answers


ROOT = Path(__file__).resolve().parents[2]


def _raw(question=None, schema=None):
    return {
        "version": 1,
        "teacher": {"base_url": "http://127.0.0.1:8000/v1", "model": "teacher"},
        "task": {
            "id": "support-routing", "domain": "Customer support", "question_id": "route request",
            "question": question or {"type": "choice", "instructions": "Select a team.",
                                     "criteria": {"billing": None, "shipping": "Delivery questions"}},
            "state_schema": schema or {"$schema": "https://json-schema.org/draft/2020-12/schema",
                                       "type": "string", "minLength": 3},
            "state_description": "A customer's support message; distances are in kilometers when present.",
        },
    }


def test_defaults_and_key_only_choice_are_preserved():
    config = validate_config(_raw())
    assert config["teacher"]["timeout"] == 120
    assert config["teacher"]["temperature"] == 0.7
    assert config["generation"]["case_types"] == ["standard", "boundary", "hard_negative"]
    assert config["task"]["question"]["criteria"] == {"billing": None, "shipping": "Delivery questions"}
    assert validate_generation({"state": "late parcel", "answer": {"choice": "shipping"}}, config) == (
        "late parcel", {"choice": "shipping"})
    with pytest.raises(ValidationError):
        validate_answer({"choice": "unknown"}, config)


@pytest.mark.parametrize("answer", [
    {"probabilities": {"billing": 1, "shipping": 0}}, {"choice": 0}, {"choice": "shipping", "extra": 1},
])
def test_choice_rejects_soft_or_malformed_labels(answer):
    with pytest.raises(ValidationError):
        validate_answer(answer, validate_config(_raw()))


def test_score_keeps_order_and_requires_integer_level():
    question = {"type": "score", "instructions": "Rate the severity.",
                "criteria": ["Low impact", "Medium impact", "High impact"]}
    config = validate_config(_raw(question))
    assert config["task"]["question"]["criteria"] == question["criteria"]
    validate_answer({"level": 2}, config)
    for answer in ({"level": True}, {"level": 2.0}, {"level": 3}, {"score": 1.5}):
        with pytest.raises(ValidationError):
            validate_answer(answer, config)


@pytest.mark.parametrize("criteria", [None, {"true": "Requests a person", "false": "Self-service"}])
def test_both_noul_paths_require_boolean(criteria):
    question = {"type": "noul", "instructions": "Does the customer ask for a person?"}
    if criteria is not None:
        question["criteria"] = criteria
    config = validate_config(_raw(question))
    validate_answer({"noul": True}, config)
    validate_answer({"noul": False}, config)
    for value in (0, 1, 0.5, "true"):
        with pytest.raises(ValidationError):
            validate_answer({"noul": value}, config)


def test_nested_json_schema_constraints_and_local_reference():
    schema = {
        "type": "object", "required": ["title", "items"], "additionalProperties": False,
        "properties": {
            "title": {"type": "string", "minLength": 3},
            "items": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/item"}},
        },
        "$defs": {"item": {"type": "object", "required": ["amount"],
                            "properties": {"amount": {"type": "number", "minimum": 0}}}},
    }
    config = validate_config(_raw(schema=schema))
    valid = {"title": "Late delivery", "items": [{"amount": 0.5}]}
    validate_state(valid, config)
    for invalid in ({"title": "ok", "items": [{"amount": 2}]},
                    {"title": "Late delivery", "items": [{"amount": -1}]},
                    {"title": "Late delivery", "items": []},
                    {**valid, "extra": True}):
        with pytest.raises(ValidationError):
            validate_state(invalid, config)
    with pytest.raises(ValidationError):
        validate_generation({"state": valid, "answer": {"choice": "billing"}, "extra": 1}, config)


@pytest.mark.parametrize("schema", [
    {"type": "number"},
    {"type": ["string", "null"]},
    {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"},
    {"type": "object", "properties": {"item": {"$ref": "https://example.com/item.json"}}},
    {"type": "object", "properties": {"item": {"$dynamicRef": "file:///tmp/item"}}},
    {"type": "object", "properties": {"item": {"$ref": "#/$defs/missing"}}},
    {"type": "object", "properties": {"item": {"$ref": "#missing-anchor"}}},
    {"type": "object", "properties": {"item": {"$schema": "http://json-schema.org/draft-07/schema#"}}},
    {"type": "object", "minProperties": "many"},
])
def test_schema_rejects_unsupported_root_draft_remote_refs_and_malformed_keywords(schema):
    with pytest.raises(ValidationError):
        validate_config(_raw(schema=schema))


@pytest.mark.parametrize("change", [
    lambda raw: raw.update(extra=1),
    lambda raw: raw["teacher"].update(timeout=True),
    lambda raw: raw["teacher"].update(temperature=float("nan")),
    lambda raw: raw["teacher"].update(request_options={"messages": []}),
    lambda raw: raw["teacher"].update(request_options={"chat_template_kwargs": {"auth": "secret"}}),
    lambda raw: raw["teacher"].update(request_options={"n": 2}),
    lambda raw: raw["teacher"].update(request_options={"max_completion_tokens": 4096}),
    lambda raw: raw["teacher"].update(request_options={"tools": [{"type": "function"}]}),
    lambda raw: raw["teacher"].update(base_url="https://user:pass@example.com/v1"),
    lambda raw: raw["task"].update(state_description="  "),
    lambda raw: raw.setdefault("generation", {}).update(count=True),
    lambda raw: raw.setdefault("generation", {}).update(case_types=["standard", "standard"]),
])
def test_config_rejects_unknown_unsafe_or_nonfinite_values(change):
    raw = _raw()
    change(raw)
    with pytest.raises(ValidationError):
        validate_config(raw)


def test_zero_temperature_and_qwen_request_option_are_valid():
    raw = _raw()
    raw["teacher"].update(temperature=0, request_options={"chat_template_kwargs": {"enable_thinking": False}})
    config = validate_config(raw)
    assert config["teacher"]["temperature"] == 0
    assert config["teacher"]["request_options"]["chat_template_kwargs"]["enable_thinking"] is False


def test_fixed_question_rejects_empty_semantics_and_boolean_yaml_keys():
    for question in (
        {"type": "choice", "instructions": "  ", "criteria": {"a": None}},
        {"type": "choice", "instructions": "Choose", "criteria": {" ": None}},
        {"type": "score", "instructions": "Rate", "criteria": ["Low", " "]},
        {"type": "noul", "instructions": "Judge", "criteria": {True: "Yes", False: "No"}},
    ):
        with pytest.raises(ValidationError):
            validate_config(_raw(question))


def test_choice_has_no_arbitrary_255_candidate_limit():
    question = {"type": "choice", "instructions": "Select a category.",
                "criteria": {f"category-{index}": None for index in range(260)}}
    config = validate_config(_raw(question))
    validate_answer({"choice": "category-259"}, config)


@pytest.mark.parametrize("text", [
    '{"state":"late","answer":{"choice":"billing","choice":"shipping"}}',
    '{"state":NaN,"answer":{"choice":"billing"}}',
    '{"state":1e999,"answer":{"choice":"billing"}}',
    'Here is JSON: {"state":"late","answer":{"choice":"billing"}}',
    '```json\n{"state":"late","answer":{"choice":"billing"}}\n```\nextra',
])
def test_parser_rejects_duplicate_nonfinite_or_embedded_json(text):
    with pytest.raises(ValidationError):
        parse_json(text)


def test_parser_accepts_only_plain_json_or_one_enclosing_fence():
    payload = {"state": "late", "answer": {"choice": "billing"}}
    import json
    encoded = json.dumps(payload)
    assert parse_json(encoded) == payload
    assert parse_json(f"```json\n{encoded}\n```") == payload


def test_yaml_rejects_duplicates_and_preserves_null_choice(tmp_path):
    raw = _raw()
    import yaml
    source = yaml.safe_dump(raw, sort_keys=False)
    path = tmp_path / "synthesis.yaml"
    path.write_text(source)
    config = load_config(path)
    assert config["task"]["question"]["criteria"]["billing"] is None
    path.write_text(source + "version: 1\n")
    with pytest.raises(ValidationError, match="Duplicate YAML key"):
        load_config(path)


def test_generation_requires_exact_payload_shape():
    config = validate_config(_raw())
    for payload in ({"state": "late"}, {"state": "late", "answer": {"choice": "billing"}, "note": "x"}):
        with pytest.raises(ValidationError):
            validate_generation(payload, config)


def test_shipped_examples_load_and_choice_schema_enforces_flat_integer_state():
    configs = {name: load_config(ROOT / "configs/synthesis" / name) for name in (
        "choice-dinosaur.yaml", "score-risk.yaml", "noul-human-escalation.yaml",
        "noul-repeat-contact.yaml")}
    choice = configs["choice-dinosaur.yaml"]
    assert choice["task"]["question"]["criteria"] == {"Jump": None, "Keep running": None}
    validate_generation({"state": {"distance_to_obstacle": 5}, "answer": {"choice": "Jump"}}, choice)
    for state in ({}, {"distance_to_obstacle": 5, "extra": 1},
                  {"distance_to_obstacle": 5.5}, {"distance_to_obstacle": -1}):
        with pytest.raises(ValidationError):
            validate_state(state, choice)

    official = json.loads((ROOT / "tests/fixtures/official_noul_escalation.request.json").read_text())
    for filename, question_id, has_criteria in (
        ("noul-human-escalation.yaml", "is_human_escalation", False),
        ("noul-repeat-contact.yaml", "is_repeat_contact", True),
    ):
        question = configs[filename]["task"]["question"]
        assert question == official["questions"][question_id]
        assert ("criteria" in question) is has_criteria
        validate_answer({"noul": True}, configs[filename])
        validate_answer({"noul": False}, configs[filename])


def test_quota_defaults_explicit_zero_and_idempotent_normalization():
    assert quota_answers(validate_config(_raw())) is None
    raw = _raw()
    raw["generation"] = {"count": 2, "label_quotas": {"billing": 2, "shipping": 0}}
    config = validate_config(raw)
    assert quota_answers(config) == [({"choice": "billing"}, 2), ({"choice": "shipping"}, 0)]
    assert validate_config(config) == config
    raw["generation"]["label_quotas"] = None
    assert quota_answers(validate_config(raw)) is None


def test_balanced_quotas_resolve_in_canonical_order_for_all_question_types():
    choice = _raw()
    choice["generation"] = {"count": 5, "label_quotas": "balanced"}
    assert quota_answers(validate_config(choice)) == [
        ({"choice": "billing"}, 3), ({"choice": "shipping"}, 2)]

    score = _raw({"type": "score", "instructions": "Rate risk.",
                  "criteria": ["low", "medium", "high"]})
    score["generation"] = {"count": 5, "label_quotas": "balanced"}
    assert quota_answers(validate_config(score)) == [
        ({"level": 0}, 2), ({"level": 1}, 2), ({"level": 2}, 1)]

    noul = _raw({"type": "noul", "instructions": "Ask for a person?"})
    noul["generation"] = {"count": 3, "label_quotas": "balanced"}
    assert quota_answers(validate_config(noul)) == [({"noul": True}, 2), ({"noul": False}, 1)]


def test_explicit_score_and_noul_quotas_use_string_keys_and_hard_answers():
    score = _raw({"type": "score", "instructions": "Rate risk.",
                  "criteria": ["low", "medium", "high"]})
    score["generation"] = {"count": 3, "label_quotas": {"2": 2, "0": 1, "1": 0}}
    assert quota_answers(validate_config(score)) == [
        ({"level": 0}, 1), ({"level": 1}, 0), ({"level": 2}, 2)]

    noul = _raw({"type": "noul", "instructions": "Ask for a person?"})
    noul["generation"] = {"count": 3, "label_quotas": {"false": 3, "true": 0}}
    assert quota_answers(validate_config(noul)) == [({"noul": True}, 0), ({"noul": False}, 3)]


@pytest.mark.parametrize("quotas", [
    {"billing": 2},
    {"billing": 1, "shipping": 1, "unknown": 0},
    {"billing": True, "shipping": 1},
    {"billing": 1.0, "shipping": 1},
    {"billing": -1, "shipping": 3},
    {"billing": 1, "shipping": 0},
    "uneven",
])
def test_explicit_quotas_reject_incomplete_extra_noninteger_or_wrong_sum(quotas):
    raw = _raw()
    raw["generation"] = {"count": 2, "label_quotas": quotas}
    with pytest.raises(ValidationError):
        validate_config(raw)


def test_balanced_requires_at_least_one_slot_per_label():
    raw = _raw()
    raw["generation"] = {"count": 1, "label_quotas": "balanced"}
    with pytest.raises(ValidationError, match="every label"):
        validate_config(raw)


def test_score_and_noul_quota_keys_must_be_quoted_yaml_strings(tmp_path):
    import yaml

    for question, quotas in (
        ({"type": "score", "instructions": "Rate risk.", "criteria": ["low", "high"]},
         {0: 1, "1": 1}),
        ({"type": "noul", "instructions": "Ask for a person?"},
         {True: 1, False: 1}),
    ):
        raw = _raw(question)
        raw["generation"] = {"count": 2, "label_quotas": quotas}
        path = tmp_path / "quotas.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        with pytest.raises(ValidationError, match="keys must be strings"):
            load_config(path)
