"""Optional parity checks using the pinned upstream renderer and real tokenizer.

Set JEVEMBED_KANANA_REFERENCE to a downloaded snapshot directory. Only the
renderer, tokenizer files, and chat template are needed; no model weights.
"""

from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

from jevembed import ModelConfig, ValidationError
from jevembed.backends.kanana_decide import KananaDecideBackend, compile_kanana_request
from jevembed.scoring import score_plan


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def reference_runtime():
    snapshot = os.environ.get("JEVEMBED_KANANA_REFERENCE")
    if not snapshot:
        pytest.skip("Set JEVEMBED_KANANA_REFERENCE to the pinned Kanana snapshot")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    root = Path(snapshot)
    template = (root / "chat_template.jinja").read_text(encoding="utf-8")
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(root / "tokenizer.json"), bos_token="<|begin_of_text|>",
        eos_token="<|im_end|>", pad_token="<|end_of_text|>", chat_template=template)

    class Processor:
        def __init__(self):
            self.tokenizer = tokenizer
        def apply_chat_template(self, messages, **kwargs):
            return tokenizer.apply_chat_template(messages, chat_template=template, **kwargs)

    spec = importlib.util.spec_from_file_location("kanana_reference", root / "decision_kanana_v_decide.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    return reference, Processor()


def make_backend(runtime, **changes):
    config = replace(ModelConfig.load(ROOT / "configs/kanana-2-v-decide-3b-2611.yaml"), **changes)
    backend = KananaDecideBackend(config=config)
    backend._reference, backend._processor = runtime
    backend._limit = config.max_input_tokens
    return backend


def request_for(kind):
    question = {"type": kind, "instructions": {"z": "원래 순서", "a": "Choose using the state"}}
    if kind == "choice":
        question["criteria"] = {"null": None, "empty": "", "structured": {"z": 1, "a": ["예", False]}}
    elif kind == "score":
        question["criteria"] = ["low", {"z": [1, False], "a": "high"}]
    elif kind == "noul_criteria":
        question.update(type="noul", criteria={"true": {"z": 1, "a": 2}, "false": ""})
    return {"state": {"z": ["한글", False, None], "a": {"b": "English", "a": 1}}, "questions": {"q": question}}


@pytest.mark.parametrize("kind", ["choice", "score", "noul", "noul_criteria"])
def test_query_and_candidates_match_native_token_inputs(reference_runtime, kind):
    backend = make_backend(reference_runtime)
    ref, processor = reference_runtime
    request = request_for(kind)
    plan = compile_kanana_request(request, backend.config.prompts)[0]
    payload, labels, candidates = ref.build_pairs(request["state"], request["questions"])["q"]
    expected_query = ref.query_messages(payload, processor.tokenizer, processor, backend._limit, truncate=True,
                                       truncate_question_candidates=True)
    expected_docs = dict(zip(labels, ref.validate_candidates(candidates, processor.tokenizer, processor,
                                                             backend._limit, truncate=True)))
    actual_query, _ = backend._messages(plan.inputs[0])
    assert ref.message_ids(actual_query, processor.tokenizer, processor, backend._limit) == ref.message_ids(
        expected_query, processor.tokenizer, processor, backend._limit)
    for label, item in zip(plan.labels, plan.inputs[1:]):
        actual, _ = backend._messages(item)
        assert tuple(ref.message_ids(actual, processor.tokenizer, processor, backend._limit)) == expected_docs[label][1]


@pytest.mark.parametrize("kind", ["choice", "score", "noul", "noul_criteria"])
def test_framework_scoring_matches_native_logits(reference_runtime, kind):
    backend = make_backend(reference_runtime)
    ref, _ = reference_runtime
    request = request_for(kind)
    plan = compile_kanana_request(request, backend.config.prompts)[0]
    _, labels, _ = ref.build_pairs(request["state"], request["questions"])["q"]
    similarities = {label: 0.23 + 0.17 * i for i, label in enumerate(labels)}
    vectors = [[1.0, 0.0]] + [[similarities[label], math.sqrt(1 - similarities[label] ** 2)] for label in plan.labels]
    actual, _ = score_plan(plan, vectors, backend.config.scoring)
    expected = ref.answer_from_logits(request["questions"]["q"], labels,
                                    [similarities[label] / backend.config.scoring.choice_temperature for label in labels])
    assert actual["type"] == expected["type"]
    if kind.startswith("noul"):
        assert actual["noul"] == pytest.approx(expected["noul"], abs=1e-12)
    else:
        assert actual["probabilities"] == pytest.approx(expected["probabilities"], abs=1e-12)
        assert actual["confidence"] == pytest.approx(expected["confidence"], abs=1e-12)
        if kind == "choice":
            assert actual["choice"] == expected["choice"]
        else:
            assert actual["score"] == pytest.approx(expected["score"], abs=1e-12)


def test_native_head_tail_truncation_and_strict_mode(reference_runtime):
    backend = make_backend(reference_runtime, max_input_tokens=140)
    request = {"state": "start " + "내용 long text " * 400 + " end",
               "questions": {"q": {"type": "choice", "instructions": "Choose", "criteria": {"a": "one", "b": "two"}}}}
    plan = compile_kanana_request(request, backend.config.prompts)[0]
    ref, processor = reference_runtime
    payload, _, _ = ref.build_pairs(request["state"], request["questions"])["q"]
    expected = ref.query_messages(payload, processor.tokenizer, processor, 140, truncate=True,
                                 truncate_question_candidates=True)
    actual, record = backend._messages(plan.inputs[0])
    assert actual == expected
    assert record["original_tokens"] > 140
    assert record["encoded_tokens"] <= 140
    strict = make_backend(reference_runtime, max_input_tokens=140, overflow_policy="error")
    with pytest.raises(ValidationError, match="truncation is disabled"):
        strict._messages(plan.inputs[0])


def test_native_large_candidate_set_hides_options_and_keeps_all_candidates(reference_runtime):
    backend = make_backend(reference_runtime)
    request = {"state": "route", "questions": {"q": {"type": "choice", "instructions": "pick",
                "criteria": {str(i): f"option {i}" for i in range(256)}}}}
    plan = compile_kanana_request(request, backend.config.prompts)[0]
    messages, _ = backend._messages(plan.inputs[0])
    assert len(plan.inputs) == 257
    assert "Options:" not in messages[1]["content"][0]["text"]
