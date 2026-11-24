from types import SimpleNamespace
import os
from pathlib import Path

import pytest

from conftest import ROOT
from jevembed import JevEmbed, ModelConfig
from jevembed.backends.clm_qwen35 import _chat_message, _option_lines, compile_clm_request, render_query
from jevembed.backends.jeva_decide import JevaDecisionBackend
from jevembed.backends.jeva_decide import compile_jeva_request


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text)}

    def decode(self, tokens):
        return "".join(tokens)

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"


def test_clm_compiler_preserves_candidate_rows_and_question_layout():
    config = ModelConfig.load(ROOT / "configs/clm-qwen3.5-2b.yaml")
    assert config.model_name_or_path == "Qwen/Qwen3.5-2B"
    assert config.projection_name_or_path == "jayavibhav/clm-qwen3.5-2b"
    request = {
        "model": config.model_id,
        "state": {"ticket": "duplicate charge"},
        "questions": {"team": {"type": "choice", "instructions": "Which team?",
                                 "criteria": {"billing": "refunds", "tech": "outage"}}},
    }
    plan, = compile_clm_request(request, config.prompts)
    assert plan.labels == ["billing", "tech"]
    assert [item.text for item in plan.inputs[1:]] == ["billing: refunds", "tech: outage"]
    query = __import__("json").loads(plan.inputs[0].text)
    assert "Which team?" in _chat_message("ticket: duplicate charge", query["question"])
    assert "Answer with the name of one option." in _chat_message("", query["question"])
    assert _option_lines({"type": "noul", "instructions": "true?"})[2] == ["false", "true"]


def test_clm_overflow_shortens_context_and_retains_question_suffix():
    tokenizer = FakeTokenizer()
    payload = {"state": "S" * 120, "question": {"type": "choice", "instructions": "choose",
                                                   "criteria": {"x": "one", "y": "two"}}}
    rendered, record = render_query(tokenizer, payload, max_tokens=140, overflow_policy="truncate")
    assert record["original_tokens"] > 139
    assert "Question: choose" in rendered
    assert "x: one" in rendered and "y: two" in rendered
    assert "S" in rendered


class FakeQuestion:
    def __init__(self, instructions, criteria=None, allow_null=False):
        self.instructions, self.criteria, self.allow_null = instructions, criteria, allow_null


class FakeNoul:
    def __init__(self, instructions):
        self.instructions = instructions


class FakeScore(FakeQuestion):
    pass


class FakeJeva:
    def predict(self, *, state, question, decision_type, options, allow_null, positive_index):
        values = [0.3, 0.7] if len(options) == 2 else [0.2, 0.7, 0.1]
        return SimpleNamespace(candidate_probabilities=tuple(values), usage=SimpleNamespace(input_tokens=23,
                                                                                           truncated_tokens=0))

    def clear_cache(self):
        self.cleared = True


def test_jeva_compiler_reuses_shared_task_plans_and_keeps_native_candidate_order():
    config = ModelConfig.load(ROOT / "configs/jeva-4b.yaml")
    request = {"state": {"b": 2, "a": 1}, "questions": {
        "choice": {"type": "choice", "instructions": "pick", "criteria": {"x": "", "y": None}},
        "score": {"type": "score", "instructions": "rate", "criteria": ["low", "high"]},
        "noul": {"type": "noul", "instructions": "binary?", "criteria": {"true": "", "false": None}},
    }}

    choice, score, noul = compile_jeva_request(request, config.prompts)
    assert choice.inputs[0].text == '{"a":1,"b":2}'
    assert [item.text for item in choice.inputs[1:]] == ["x", "y"]
    assert [item.text for item in score.inputs[1:]] == ["low", "high"]
    assert noul.labels == ["false", "true"]
    assert [item.text for item in noul.inputs[1:]] == ["null", "true"]


def test_jeva_native_head_maps_all_typed_answers_without_vector_scoring():
    config = ModelConfig.load(ROOT / "configs/jeva-4b.yaml")
    backend = JevaDecisionBackend(config=config, model=FakeJeva())
    client = JevEmbed(config=config, backend=backend)
    request = {
        "model": config.model_id, "state": "A service is down",
        "questions": {
            "choice": {"type": "choice", "instructions": "Which team?", "criteria": {"ops": "operations", "eng": "engineering"}},
            "noul": {"type": "noul", "instructions": "Is it down?"},
            "score": {"type": "score", "instructions": "Severity?", "criteria": ["low", "medium", "high"]},
        },
    }
    trace = client.evaluate_with_trace(request)
    response = trace["response"]
    assert response["usage"]["input_tokens"] == 69
    assert response["answers"]["choice"]["choice"] == "eng"
    assert response["answers"]["noul"]["noul"] == 0.7
    assert abs(response["answers"]["score"]["score"] - 0.9) < 1e-12
    assert trace["trace"]["prompt_diagnostics"]["vector_similarity_scoring"] is False
    client.clear_cache(config.model_id)
    assert backend._model.cleared


def test_pinned_clm_head_matches_existing_paired_projection_architecture():
    torch = pytest.importorskip("torch")
    from jevembed.backends.paired_projection import _make_head

    head_path = Path(os.environ.get("JEVEMBED_CLM_QWEN35_HEAD", ROOT / "artifacts/clm-qwen35-reference/clm_heads.pt"))
    if not head_path.is_file():
        pytest.skip("Set JEVEMBED_CLM_QWEN35_HEAD to the pinned optional projection checkpoint")
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
    cfg = checkpoint["cfg"]
    assert (cfg["hidden_size"], cfg["width"], cfg["depth"], cfg["projection_dim"]) == (2048, 1536, 3, 512)
    assert cfg["activation"] == "gelu" and cfg["layernorm"] and not cfg["residual"]
    head = _make_head(torch.nn, hidden=2048, width=1536, depth=3, projection=512,
                      activation="gelu", layernorm=True, residual=False)
    head.load_state_dict(checkpoint["state_head"], strict=True)
    head.load_state_dict(checkpoint["action_head"], strict=True)
    assert head(torch.zeros(1, 2048)).shape == (1, 512)
