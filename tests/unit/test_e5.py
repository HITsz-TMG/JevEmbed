"""E5's instruct variant uses the common mapping, without query:/passage: prefixes."""
from copy import deepcopy
from dataclasses import replace

import pytest

from conftest import FixedBackend, ROOT
from jevembed import JevEmbed, ModelConfig


@pytest.mark.parametrize("fixture_name", ["choice_exchange", "score_safari", "noul_escalation"])
def test_e5_compilation_matches_existing_four_paths(fixtures, fixture_name):
    cfg = ModelConfig.load(ROOT / "configs/multilingual-e5-large-instruct.yaml")
    assert cfg.expected_dimension == 1024
    assert cfg.trust_remote_code is False and cfg.overflow_policy == "error"
    client = JevEmbed(config=cfg)
    request = deepcopy(fixtures[fixture_name])
    request["model"] = cfg.model_id
    actual = client.explain(request)
    baseline_cfg = ModelConfig.load(ROOT / "configs/kalm-embedding-v2.5.yaml")
    baseline = JevEmbed(config=baseline_cfg).explain({**request, "model": baseline_cfg.model_id})
    assert actual["tasks"] == baseline["tasks"]
    for task in actual["tasks"]:
        for item in task["inputs"]:
            if item["role"] == "query":
                assert item["rendered"] == f"Instruct: {item['instruction']}\nQuery: {item['text']}"
            else:
                assert item["rendered"] == item["text"]


def test_e5_alias_switch_and_cache_isolation(fixtures):
    client = JevEmbed()
    for name in ("qwen3-embedding-0.6b", "multilingual-e5-large-instruct"):
        # This fixture backend emits four-dimensional vectors; real reports check 1024.
        cfg = replace(ModelConfig.load(ROOT / f"configs/{name}.yaml"), expected_dimension=4)
        client.register(cfg, backend=FixedBackend())
    request = deepcopy(fixtures["noul_escalation"])
    client.evaluate({**request, "model": "qwen3-embedding-0.6b"})
    first = client.evaluate({**request, "model": "intfloat/multilingual-e5-large-instruct"})
    cached = client.evaluate({**request, "model": "multilingual-e5-large-instruct"})
    assert first["model"] == cached["model"] == "multilingual-e5-large-instruct"
    assert first["usage"]["input_tokens"] > 0 and cached["usage"]["input_tokens"] == 0
    assert first["answers"] == cached["answers"]
