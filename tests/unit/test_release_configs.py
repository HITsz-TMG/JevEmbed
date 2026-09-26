"""Public merged checkpoints use the established Jev prompt and scoring path."""

from copy import deepcopy
from dataclasses import replace

import pytest

from conftest import FixedBackend, ROOT
from jevembed import JevEmbed, ModelConfig


@pytest.mark.parametrize("release,base,repo", [
    ("jevembed-qwen3-embedding-0.6b", "qwen3-embedding-0.6b", "JevEmbed-Qwen3-Embedding-0.6B"),
    ("jevembed-kalm-embedding-v2.5", "kalm-embedding-v2.5", "JevEmbed-KaLM-Embedding-V2.5"),
])
@pytest.mark.parametrize("fixture_name", ["choice_exchange", "score_safari", "noul_escalation"])
def test_merged_release_config_uses_existing_compiler_and_scoring(
    fixtures, release, base, repo, fixture_name
):
    config = ModelConfig.load(ROOT / f"configs/{release}.yaml")
    baseline = ModelConfig.load(ROOT / f"configs/{base}.yaml")
    hub_id = f"HIT-TMG/{repo}"
    assert config.model_id == release
    assert config.model_name_or_path == hub_id
    assert config.aliases == (hub_id, repo)
    assert config.local_files_only is False
    assert config.expected_dimension == baseline.expected_dimension
    assert config.trust_remote_code == baseline.trust_remote_code
    assert config.max_input_tokens == 1024 and config.overflow_policy == "truncate"
    assert config.pooling == "model_default"
    assert config.prompts == baseline.prompts
    assert config.scoring == baseline.scoring

    request = deepcopy(fixtures[fixture_name])
    released = JevEmbed(config=replace(config, expected_dimension=4), backend=FixedBackend())
    original = JevEmbed(config=replace(baseline, expected_dimension=4), backend=FixedBackend())
    for name in (hub_id, repo):
        request["model"] = name
        actual = released.explain(request)
        assert actual["model"] == release
        request["model"] = baseline.model_id
        assert actual["tasks"] == original.explain(request)["tasks"]
        request["model"] = name
        assert released.evaluate(request)["answers"] == original.evaluate(
            {**request, "model": baseline.model_id})["answers"]
