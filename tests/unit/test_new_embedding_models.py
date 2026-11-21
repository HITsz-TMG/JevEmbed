"""New standard embedding configs compile offline without downloading weights."""

from copy import deepcopy

import pytest

from conftest import ROOT
from jevembed import JevEmbed, ModelConfig


@pytest.mark.parametrize(
    "config_name,model_id,repo,revision,dimension,limit,trust",
    [
        (
            "nemotron-embed-1b-v2",
            "nemotron-embed-1b-v2",
            "nvidia/llama-nemotron-embed-1b-v2",
            "113abe4acafa848e77ead9c0623205e511932348",
            2048,
            8192,
            True,
        ),
    ],
)
@pytest.mark.parametrize("fixture_name", ["choice_exchange", "score_safari", "noul_escalation"])
def test_new_embedding_configs_compile_all_decision_types_offline(
    fixtures, config_name, model_id, repo, revision, dimension, limit, trust, fixture_name
):
    config = ModelConfig.load(ROOT / f"configs/{config_name}.yaml")
    assert config.backend == "sentence_transformers"
    assert config.model_id == model_id
    assert config.model_name_or_path == repo
    assert config.aliases == (repo,)
    assert config.revision == revision
    assert config.expected_dimension == dimension
    assert config.max_input_tokens == limit
    assert config.overflow_policy == "truncate"
    assert config.trust_remote_code is trust
    assert config.local_files_only is False

    request = deepcopy(fixtures[fixture_name])
    request["model"] = model_id
    result = JevEmbed(config=config).explain(request)
    assert result["model"] == model_id
    paths = [task["path"] for task in result["tasks"]]
    if fixture_name == "choice_exchange":
        assert paths == ["choice"]
    elif fixture_name == "score_safari":
        assert paths == ["score"]
    else:
        assert paths == ["noul_without_criteria", "noul_with_criteria"]


def test_new_embedding_examples_select_their_registered_models():
    import json

    for name, model_id in (
        ("nemotron_decide.json", "nemotron-embed-1b-v2"),
    ):
        request = json.loads((ROOT / "examples" / name).read_text(encoding="utf-8-sig"))
        config = ModelConfig.load(ROOT / f"configs/{model_id}.yaml")
        assert request["model"] == model_id
        result = JevEmbed(config=config).explain(request)
        assert {task["path"] for task in result["tasks"]} == {
            "choice",
            "noul_without_criteria",
            "score",
        }
