import json
from dataclasses import replace

import pytest

from jevembed import JevEmbed, ModelConfig, ValidationError
from jevembed.training.data import check_disjoint, load_examples
from jevembed.training.run import TrainingConfig


def record(kind="choice", answer=None):
    criteria = {"a": "alpha", "b": "beta"} if kind == "choice" else ["low", "high"]
    return {"id": "one", "request": {"state": "state", "questions": {"q": {
        "type": kind, "instructions": "instruction", "criteria": criteria}}},
        "answers": {"q": answer or {"choice": "a"}}}


def read(tmp_path, records):
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return load_examples(p, ModelConfig(model_id="test"))


def test_compilation_matches_inference(tmp_path, client):
    row = record()
    example = read(tmp_path, [row])[0]
    trace = client.explain(row["request"])
    assert example.texts == [i["rendered"] for i in trace["tasks"][0]["inputs"]]
    assert example.target == [1, 0]
    assert "choice" not in example.texts


@pytest.mark.parametrize("answer", [{"choice": "missing"}, {"choice": True}, {"choice": "a", "confidence": 1},
                                    {"probabilities": {"a": 0.5}}, {"probabilities": {"a": 1, "b": 1}},
                                    {"probabilities": {"a": float("nan"), "b": 0}}])
def test_invalid_choice_targets(tmp_path, answer):
    with pytest.raises(ValidationError):
        read(tmp_path, [record(answer=answer)])


@pytest.mark.parametrize("answer,mode,target", [({"level": 1},0,[0,1]), ({"score": 0.7},1,[0.7]),
                                              ({"probabilities": {"0": .2,"1": .8}},0,[.2,.8])])
def test_score_supervision(tmp_path, answer, mode, target):
    e = read(tmp_path, [record("score",answer)])[0]
    assert (e.target_mode,e.target) == (mode,target)


@pytest.mark.parametrize("answer", [{"level": True}, {"level": 1.1}, {"score": 2}, {"score": -1}])
def test_invalid_score(tmp_path, answer):
    with pytest.raises(ValidationError):
        read(tmp_path, [record("score",answer)])


def test_noul_and_split_leakage(tmp_path):
    row = record()
    row["request"]["questions"]["q"] = {"type":"noul", "instructions":"meaning"}
    row["answers"]["q"] = {"noul": False}
    e = read(tmp_path,[row])[0]
    assert e.target == [0] and e.plan.path == "noul_without_criteria"
    assert all(t.startswith("Instruct: Retrieve semantically similar text.") for t in e.texts)
    with pytest.raises(ValidationError,match="state/question"):
        check_disjoint([e],[replace(e,record_id="different")])
    with pytest.raises(ValidationError,match="groups"):
        check_disjoint([replace(e,group="g")],[replace(e,record_id="other",fingerprint="other",group="g")])


def test_noul_with_criteria_training_matches_inference(tmp_path):
    row = record()
    row["request"]["questions"]["q"] = {"type": "noul", "instructions": "meaning",
                                         "criteria": {"true": "yes", "false": "no"}}
    row["answers"]["q"] = {"noul": True}
    path = tmp_path / "noul.jsonl"
    path.write_text(json.dumps(row))
    config = ModelConfig(model_id="test")
    example = load_examples(path, config)[0]
    trace = JevEmbed(config=config, backend=object()).explain(row["request"])
    assert example.plan.path == "noul_with_criteria"
    assert example.target == [1]
    assert example.texts == [item["rendered"] for item in trace["tasks"][0]["inputs"]]
    assert len(example.texts) == 3


def test_duplicates_and_missing_answers(tmp_path):
    with pytest.raises(ValidationError):
        read(tmp_path,[record(),record()])
    row=record();row["answers"]={}
    with pytest.raises(ValidationError):
        read(tmp_path,[row])


def test_repeated_inputs_keep_the_official_training_weight(tmp_path):
    a, b = record(), record()
    b['id'] = 'two'
    assert len(read(tmp_path,[a,b])) == 2
    b['answers']['q'] = {'choice':'b'}
    with pytest.raises(ValidationError,match='Conflicting'):
        read(tmp_path,[a,b])


@pytest.mark.parametrize("options", [{"batch_size":0}, {"lora_rank":True}, {"learning_rate":float("nan")},
                                   {"max_steps":0}, {"target_modules":[]}, {"overflow_policy":"silent"}])
def test_training_configuration_validation(options):
    with pytest.raises(ValidationError):
        TrainingConfig(**options)
