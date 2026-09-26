"""Prepared training data must match the strict JSONL loader without retaining rows in RAM."""

import json

import pytest

pytest.importorskip("datasets")

from datasets.table import MemoryMappedTable

from jevembed import ModelConfig, PromptConfig, ValidationError
from jevembed.training.data import load_examples
from jevembed.training.prepared import open_prepared, prepare_data


def _record(record_id, kind, criteria, answer, *, state=None):
    question = {"type": kind, "instructions": "Choose carefully"}
    if criteria is not None:
        question["criteria"] = criteria
    return {"id": record_id, "group": f"group-{record_id}",
            "request": {"state": state or f"state {record_id}", "questions": {"q": question}},
            "answers": {"q": answer}}


def _write(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return path


def _mixed_rows():
    rows = [
        _record("choice-hard", "choice", {"a": "alpha", "b": "beta"}, {"choice": "b"}),
        _record("choice-soft", "choice", {"b": "beta", "a": "alpha"},
                {"probabilities": {"a": 0.2, "b": 0.8}}),
        _record("score-level", "score", ["low", "high"], {"level": 1}),
        _record("score-soft", "score", ["low", "medium", "high"],
                {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}),
        _record("score-value", "score", ["low", "high"], {"score": 0.75}),
        _record("noul-with", "noul", {"true": "yes", "false": "no"}, {"noul": True}),
        _record("noul-without", "noul", None, {"noul": False}),
    ]
    duplicate = json.loads(json.dumps(rows[0]))
    duplicate["id"] = "choice-hard-duplicate"
    duplicate["group"] = "group-choice-hard-duplicate"
    rows.append(duplicate)
    return rows


def _assert_example_equal(actual, expected):
    assert actual.record_id == expected.record_id
    assert actual.question_id == expected.question_id
    assert actual.group == expected.group
    assert actual.fingerprint == expected.fingerprint
    assert actual.plan == expected.plan
    assert actual.texts == expected.texts
    assert actual.target_mode == expected.target_mode
    assert actual.target == expected.target


def test_prepared_rows_and_lazy_validation_match_jsonl_loader(tmp_path):
    config = ModelConfig(model_id="test")
    train_path = _write(tmp_path / "train.jsonl", _mixed_rows())
    validation_path = _write(tmp_path / "validation.jsonl", [
        _record("validation-choice", "choice", {"b": "beta", "a": "alpha"}, {"choice": "a"}),
        _record("validation-noul", "noul", None, {"noul": False}),
    ])
    train_expected = load_examples(train_path, config)
    validation_expected = load_examples(validation_path, config)

    cache_dir = tmp_path / "cache"
    metadata = prepare_data(train_path, validation_path, config, cache_dir)
    train, validation, validation_examples = open_prepared(cache_dir)

    assert metadata["train_questions"] == len(train_expected)
    assert metadata["validation_questions"] == len(validation_expected)
    assert len(train) == len(train_expected)
    assert len(validation) == len(validation_expected)
    assert isinstance(train._data, MemoryMappedTable)
    assert isinstance(validation._data, MemoryMappedTable)
    assert set(train.column_names) == {"texts", "path", "target_mode", "target"}
    for index, expected in enumerate(train_expected):
        assert train[index] == expected.training_row()
    for index, expected in enumerate(validation_expected):
        assert validation[index] == expected.training_row()
    assert len(validation_examples) == len(validation_expected)
    _assert_example_equal(validation_examples[0], validation_expected[0])
    for actual, expected in zip(validation_examples[:], validation_expected):
        _assert_example_equal(actual, expected)
    assert train_expected[0].fingerprint == train_expected[-1].fingerprint
    assert train[0] == train[-1]


def test_prepared_data_without_validation_split(tmp_path):
    config = ModelConfig(model_id="test")
    train_path = _write(tmp_path / "train.jsonl", [_mixed_rows()[0]])
    cache_dir = tmp_path / "cache"
    metadata = prepare_data(train_path, None, config, cache_dir)
    train, validation, validation_examples = open_prepared(cache_dir)
    assert metadata["train_questions"] == len(train) == 1
    assert metadata["validation_questions"] == 0
    assert validation is None
    assert len(validation_examples) == 0


def test_prepared_cache_rebuilds_for_input_and_prompt_changes(tmp_path):
    config = ModelConfig(model_id="test")
    row = _mixed_rows()[0]
    train_path = _write(tmp_path / "train.jsonl", [row])
    cache_dir = tmp_path / "cache"
    first = prepare_data(train_path, None, config, cache_dir)
    initial_texts = open_prepared(cache_dir)[0][0]["texts"]

    row["request"]["state"] = "a changed state"
    _write(train_path, [row])
    second = prepare_data(train_path, None, config, cache_dir)
    changed_texts = open_prepared(cache_dir)[0][0]["texts"]
    assert second["dataset_sha256"] != first["dataset_sha256"]
    assert changed_texts != initial_texts

    new_config = ModelConfig(model_id="test", prompts=PromptConfig(query_template="Q: {text}"))
    prepare_data(train_path, None, new_config, cache_dir)
    prompted_texts = open_prepared(cache_dir)[0][0]["texts"]
    assert prompted_texts != changed_texts
    assert prompted_texts[0].startswith("Q: ")


def test_same_count_stale_cache_is_rejected_by_rank_zero_identity(tmp_path):
    config = ModelConfig(model_id="test")
    first_row = _mixed_rows()[0]
    second_row = json.loads(json.dumps(first_row))
    second_row["request"]["state"] = "same count, different source"
    first_path = _write(tmp_path / "first.jsonl", [first_row])
    second_path = _write(tmp_path / "second.jsonl", [second_row])
    first_cache, second_cache = tmp_path / "first-cache", tmp_path / "second-cache"
    first = prepare_data(first_path, None, config, first_cache)
    second = prepare_data(second_path, None, config, second_cache)
    assert first["train_questions"] == second["train_questions"] == 1
    with pytest.raises(ValidationError, match="identity"):
        open_prepared(first_cache, expected_metadata=second)

    other_config = ModelConfig(model_id="test", prompts=PromptConfig(query_template="Q: {text}"))
    other_cache = tmp_path / "other-cache"
    other = prepare_data(first_path, None, other_config, other_cache)
    assert first["dataset_sha256"] == other["dataset_sha256"]
    assert first["cache_signature"] != other["cache_signature"]
    with pytest.raises(ValidationError, match="identity"):
        open_prepared(first_cache, expected_metadata=other)


def test_complete_cache_allows_fresh_retry_but_other_output_does_not(tmp_path):
    from jevembed.training.run import check_output_directory

    output = tmp_path / "run"
    check_output_directory(output, None)
    output.mkdir()
    check_output_directory(output, None)
    (output / ".data-cache").mkdir()
    with pytest.raises(ValidationError, match="not empty"):
        check_output_directory(output, None)

    train_path = _write(tmp_path / "train.jsonl", [_mixed_rows()[0]])
    prepare_data(train_path, None, ModelConfig(model_id="test"), output / ".data-cache")
    check_output_directory(output, None)
    (output / "trainer_state.json").write_text("{}")
    with pytest.raises(ValidationError, match="not empty"):
        check_output_directory(output, None)


def test_unchanged_cache_is_reused_without_reparsing(tmp_path, monkeypatch):
    from jevembed.training import prepared

    config = ModelConfig(model_id="test")
    train_path = _write(tmp_path / "train.jsonl", [_mixed_rows()[0]])
    cache_dir = tmp_path / "cache"
    first = prepare_data(train_path, None, config, cache_dir)

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("cache reuse reparsed JSONL")

    monkeypatch.setattr(prepared, "scan_splits", unexpected_scan)
    assert prepare_data(train_path, None, config, cache_dir) == first


def test_source_mutation_during_rebuild_keeps_old_cache(tmp_path, monkeypatch):
    from jevembed.training import prepared

    config = ModelConfig(model_id="test")
    row = _mixed_rows()[0]
    train_path = _write(tmp_path / "train.jsonl", [row])
    cache_dir = tmp_path / "cache"
    prepare_data(train_path, None, config, cache_dir)
    old_row = open_prepared(cache_dir)[0][0]

    row["request"]["state"] = "first edit"
    _write(train_path, [row])
    scan = prepared.scan_splits

    def edit_after_scan(*args, **kwargs):
        result = scan(*args, **kwargs)
        row["request"]["state"] = "second edit"
        _write(train_path, [row])
        return result

    monkeypatch.setattr(prepared, "scan_splits", edit_after_scan)
    with pytest.raises(ValidationError, match="changed"):
        prepare_data(train_path, None, config, cache_dir)
    assert open_prepared(cache_dir)[0][0] == old_row


def test_validation_slice_is_lazy(tmp_path):
    config = ModelConfig(model_id="test")
    train_path = _write(tmp_path / "train.jsonl", [_mixed_rows()[0]])
    validation_path = _write(tmp_path / "validation.jsonl", [
        _record("val-1", "choice", {"a": "alpha", "b": "beta"}, {"choice": "a"}),
        _record("val-2", "noul", None, {"noul": True}),
    ])
    cache_dir = tmp_path / "cache"
    prepare_data(train_path, validation_path, config, cache_dir)
    examples = open_prepared(cache_dir)[2]

    class CountingRows:
        def __init__(self, rows):
            self.rows = rows
            self.reads = 0

        def __getitem__(self, index):
            self.reads += 1
            return self.rows[index]

    details = CountingRows(examples._metadata)
    examples._metadata = details
    head = examples[:1]
    assert len(head) == 1
    assert details.reads == 0
    assert head[0].record_id == "val-1"
    assert details.reads == 1


def test_small_byte_budget_flushes_writer_without_changing_rows(tmp_path, monkeypatch):
    from jevembed.training import prepared

    config = ModelConfig(model_id="test")
    rows = [_record(f"case-{i}", "choice", {"a": "alpha", "b": "beta"},
                    {"choice": "a"}) for i in range(5)]
    train_path = _write(tmp_path / "train.jsonl", rows)
    expected = [example.training_row() for example in load_examples(train_path, config)]
    monkeypatch.setattr(prepared, "_ROWS_PER_BATCH", 1000)
    monkeypatch.setattr(prepared, "_BYTES_PER_BATCH", 100)
    flushes = []
    flush = prepared._BoundedWriter.flush

    def counted_flush(self):
        if self.rows:
            flushes.append(self.bytes)
        return flush(self)

    monkeypatch.setattr(prepared._BoundedWriter, "flush", counted_flush)
    cache_dir = tmp_path / "cache"
    prepare_data(train_path, None, config, cache_dir)
    train = open_prepared(cache_dir)[0]
    assert [train[index] for index in range(len(train))] == expected
    assert len(flushes) >= 2


def test_invalid_rebuild_preserves_complete_previous_cache(tmp_path):
    config = ModelConfig(model_id="test")
    valid = _mixed_rows()[0]
    train_path = _write(tmp_path / "train.jsonl", [valid])
    cache_dir = tmp_path / "cache"
    prepare_data(train_path, None, config, cache_dir)
    expected = open_prepared(cache_dir)[0][0]

    conflict = json.loads(json.dumps(valid))
    conflict["id"] = "conflict"
    conflict["answers"]["q"] = {"choice": "a"}
    _write(train_path, [valid, conflict])
    with pytest.raises(ValidationError, match="Conflicting"):
        prepare_data(train_path, None, config, cache_dir)
    assert open_prepared(cache_dir)[0][0] == expected
    assert len(open_prepared(cache_dir)[0]) == 1

    _write(train_path, [valid])
    validation_path = _write(tmp_path / "validation.jsonl", [
        {**valid, "id": "validation-copy", "group": "new-group"}])
    with pytest.raises(ValidationError, match="overlap"):
        prepare_data(train_path, validation_path, config, cache_dir)
    assert open_prepared(cache_dir)[0][0] == expected


def test_invalid_fresh_preparation_publishes_no_cache(tmp_path):
    config = ModelConfig(model_id="test")
    valid = _mixed_rows()[0]
    invalid = json.loads(json.dumps(valid))
    invalid["id"] = "bad"
    invalid["answers"] = {}
    train_path = _write(tmp_path / "train.jsonl", [valid, invalid])
    cache_dir = tmp_path / "cache"
    with pytest.raises(ValidationError, match="answers"):
        prepare_data(train_path, None, config, cache_dir)
    assert not (cache_dir / "manifest.json").exists()
    with pytest.raises((FileNotFoundError, ValidationError)):
        open_prepared(cache_dir)
