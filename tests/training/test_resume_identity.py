"""Resume must bind the optimizer checkpoint to the exact base artifacts."""

import json
import shutil
import sys
from types import SimpleNamespace

import pytest

from jevembed.errors import ValidationError
from jevembed.training.identity import (
    base_artifact_identity, portable_base_reference, validate_resume_manifest)


def make_base(root):
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"weights-A")
    (root / "config.json").write_text('{"hidden_size": 8}')
    (root / "tokenizer.json").write_text('{"vocab": {"a": 1}}')
    (root / "README.md").write_text("Unrelated documentation")
    return root


def test_base_fingerprint_uses_bytes_and_is_portable(tmp_path):
    first = make_base(tmp_path / "base-one")
    moved = tmp_path / "somewhere-else" / "base-two"
    shutil.copytree(first, moved)
    expected = base_artifact_identity(first, None)
    assert base_artifact_identity(moved, None) == expected

    (moved / "README.md").write_text("Changed documentation")
    (moved / "example.py").write_text("print('unrelated')")
    (moved / "pytorch_model.bin").write_bytes(b"unused alternate format")
    (moved / "model.safetensors.index.json").write_text(
        '{"weight_map": {"layer": "model-00001-of-00001.safetensors"}}')
    (moved / "model-00001-of-00001.safetensors").write_bytes(b"unused shard")
    (moved / "model.safetensors").touch()
    assert base_artifact_identity(moved, None) == expected

    # The replacement has the same size; metadata-only checks would miss it.
    (moved / "model.safetensors").write_bytes(b"weights-B")
    assert base_artifact_identity(moved, None) != expected
    (moved / "model.safetensors").write_bytes(b"weights-A")
    (moved / "tokenizer.json").write_text('{"vocab": {"b": 1}}')
    assert base_artifact_identity(moved, None) != expected


def test_trusted_model_code_is_part_of_base_identity(tmp_path):
    base = make_base(tmp_path / "base")
    code = base / "modeling.py"
    code.write_text("class Model: pass\n")
    expected = base_artifact_identity(base, None, trust_remote_code=True)
    code.write_text("class Model: ...\n")
    assert base_artifact_identity(base, None, trust_remote_code=True) != expected


def test_sharded_index_selects_only_referenced_weights(tmp_path):
    base = make_base(tmp_path / "base")
    (base / "model.safetensors").unlink()
    shard = base / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"used shard")
    unused = base / "model-00002-of-00002.safetensors"
    unused.write_bytes(b"unused shard")
    (base / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"layer": shard.name}}))
    expected = base_artifact_identity(base, None)
    unused.write_bytes(b"different unused shard")
    assert base_artifact_identity(base, None) == expected
    shard.write_bytes(b"changed used shard")
    assert base_artifact_identity(base, None) != expected


def test_remote_base_uses_loaded_commit_in_cached_snapshot(tmp_path, monkeypatch):
    snapshot = make_base(tmp_path / "snapshot")
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda **kwargs: (calls.append(kwargs), str(snapshot))[1]))
    assert base_artifact_identity("owner/model", "a" * 40) == base_artifact_identity(snapshot, None)
    assert calls == [{"repo_id": "owner/model", "revision": "a" * 40,
                      "local_files_only": True}]
    with pytest.raises(ValidationError, match="revision is unknown"):
        base_artifact_identity("owner/model", None)


def test_manifest_reference_does_not_record_local_paths_or_optional_cache_commit(tmp_path):
    base = make_base(tmp_path / "base")
    assert portable_base_reference(str(base), str(base), "a" * 40) == ("local_snapshot", None)
    assert portable_base_reference("owner/model", str(base), "a" * 40) == ("owner/model", None)
    assert portable_base_reference("owner/model", "owner/model", "a" * 40) == (
        "owner/model", "a" * 40)


def test_resume_rejects_changed_base_and_legacy_manifest(tmp_path):
    base = make_base(tmp_path / "base")
    output = tmp_path / "run"
    checkpoint = output / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    manifest_path = output / "training_manifest.json"
    expected = {"base_artifacts": base_artifact_identity(base, None), "training": {"seed": 42}}
    manifest_path.write_text(json.dumps(expected))
    validate_resume_manifest(manifest_path, checkpoint, output, expected)

    (base / "model.safetensors").write_bytes(b"weights-B")
    changed = {**expected, "base_artifacts": base_artifact_identity(base, None)}
    with pytest.raises(ValidationError, match="base artifacts differ"):
        validate_resume_manifest(manifest_path, checkpoint, output, changed)

    manifest_path.write_text(json.dumps({"training": {"seed": 42}}))
    with pytest.raises(ValidationError, match="legacy run"):
        validate_resume_manifest(manifest_path, checkpoint, output, expected)
    assert validate_resume_manifest(manifest_path, checkpoint, output, expected,
                                    allow_unverified_base=True) is False
    with pytest.raises(ValidationError, match="saved legacy manifest"):
        validate_resume_manifest(manifest_path, checkpoint, output,
                                 {**expected, "training": {"seed": 43}},
                                 allow_unverified_base=True)

    manifest_path.write_text(json.dumps(expected))
    with pytest.raises(ValidationError, match="base artifacts differ"):
        validate_resume_manifest(manifest_path, checkpoint, output, changed,
                                 allow_unverified_base=True)


def test_resume_rejects_foreign_checkpoint(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    manifest_path = output / "training_manifest.json"
    expected = {"base_artifacts": {"algorithm": "sha256-artifacts-v1", "sha256": "abc", "file_count": 3}}
    manifest_path.write_text(json.dumps(expected))
    foreign = tmp_path / "other-run" / "checkpoint-10"
    foreign.mkdir(parents=True)
    with pytest.raises(ValidationError, match="directly inside"):
        validate_resume_manifest(manifest_path, foreign, output, expected)
