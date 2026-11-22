import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest
import yaml

from conftest import ROOT


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jeva_local_config_distinguishes_external_and_complete_local_backbones(tmp_path, monkeypatch):
    helper = load_script("prepare_jev_models")
    weights, output = tmp_path / "weights", tmp_path / "configs"
    model = weights / "jeva-4b"
    adapter = model / "backbone_adapter"
    adapter.mkdir(parents=True)
    (model / "head.pt").write_bytes(b"fixture")
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    config = model / "config.yaml"
    config.write_text("model:\n  backbone: Qwen/Qwen3.5-4B\n")
    monkeypatch.setattr(sys, "argv", ["prepare", "local-configs", "--model", "jeva-4b",
                                     "--weights-dir", str(weights), "--output-dir", str(output)])
    helper.main()
    local = output / "jeva-4b.local.yaml"
    assert yaml.safe_load(local.read_text())["local_files_only"] is False
    backbone = model / "backbone"
    backbone.mkdir()
    config.write_text("model:\n  backbone: backbone\n")
    (backbone / "config.json").write_text("{}")
    (backbone / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer":"shard.safetensors"}}')
    with pytest.raises(FileNotFoundError):
        helper.main()
    (backbone / "shard.safetensors").write_bytes(b"fixture")
    helper.main()
    assert yaml.safe_load(local.read_text())["local_files_only"] is True


def test_deployment_examples_compile_in_subprocesses_without_weights(tmp_path):
    smoke = load_script("smoke_a100_models")
    argv = ["--explain", "--output-dir", str(tmp_path)]
    for name in ("decision", "gemma"):
        argv.extend([f"--{name}-python", sys.executable])
    assert smoke.main(argv) == 0
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["mode"] == "compile_only" and result["ok"]
    assert len(result["models"]) == 4
    assert all(row["status"] == "compiled" for row in result["models"])


def test_smoke_failure_cannot_reuse_an_old_success_output(tmp_path, monkeypatch):
    smoke = load_script("smoke_a100_models")
    model_id = "clm-qwen3.5-2b"
    stale = tmp_path / f"{model_id}.json"
    stale.write_text('{"model":"stale"}')
    monkeypatch.setattr(smoke.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=2, stdout="", stderr="Model load failed"))
    assert smoke.main(["--model", model_id, "--configs-dir", str(ROOT / "configs"),
                       "--output-dir", str(tmp_path), "--decision-python", sys.executable]) == 1
    result = json.loads((tmp_path / "summary.json").read_text())
    assert not result["ok"] and result["mode"] == "real_inference"
    assert not stale.exists()
    assert "Model load failed" in (tmp_path / f"{model_id}.log").read_text()
