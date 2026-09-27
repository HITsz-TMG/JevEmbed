import json
import os
import subprocess
import sys

import pytest

from conftest import ROOT
from jevembed.cli import main


def test_cli_explain_and_invalid_request(tmp_path):
    output = tmp_path / "explain.json"
    assert main(["--config", str(ROOT / "configs/kalm-embedding-v2.5.yaml"), "--input",
                 str(ROOT / "examples/official_noul_escalation.json"), "--explain", "--output", str(output)]) == 0
    assert len(json.loads(output.read_text())["tasks"]) == 2
    assert main(["--config", str(ROOT / "configs/kalm-embedding-v2.5.yaml"), "--input",
                 str(ROOT / "tests/fixtures/official_noul_escalation.request.json"), "--explain"]) == 2


def test_core_import_is_lightweight():
    result = subprocess.run([sys.executable, "-c", "import sys; import jevembed; assert 'torch' not in sys.modules; assert 'sentence_transformers' not in sys.modules; assert 'httpx' not in sys.modules"],
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_passes_http_limits_to_app(monkeypatch):
    import uvicorn
    from jevembed import server

    observed = []
    monkeypatch.setattr(server, "create_app", lambda client, *, limits: observed.append(limits))
    monkeypatch.setattr(uvicorn, "run", lambda app, *, host, port: None)
    assert main(["--config", str(ROOT / "configs/kalm-embedding-v2.5.yaml"), "--serve",
                 "--max-request-bytes", "4096", "--max-questions", "2",
                 "--max-embedding-inputs", "8192", "--max-concurrent-requests", "3",
                 "--body-read-timeout-seconds", "2.5"]) == 0
    limits = observed[0]
    assert (limits.max_body_bytes, limits.max_questions, limits.max_embedding_inputs,
            limits.max_concurrent_requests) == (4096, 2, 8192, 3)
    assert limits.body_read_timeout_seconds == 2.5


@pytest.mark.parametrize("config_names", [[], ["kalm-embedding-v2.5"],
                                        ["qwen3-embedding-0.6b", "jevembed-qwen3-embedding-0.6b"]])
def test_playground_flag_starts_server_without_reading_stdin(monkeypatch, tmp_path, config_names):
    import uvicorn
    from jevembed import server

    class NoStdin:
        def read(self):
            raise AssertionError("Playground must not read a CLI request")

    observed = {}
    app = object()

    def fake_create_app(client, *, limits, enable_playground=False):
        observed["models"] = [model["id"] for model in client.models()]
        observed["limits"] = limits
        observed["enable_playground"] = enable_playground
        return app

    def fake_run(received_app, *, host, port):
        observed["run"] = (received_app, host, port)

    monkeypatch.setattr(sys, "stdin", NoStdin())
    monkeypatch.setattr(server, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_run)

    monkeypatch.chdir(tmp_path)
    arguments = [argument for name in config_names
                 for argument in ("--config", str(ROOT / "configs" / f"{name}.yaml"))]
    assert main([*arguments, "--playground"]) == 0
    assert observed["enable_playground"] is True
    assert observed["models"] == (config_names or ["qwen3-embedding-0.6b", "jevembed-qwen3-embedding-0.6b"])
    assert observed["run"] == (app, "127.0.0.1", 8000)


def test_playground_defaults_match_shipped_hugging_face_configs():
    from jevembed.cli import _default_playground_models
    from jevembed.config import ModelConfig

    for model in _default_playground_models():
        assert model == ModelConfig.load(ROOT / "configs" / f"{model.model_id}.yaml")
        assert not model.local_files_only


def test_config_remains_required_outside_playground():
    with pytest.raises(SystemExit) as exc:
        main(["--serve"])
    assert exc.value.code == 2
