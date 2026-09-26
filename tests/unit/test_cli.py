import json
import os
import subprocess
import sys

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
