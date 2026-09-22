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
