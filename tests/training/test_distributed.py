"""CPU smoke tests for the torchrun training entry point."""

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("datasets")
pytest.importorskip("sentence_transformers")

from sentence_transformers import SentenceTransformer, models
from transformers import BertConfig, BertModel, BertTokenizerFast


@pytest.mark.parametrize("world_size", [2, 4])
def test_torchrun_saves_one_complete_run(tmp_path, world_size):
    per_device_batch_size = 2
    gradient_accumulation_steps = 2
    train_rows = 17
    base = tmp_path / "base"
    base.mkdir()
    (base / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\nd\n")
    BertTokenizerFast(vocab_file=str(base / "vocab.txt")).save_pretrained(base)
    BertModel(BertConfig(vocab_size=9, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2)).save_pretrained(base)
    SentenceTransformer(modules=[models.Transformer(str(base)), models.Pooling(8)], device="cpu").save(str(base))

    model_config = tmp_path / "model.yaml"
    model_config.write_text(yaml.safe_dump({"model_id": "tiny", "backend": "sentence_transformers",
                                            "model_name_or_path": str(base), "local_files_only": True,
                                            "expected_dimension": 8}))
    training_config = tmp_path / "training.yaml"
    training_config.write_text(yaml.safe_dump({"epochs": 1, "batch_size": per_device_batch_size,
                                               "gradient_accumulation_steps": gradient_accumulation_steps, "max_input_tokens": 32,
                                               "gradient_checkpointing": True, "lora_rank": 2,
                                               "lora_alpha": 4, "target_modules": ["query", "value"],
                                               "logging_steps": 1, "save_steps": 10, "threads": 1}))

    def record(index):
        return {"id": f"case-{index}", "group": f"group-{index}",
                "request": {"state": f"a {index}", "questions": {"q": {
                    "type": "choice", "instructions": "a", "criteria": {"b": "b", "c": "c"}}}},
                "answers": {"q": {"choice": "b" if index % 2 else "c"}}}

    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text("".join(json.dumps(record(i)) + "\n" for i in range(train_rows)))
    validation.write_text("".join(json.dumps(record(i)) + "\n" for i in range(train_rows, train_rows + 3)))
    output = tmp_path / "run"
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root / "src"), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc-per-node={world_size}", "-m", "jevembed.training", "--config", str(model_config),
               "--training-config", str(training_config), "--train-data", str(train),
               "--eval-data", str(validation), "--output", str(output), "--device", "cpu",
               "--dtype", "float32"]
    process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        pytest.fail(f"torchrun timed out after 600 seconds:\n{stdout[-4000:]}{stderr[-4000:]}")
    assert process.returncode == 0, stdout[-4000:] + stderr[-4000:]
    manifest = json.loads((output / "training_manifest.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    state = json.loads((output / "trainer_state.json").read_text())
    assert manifest["world_size"] == world_size
    assert manifest["effective_batch_size"] == per_device_batch_size * gradient_accumulation_steps * world_size
    assert manifest["lengths"]["train"]["questions"] == train_rows
    assert manifest["lengths"]["train"]["inputs"] == 3 * train_rows
    assert manifest["lengths"]["validation"]["questions"] == 3
    expected_steps = math.ceil(train_rows / (per_device_batch_size * gradient_accumulation_steps * world_size))
    assert state["global_step"] == expected_steps  # Uneven final batches must not hang or add a step.
    assert metrics["validation_base"]["questions"] == 3
    assert metrics["validation_base"]["choice_accuracy_n"] == 3
    assert metrics["validation_after"]["questions"] == 3
    assert (output / "adapter" / "adapter_model.safetensors").is_file()
    assert (output / "inference.yaml").is_file()
