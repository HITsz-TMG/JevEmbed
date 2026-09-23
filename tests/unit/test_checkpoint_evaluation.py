"""Checkpoint retention and evaluation schedule without loading model weights."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

spec = importlib.util.spec_from_file_location(
    'checkpoint_evaluation', Path(__file__).resolve().parents[2] / 'scripts/evaluate_lora_checkpoints.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def checkpoint(root, step, complete=True):
    path = root / f'checkpoint-{step}'
    path.mkdir(parents=True)
    (path / 'adapter_config.json').write_text('{}')
    (path / 'adapter_model.safetensors').write_bytes(b'example adapter')
    if complete:
        (path / 'trainer_state.json').write_text(json.dumps({'global_step': step}))
    return path


def test_archive_survives_rotation_and_waits_for_completed_save(tmp_path):
    training, archive = tmp_path / 'training', tmp_path / 'archive'
    saved = checkpoint(training, 100)
    pending = checkpoint(training, 200, complete=False)
    evaluation.archive_checkpoints(training, archive)
    assert (archive / 'checkpoint-100' / 'adapter_model.safetensors').read_bytes() == b'example adapter'
    assert not (archive / 'checkpoint-200').exists()
    shutil.rmtree(saved)
    (pending / 'trainer_state.json').write_text('{')
    evaluation.archive_checkpoints(training, archive)
    assert not (archive / 'checkpoint-200').exists()
    (pending / 'trainer_state.json').write_text('{"global_step": 200}')
    evaluation.archive_checkpoints(training, archive)
    assert (archive / 'checkpoint-100').is_dir()
    assert (archive / 'checkpoint-200' / 'archive.json').is_file()


def test_schedule_requires_every_save_and_includes_final_once(tmp_path):
    training, archive = tmp_path / 'training', tmp_path / 'archive'
    training.mkdir()
    (training / 'training_manifest.json').write_text('{"training": {"save_steps": 100}}')
    (training / 'trainer_state.json').write_text('{"global_step": 219}')
    final = training / 'adapter'; final.mkdir()
    (final / 'adapter_model.safetensors').write_bytes(b'final adapter')
    checkpoint(archive, 100)
    with pytest.raises(RuntimeError, match='Missing checkpoint-200'):
        evaluation.evaluation_cases(training, archive)
    checkpoint(archive, 200)
    checkpoint(archive, 219)
    cases = evaluation.evaluation_cases(training, archive)
    assert [step for _, step, _ in cases] == [0, 100, 200, 219]
    assert cases[-1][2] == final
