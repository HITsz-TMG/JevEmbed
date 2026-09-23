"""Archive LoRA checkpoints during training, then compare them on public JevBench."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_checkpoints(training, archive):
    """Trainer state is written after adapter weights; copy completed saves only."""
    archive.mkdir(parents=True, exist_ok=True)
    for source in sorted(training.glob('checkpoint-*')):
        if not source.name.removeprefix('checkpoint-').isdigit():
            continue
        destination = archive / source.name
        if destination.exists():
            continue
        files = ('adapter_config.json', 'adapter_model.safetensors', 'trainer_state.json')
        try:
            state = json.loads((source / 'trainer_state.json').read_text())
            if state['global_step'] != int(source.name.split('-')[-1]):
                raise ValueError('Checkpoint directory and recorded step disagree')
            json.loads((source / 'adapter_config.json').read_text())
            if not all((source / name).is_file() for name in files):
                continue
            temporary = archive / ('.' + source.name + '.partial')
            temporary.mkdir(exist_ok=True)
            for name in files:
                shutil.copy2(source / name, temporary / name)
            write_json(temporary / 'archive.json', {
                'step': state['global_step'],
                'sha256': {name: digest(temporary / name) for name in files},
                'purpose': 'Inference adapter snapshot; optimizer state is not archived.',
            })
            temporary.rename(destination)
            print('ARCHIVED', source.name, flush=True)
        except (FileNotFoundError, json.JSONDecodeError):
            # Retry a save still in progress. A rotated save will be reported
            # as missing when checking the complete schedule before evaluation.
            continue


def evaluation_cases(training, archive):
    manifest = json.loads((training / 'training_manifest.json').read_text())
    final = json.loads((training / 'trainer_state.json').read_text())['global_step']
    interval = manifest['training']['save_steps']
    cases = [('base', 0, None)]
    for step in range(interval, final, interval):
        adapter = archive / f'checkpoint-{step}'
        if not (adapter / 'adapter_model.safetensors').is_file():
            raise RuntimeError(f'Missing checkpoint-{step}; cannot claim a complete comparison')
        cases.append((f'step-{step}', step, adapter))
    adapter = training / 'adapter'
    if not (adapter / 'adapter_model.safetensors').is_file():
        raise RuntimeError('Final adapter has not been saved')
    cases.append((f'step-{final}', final, adapter))
    return cases


def compare(output, rows, protocol):
    write_json(output / 'comparison.json', {'protocol': protocol, 'results': rows})
    lines = ['# KaLM LoRA checkpoint comparison', '',
             '231 public JevBench tasks; not an official full-benchmark score.', '',
             'Identical base configuration, input mapping, context policy, and scoring for every checkpoint. '
             'Checkpoint comparisons are exploratory; do not treat benchmark-selected checkpoints as held-out results.', '',
             '| Step | Accuracy | Change (pp) | Choice | Score | Noul | Score MAE | Valid |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        s = row['summary']; overall = s['overall']; kinds = s['primitives']
        lines.append(f"| {row['step']} | {overall['accuracy']:.2%} | {row['accuracy_delta_pp']:+.2f} | "
                     f"{kinds['choice']['accuracy']:.2%} | {kinds['score']['accuracy']:.2%} | "
                     f"{kinds['noul']['accuracy']:.2%} | {overall['ordinal_mae']:.4f} | "
                     f"{overall['n_valid']}/{overall['n_planned']} |")
    (output / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output', type=Path, required=True)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='Same base inference config for every run')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for-training', action='store_true')
    parser.add_argument('--training-pid', type=int, help='Optional process to monitor while waiting')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float32'])
    args = parser.parse_args(argv)
    import yaml
    training, output = args.training_output.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A lock prevents two watchers from launching duplicate GPU evaluations.
    import fcntl
    lock = (output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'waiting_for_training', 'watcher_pid': os.getpid(), 'completed': []}
    status_path = output / 'status.json'
    try:
        write_json(status_path, state)
        while True:
            archive_checkpoints(training, output / 'adapters')
            if (training / 'metrics.json').exists():
                # metrics.json is written only after final validation. Confirm
                # it is completely written before starting GPU evaluation.
                try:
                    json.loads((training / 'metrics.json').read_text())
                    break
                except json.JSONDecodeError:
                    pass
            if not args.wait_for_training:
                raise RuntimeError('Training has not completed; use --wait-for-training to archive and wait')
            if args.training_pid:
                try:
                    os.kill(args.training_pid, 0)
                except ProcessLookupError:
                    raise RuntimeError('Training process exited before producing final metrics')
            time.sleep(5)

        config = yaml.safe_load(args.config.read_text())
        if config.get('adapter_name_or_path'):
            raise ValueError('--config must describe the base model without an adapter')
        protocol = {'scope': '231 public tasks; not an official full score',
                    'model_id': config['model_id'], 'scoring': config['scoring'],
                    'max_input_tokens': config.get('max_input_tokens', 'model_default'),
                    'overflow_policy': config.get('overflow_policy', 'error'),
                    'dtype': args.dtype, 'training_manifest_sha256': digest(training / 'training_manifest.json'),
                    'base_config_sha256': digest(args.config),
                    'checkpoint_selection': 'All scheduled saves and final adapter; no benchmark-based tuning'}
        cases = evaluation_cases(training, output / 'adapters')
        rows = []
        script = Path(__file__).with_name('run_jevbench.py')
        baseline_hashes = None
        for name, step, adapter in cases:
            state.update(status='evaluating', current=name)
            write_json(status_path, state)
            cfg = dict(config)
            cfg['adapter_name_or_path'] = str(adapter) if adapter else None
            cfg['adapter_revision'] = None
            config_path = output / f'{name}.yaml'
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            result_dir = output / 'runs' / name
            result_dir.parent.mkdir(exist_ok=True)
            command = [sys.executable, '-u', str(script), '--benchmark-root', str(args.benchmark_root.resolve()),
                       '--config', str(config_path), '--output', str(result_dir),
                       '--device', args.device, '--dtype', args.dtype]
            if args.model_path:
                command += ['--model-path', str(args.model_path.resolve())]
            print('EVALUATING', name, flush=True)
            with (output / f'{name}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = json.loads((result_dir / 'summary.json').read_text())
            metadata = json.loads((result_dir / 'metadata.json').read_text())
            hashes = {k: metadata[k] for k in ('dataset_sha256', 'model_weight_sha256',
                                              'benchmark_code_sha256', 'jevembed_code_sha256')}
            if baseline_hashes is None:
                baseline_hashes = hashes
            elif hashes != baseline_hashes:
                raise RuntimeError('Data, base weights, or evaluation code changed between runs')
            if summary['overall']['n_attempted'] != 231 or summary['overall']['n_valid'] != 231:
                raise RuntimeError('Incomplete benchmark coverage')
            base_accuracy = rows[0]['summary']['overall']['accuracy'] if rows else summary['overall']['accuracy']
            rows.append({'step': step, 'adapter_sha256': digest(adapter / 'adapter_model.safetensors') if adapter else None,
                         'accuracy_delta_pp': 100 * (summary['overall']['accuracy'] - base_accuracy), 'summary': summary})
            compare(output, rows, protocol)
            state['completed'].append(name)
            write_json(status_path, state)
        state.update(status='completed', current=None)
        write_json(status_path, state)
        print('COMPLETED', len(rows), 'models/checkpoints', flush=True)
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(status_path, state)
        raise


if __name__ == '__main__':
    main()
