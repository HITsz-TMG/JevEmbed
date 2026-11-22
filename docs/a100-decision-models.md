# Sequential A100 inference for four decision models

This guide deploys CLM-Qwen3.5-2B, Jeva-4B, and both EmbeddingGemma decision checkpoints from one fixed JevEmbed branch. It keeps their weights and two compatible Python environments on the server, then runs one model process at a time. Kanana has a separate environment and is not reinstalled here.

## Check out the tested source

The contribution branch is `https://github.com/aikacl/JevEmbed`, branch `feat/decision-embedding-model-adapters`. Run on the A100 as the account that owns the model workspace. Confirm the branch exists before cloning; stop if it is unavailable rather than testing a different revision:

```bash
export TASK_ROOT=/path/to/jevembed-deployment
mkdir -p "$TASK_ROOT"
git clone --branch feat/decision-embedding-model-adapters --single-branch \
  https://github.com/aikacl/JevEmbed.git "$TASK_ROOT/JevEmbed"
cd "$TASK_ROOT/JevEmbed"
git pull --ff-only origin feat/decision-embedding-model-adapters
git rev-parse HEAD | tee "$TASK_ROOT/source-commit.txt"
git status --short --branch
```

If this checkout already exists, run `git fetch origin`, check out the named branch, and fast-forward it before testing. Record the commit hash with the results. Do not pull another commit partway through inference. The expected available space is about 100 GB; confirm actual filesystem space and personal quota with `df -h / /mnt/data` before installing. Stop if the space or quota is lower than expected. Do not remove unrelated files or caches.

## Set task-local paths

Keep environments, model files, Hugging Face downloads, temporary installation files, and results under the task directory. The existing Kanana environment and its caches are separate and remain untouched.

```bash
cat > "$TASK_ROOT/env.sh" <<'SH'
export TASK_ROOT=/path/to/jevembed-deployment
export JEV_ROOT=$TASK_ROOT/JevEmbed
export JEV_WEIGHTS=$TASK_ROOT/weights
export JEV_RESULTS=$TASK_ROOT/results
export TMPDIR=$TASK_ROOT/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export PIP_NO_CACHE_DIR=1
export HF_XET_HIGH_PERFORMANCE=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_ENABLE_HF_TRANSFER
mkdir -p "$JEV_WEIGHTS" "$JEV_RESULTS" "$TMPDIR"
cd "$JEV_ROOT"
SH
source "$TASK_ROOT/env.sh"
```

Use two isolated environments because the model groups require different Transformers versions. Keep both environments and all four model snapshots; no disk cleanup between groups is needed when the 100 GB allocation is available.

## Install the Gemma environment

Use the Python 3.12 executable available on the server. If it is supplied by the existing Kanana venv, use that interpreter only to create a new venv; do not install into the Kanana environment.

```bash
export JEV_GROUP=gemma
export JEV_ENV=$TASK_ROOT/env-$JEV_GROUP
export HF_HOME=$TASK_ROOT/hf-$JEV_GROUP
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet
export HF_MODULES_CACHE=$HF_HOME/modules
mkdir -p "$HF_HOME"
/path/to/venv-kanana/bin/python -m venv "$JEV_ENV"
source "$JEV_ENV/bin/activate"
cd "$JEV_ROOT"
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir -e '.[gemma-decision,server]'
python -m pip check
python -m pip freeze > "$JEV_RESULTS/$JEV_GROUP-pip-freeze.txt"
deactivate
```

If the Kanana interpreter is not present, substitute the server's Python 3.12 path. The resulting venv is isolated and does not inherit Kanana's Torch or Transformers installation.

## Install the CLM and Jeva environment

```bash
export JEV_GROUP=decision
export JEV_ENV=$TASK_ROOT/env-$JEV_GROUP
export HF_HOME=$TASK_ROOT/hf-$JEV_GROUP
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet
export HF_MODULES_CACHE=$HF_HOME/modules
mkdir -p "$HF_HOME"
/path/to/venv-kanana/bin/python -m venv "$JEV_ENV"
source "$JEV_ENV/bin/activate"
cd "$JEV_ROOT"
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir torch==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir -e '.[clm,jeva,server]'
python -m pip check
jeva-doctor
python -m pip freeze > "$JEV_RESULTS/$JEV_GROUP-pip-freeze.txt"
deactivate
```

Jeva installs its inference runtime from the pinned GitHub commit in `pyproject.toml`, so GitHub access is needed. The default reference path may be slower than a build with optional Qwen3.5 kernels.

## Download and configure the four checkpoints

Download each model separately so logs identify any incomplete transfer. `--model` is required on every invocation. The helper uses the immutable revisions in the YAML files and downloads native inference artifacts without executing repository code.

| Order | Model ID | Environment |
| --- | --- | --- |
| 1 | `embeddinggemma-2-ft-typed-decision` | `gemma` |
| 2 | `qnaug-embeddinggemma2-decision` | `gemma` |
| 3 | `clm-qwen3.5-2b` | `decision` |
| 4 | `jeva-4b` | `decision` |

For each model, activate the corresponding venv and export its matching `JEV_GROUP`, `JEV_ENV`, `HF_HOME`, `HF_HUB_CACHE`, `HF_XET_CACHE`, and `HF_MODULES_CACHE` as shown above. Then run:

```bash
cd "$JEV_ROOT"
export MODEL_ID=embeddinggemma-2-ft-typed-decision
python scripts/prepare_jev_models.py download \
  --weights-dir "$JEV_WEIGHTS" --model "$MODEL_ID"
python scripts/prepare_jev_models.py local-configs \
  --weights-dir "$JEV_WEIGHTS" --model "$MODEL_ID"
```

Repeat with the other three IDs. Jeva's checkpoint can refer to an external Qwen3.5 backbone. Keep Hub access enabled and enough disk space available; its first inference may download the extra backbone into the task-local HF cache. `local-configs` reports this condition. It checks files and shards, not whether inference will succeed.

For the first smoke run, make a local copy suitable for A100 memory and short examples. Apply this to the current model's file after `local-configs` creates it:

```bash
MODEL_CONFIG="configs/local/$MODEL_ID.local.yaml" python - <<'PY'
import os
from pathlib import Path
import yaml
path = Path(os.environ['MODEL_CONFIG'])
config = yaml.safe_load(path.read_text(encoding='utf-8'))
config['batch_size'] = 1
config['cache_capacity'] = 128
config['max_input_tokens'] = min(config['max_input_tokens'], 2048)
path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
print(path)
PY
```

This only changes the ignored local config. Gemma must use BF16 or FP32, not FP16.

## Run real inference and preserve the reports

The smoke runner starts an isolated subprocess per model and processes the four models sequentially. Run it with both environments installed and all four local YAML files prepared:

```bash
cd "$JEV_ROOT"
source "$TASK_ROOT/env-decision/bin/activate"
python scripts/smoke_a100_models.py \
  --decision-python "$TASK_ROOT/env-decision/bin/python" \
  --gemma-python "$TASK_ROOT/env-gemma/bin/python" \
  --timeout 3600 \
  --output-dir "$JEV_RESULTS/four-models-gpu"
```

Do not add `--explain`: that flag compiles requests without loading weights. A model passes only if its row in `summary.json` has `status: passed`, the report has `mode: real_inference` and `ok: true`, and the command exits with code 0. A failure keeps that model's log; fix and retry before moving on. A passing smoke check validates loading and response structure, not task accuracy or probability calibration.

Save a concise system and runtime record alongside the reports:

```bash
df -h / /mnt/data > "$JEV_RESULTS/disk-after.txt"
nvidia-smi > "$JEV_RESULTS/nvidia-smi.txt"
git -C "$JEV_ROOT" rev-parse HEAD > "$JEV_RESULTS/source-commit.txt"
```

If Jeva's backbone was cached under the `decision` HF home, keep it for repeat inference. Do not clear the cache as part of this deployment.

## Verify the existing Kanana adapter

Kanana is not included in the four-model runner because its Transformers 5.5.3 environment is incompatible with these groups. Do not reinstall or modify its environment. Since Kanana was already run on the A100, the deployment can rely on that previous real run; record the source commit and result if available. If the weights and environment remain available and a fresh check is needed, use the current checkout and existing interpreter:

```bash
cd "$JEV_ROOT"
PYTHONPATH="$JEV_ROOT/src" /path/to/venv-kanana/bin/python -m jevembed \
  --config configs/kanana-2-v-decide-3b-2611.yaml \
  --input examples/kanana_decide.json \
  --output "$JEV_RESULTS/kanana.json"
```

The pull request must describe which adapters had fresh inference on this branch and which result is from the earlier Kanana run. Do not label the four-model smoke run as a Kanana test.

## Make the results easy to inspect

Create `$JEV_RESULTS/index.html` after inference with a row for each model, its pass/fail result, predicted Choice/Score/Noul values, probabilities, token usage, and expandable raw response and log text. The runner records total start and finish times, not per-model elapsed time; show a model's elapsed time only when it was measured and otherwise label it “not recorded.” Generate the page from saved JSON and reports; do not load models to populate it. Serve it only on loopback:

```bash
cd "$JEV_RESULTS"
python -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/` through Codex's browser or forwarded port. If neither is available, from Windows start an SSH tunnel and then open the same URL locally:

```powershell
ssh -L 8765:127.0.0.1:8765 zxp_hitsz
```

Keep the server process and tunnel open while viewing the page. Do not bind this temporary results server to a public address.

## Open the pull request after the tests pass

Once all four current-branch inference runs pass, review the changed-file list and ensure it removes Nemotron while retaining the Kanana adapter. Do not commit A100 weights, local YAML, logs, result pages, or personal server details. Commit only source, model configs/examples, docs, and tests, then create a PR from `aikacl:JevEmbed` branch `feat/decision-embedding-model-adapters` into `HITsz-TMG/JevEmbed:main`. Include the tested source commit and summarize the four fresh A100 results. State Kanana's earlier test separately. If all model tests do not pass, fix and rerun them before opening the PR.
