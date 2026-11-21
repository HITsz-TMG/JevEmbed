# Sequential A100 inference for five models

This guide runs CLM-Qwen3.5-2B, Jeva-4B, both EmbeddingGemma decision checkpoints, and Nemotron one at a time. They use three incompatible Python environments, so each process exits before the next model starts. Kanana has its own environment and is not part of this five-model smoke run.

## Prepare the deployment

Use the same JevEmbed source revision on the download machine and the A100. Install Python 3.12 and Git. Create three environments and install one compatible extra set in each from the repository root. Choose CUDA-enabled PyTorch wheels that match the server before installing the JevEmbed extras.

```bash
python3.12 -m venv .venvs/decision
source .venvs/decision/bin/activate
python -m pip install -e '.[clm,jeva,server]'
deactivate

python3.12 -m venv .venvs/gemma
source .venvs/gemma/bin/activate
python -m pip install -e '.[gemma-decision,server]'
deactivate

python3.12 -m venv .venvs/standard
source .venvs/standard/bin/activate
python -m pip install -e '.[local,server]'
deactivate
```

Download the pinned model files on a machine with suitable Hugging Face access, then transfer the resulting directory to the A100:

```bash
python scripts/prepare_jev_models.py download --weights-dir /path/to/weights
```

On the A100, generate local configurations after setting `WEIGHTS_DIR` to the transferred directory:

```bash
python scripts/prepare_jev_models.py local-configs --weights-dir "$WEIGHTS_DIR"
```

The helper checks for required configuration and weight files before writing local YAML files under `configs/local/`. Jeva checkpoints that reference an external backbone still need Hugging Face access or a populated cache. Keep `configs/local/`, downloaded weights, and smoke outputs outside version control.

## Run each model

Set the Python executable for each environment, then run the five requests sequentially:

```bash
python scripts/smoke_a100_models.py \
  --decision-python /path/to/venv-decision/bin/python \
  --gemma-python /path/to/venv-gemma/bin/python \
  --standard-python /path/to/venv-standard/bin/python
```

The script runs one model process at a time, validates the response schema and probability distributions, and writes one JSON result and log per model plus `summary.json`. A successful `--explain` run only checks request compilation; omit that flag for real inference. Failed runs return a nonzero exit code and retain the corresponding log.

The script does not launch persistent or parallel services. To serve a model, start the JevEmbed API using that model's own environment and configuration, and stop it before switching environments. The API loads model weights on its first inference request, so check a real request rather than only `/v1/models`.
