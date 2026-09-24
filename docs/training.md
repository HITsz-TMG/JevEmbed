# LoRA training for Jev tasks

JevEmbed fine-tunes Sentence Transformers embedding encoders with PEFT. The input compiler, prompts, pooling, and scoring match inference for those models. Only LoRA parameters are updated; the base weights remain frozen. This implementation supports one CPU or one visible CUDA device per process. CLM's paired-projection backend is currently inference-only.

## Install

```bash
python -m pip install -e '.[train]' -c requirements-models-tested.txt
```

The training extra adds PEFT 0.18.1, Accelerate 1.13.0, and Datasets 4.8.4 to the supported local inference stack. Loading adapters in an inference-only installation additionally requires `python -m pip install -e '.[adapter]'`. No bitsandbytes, FlashAttention, external tracking service, or model upload is required.

The implementation follows the [Sentence Transformers PEFT interface](https://www.sbert.net/examples/sentence_transformer/training/peft/README.html) and its [custom loss contract](https://www.sbert.net/docs/sentence_transformer/loss_overview.html#custom-loss-functions). It uses `SentenceTransformerTrainer`, `add_adapter`, and differentiable model forwards; `encode()` is used only for evaluation.

## Prepare Open-Jev

The data adapter supports [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev), defaulting to `release-v2-redistributable` at commit `c67699e13d0ae25e35b77165a4b6b079bedc8aba`.

```bash
# Optional, when the official endpoint is unavailable:
# export HF_ENDPOINT=https://hf-mirror.com

python -m jevembed.training.open_jev --output artifacts/open-jev
```

The converter downloads the frozen raw JSONL files and preserves all **79,116 train rows** and **3,723 validation rows**, including repeated supervised inputs. It never downloads or trains on calibration, test, or OOD splits. IDs, groups, and exact state/question pairs are checked for cross-split overlap; conflicting targets are rejected. `manifest.json` records the revision, input hashes, task counts, and conversion policy.

`--subset` selects one other published config; subsets are not automatically combined because some overlap. `--revision` requires an immutable commit SHA. For an already downloaded snapshot, use `--source-dir ./data/open-jev-snapshot`, containing `raw/<subset>/train.jsonl.gz` and `validation.jsonl.gz`.

`--deduplicate` is optional and changes the official row counts. For the default release it produces 77,898 train and 3,657 validation rows. It is not used for a full 79,116-row training run.

| Open-Jev input | JevEmbed mapping |
| --- | --- |
| `state`, `question` | Original state and instructions |
| Choice `options`, `target` | Exact option strings as labels with null descriptions; full target distribution |
| Score `options`, `target` | Ordered criterion descriptions; full target distribution over level indices |
| Noul `options`, `target` | Locate `yes` by label and use its probability; criteria-free Noul |
| IDs and groups | Split validation only |
| Metadata, provenance, source, and split | Excluded from model inputs |

Noul source rows provide no descriptive true/false criteria, so the converter does not invent them. Non-index Score values, unknown task kinds, and independent multilabel vectors are rejected instead of being interpreted as categorical distributions. These are controlled reference targets, not official Jev labels or model confidence measurements.

Every Noul input is encoded as a query under `Retrieve semantically similar text.` Without criteria, the question and state are separate queries, and the logit is `noul_without_criteria.slope * cos(question, state) + noul_without_criteria.intercept`. With criteria, one query contains the question and state on separate lines; two more contain `true: {criterion}` and `false: {criterion}`. The logit is `noul_with_criteria.slope * (cos(question+state, true_criterion) - cos(question+state, false_criterion)) + noul_with_criteria.intercept`. The public response stays `{"type":"noul","noul":probability}` in either case.

## Train

Validate the data contract without loading weights:

```bash
python -m jevembed.training \
  --config configs/kalm-embedding-v2.5.yaml \
  --training-config configs/training/lora.yaml \
  --train-data artifacts/open-jev/train.jsonl \
  --eval-data artifacts/open-jev/validation.jsonl \
  --output artifacts/lora/kalm --validate-only
```

Run the same command without `--validate-only` to train on a CUDA device in BF16. Select one GPU with `CUDA_VISIBLE_DEVICES` when several are available. CPU training uses `--device cpu --dtype float32`. Add `--model-path ./models/<snapshot>` to use offline base weights without changing the public model configuration.

The shared [training configuration](../configs/training/lora.yaml) defaults to **one epoch** and controls LoRA rank/alpha/dropout, learning rate, question batch size, gradient accumulation, checkpoint frequency, seed, and token limits. The command above targets KaLM v2.5; use the corresponding model configuration to run Qwen3 0.6B, Qwen3 4B, or E5. Results from the KaLM and Qwen3 0.6B reference runs are linked in the [README](../README.md#lora-fine-tuning).

| Model | Automatic LoRA target suffixes |
| --- | --- |
| KaLM v2.5 | `q_proj`, `v_proj` |
| Qwen3 0.6B / 4B | `q_proj`, `v_proj` |
| E5 large instruct | `query`, `value` |

Explicit `target_modules` must name existing linear modules. The trainer checks that only LoRA parameters are trainable. Trainable adapter weights use FP32, including when the frozen base uses BF16.

The default effective batch is **128 questions**: batch size **32** with four accumulation steps. Each question encodes its query and all candidates, so memory also depends on candidate counts and sequence lengths. Reduce the question batch size for smaller GPUs or larger models. Different questions' candidates are never used as in-batch negatives.

The generic recipe caps inputs at **1024 tokens** and rejects overflow. Token lengths are checked for every training and validation input before optimization. Increase `max_input_tokens` for a longer-context model, or explicitly select `overflow_policy: truncate`; the effective limit never exceeds the model's supported limit. E5 remains limited to 512 tokens. Truncation counts are recorded because cutting a state may remove evidence required by its label. `--validate-only` checks structure and split isolation, not token lengths.

## Supervision format and objectives

Each JSONL row contains `id`, optional `group`, a standard Jev `request`, and `answers` keyed by the same question IDs. A request may contain multiple questions; each is one equally weighted training example.

```json
{"id":"route-001","group":"conversation-001","request":{"state":"My parcel has not arrived.","questions":{"route":{"type":"choice","instructions":"Route the support request.","criteria":{"shipping":"Delivery or tracking","billing":"Invoices or payments"}}}},"answers":{"route":{"choice":"shipping"}}}
```

Each answer contains exactly one supervision field:

| Task | Target example | Objective |
| --- | --- | --- |
| Choice | `{"choice":"shipping"}` | Cross-entropy over this request's candidates |
| Choice | `{"probabilities":{"shipping":0.8,"billing":0.2}}` | Soft-target cross-entropy |
| Score, labeled level | `{"level":2}` | Cross-entropy over ordinal levels |
| Score, target distribution | `{"probabilities":{"0":0.1,"1":0.2,"2":0.7}}` | Soft-target cross-entropy |
| Score, expected value | `{"score":1.6}` | Squared expected-level error, divided by `(number_of_levels - 1)^2` |
| Noul, either mapping | `{"noul":true}` or `{"noul":0.7}` | Binary cross-entropy with the inference logit |

Choice/Score logits are cosine similarities divided by their configured temperatures. Criteria-based Noul uses a true-minus-false similarity difference and `noul_with_criteria` slope/intercept; criteria-free Noul uses a cosine and `noul_without_criteria`. These scoring parameters stay fixed; training does not fit calibration parameters. The current defaults are temperature **0.1** for Choice/Score and slope **10** with intercept **0** for both Noul parameter pairs.

Cosines are bounded. With the default slope/intercept, criteria-based Noul can only output approximately 0.000000002–0.999999998; criteria-free Noul approximately 0.000045–0.999955. LoRA can change rankings and similarities but does not remove those theoretical probability bounds. Hard targets therefore do not imply that training loss can reach zero.

Training loss averages questions across task types, so short-term changes also reflect the sampled task mix and label difficulty. The default learning rate warms up over the first 10% of optimizer steps. Compare the fixed validation split before and after training, including task accuracy and MAE, before judging convergence. Choice and Score use their configured temperatures; Noul uses its slope and intercept instead.

Small synthetic format examples are in [examples/training](../examples/training/README.md). They are for checking the pipeline, not for measuring model quality. JevBench remains a separate evaluation set and is not mixed into training.

## Outputs, inference, and resume

An output directory contains:

- `adapter/`: adapter configuration and LoRA safetensors; no copied base weights.
- `inference.yaml`: base model, adapter location, original prompts/scoring, and effective token policy.
- `training_manifest.json`: parameters, dataset hashes, base identity, token checks, and trainable parameter counts.
- `metrics.json`: training metrics and before/after validation metrics when validation data is supplied.
- `checkpoint-*/`: adapter, optimizer, scheduler, RNG, and Trainer state for resume.

```bash
python -m jevembed --config artifacts/lora/kalm/inference.yaml \
  --input examples/official_choice_exchange.json

python -m jevembed.training \
  --config configs/kalm-embedding-v2.5.yaml \
  --training-config configs/training/lora.yaml \
  --train-data artifacts/open-jev/train.jsonl \
  --eval-data artifacts/open-jev/validation.jsonl \
  --output artifacts/lora/kalm \
  --resume-from-checkpoint artifacts/lora/kalm/checkpoint-100
```

Resume requires the original output directory and matching settings/data. It restores the adapter, optimizer, scheduler, and RNG state. Adapter-only inference is different from optimizer-state resume. Use checkpoints produced by your own trusted runs.

The final adapter is the final training step, not an automatically selected best checkpoint. Epoch validation reports loss. Final validation additionally reports hard-label accuracy where defined, Score MAE, Noul MAE, and Brier/TVD against target distributions, each with its denominator. Soft targets are not silently converted to hard labels. Without `--eval-data`, no held-out metrics are claimed.

## Compare checkpoints on JevBench

Start the checkpoint watcher while training is running, before older saves rotate out:

```bash
python scripts/evaluate_lora_checkpoints.py \
  --training-output artifacts/lora/kalm \
  --benchmark-root ../jevbench-main \
  --config configs/kalm-embedding-v2.5.yaml \
  --output artifacts/jevbench/kalm-checkpoints \
  --wait-for-training
```

For offline base weights, add `--model-path ./models/kalm`. Optionally pass the training process ID with `--training-pid` so the watcher detects an interrupted job. The watcher archives completed adapter saves every five seconds, then waits for final training metrics before running GPU evaluations. It does not keep optimizer state. Run it in a persistent session if it must survive a terminal disconnect.

The comparison includes the base model, every scheduled checkpoint, and the final adapter. Missing scheduled checkpoints cause an error. Each run evaluates all 231 public tasks using the upstream scoring implementation, with identical prompts, temperature, and input limits. The supplied **base inference configuration** controls evaluation length; it does not inherit the training token cap. The example uses the model's native context limit for both base and adapted models, while training remains capped at 1024 tokens.

Results include per-checkpoint raw evidence, `comparison.json`, and a `COMPARISON.md` table of accuracy, changes from the base model, task-specific accuracy, and Score MAE. These are exploratory comparisons on public tasks, not an official full-benchmark score or an independent test of a checkpoint selected using these same results. Runtime artifacts may contain local paths and remain under the ignored `artifacts/` directory.

The generated inference configuration uses the original base model identifier. To use local base weights, change its `model_name_or_path` and set `local_files_only: true`, or use `dataclasses.replace` in Python. Relative adapter paths resolve from the working directory. Keep run artifacts and data under ignored directories; generated runtime files may contain local paths.

Multi-GPU/DDP, QLoRA, automatic task reweighting, and adapter merging are outside this implementation.
