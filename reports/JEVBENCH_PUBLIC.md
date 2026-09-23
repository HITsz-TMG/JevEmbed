# JevBench public evaluation

Evaluated 2026-09-23 with the shipped model configurations: Choice/Score temperature **0.1**, Noul slopes **10**, and retrieval-style Noul query encoding. All runs use BF16, disabled caching, and the same 231 public tasks. These are public-subset results, not an official full-benchmark score.

## Accuracy

| Model | Easy (48) | Standard (72) | Hard (111) | Overall (231) |
| --- | ---: | ---: | ---: | ---: |
| `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` | 93.75% (45/48) | 51.39% (37/72) | 35.14% (39/111) | 52.38% (121/231) |
| `Qwen/Qwen3-Embedding-0.6B` | 93.75% (45/48) | 59.72% (43/72) | 34.23% (38/111) | 54.55% (126/231) |
| `Qwen/Qwen3-Embedding-4B` | 97.92% (47/48) | 62.50% (45/72) | 39.64% (44/111) | 58.87% (136/231) |
| `Qwen/Qwen3-Embedding-8B` | 95.83% (46/48) | 63.89% (46/72) | 36.04% (40/111) | 57.14% (132/231) |
| `intfloat/multilingual-e5-large-instruct` (default) | 93.75% (45/48) | 54.17% (39/72) | 26.13% (29/111) | 48.92% (113/231) |
| `intfloat/multilingual-e5-large-instruct` (explicit truncation) | 93.75% (45/48) | 54.17% (39/72) | 33.33% (37/111) | 52.38% (121/231) |

E5's default 512-token policy rejected **53** tasks, counted as incorrect. The explicit truncation run processes them at the same 512-token limit; the other models return all 231 answers.

## By primitive

| Model | Choice (139) | Score (18) | Noul (74) |
| --- | ---: | ---: | ---: |
| `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` | 49.64% (69/139) | 61.11% (11/18) | 55.41% (41/74) |
| `Qwen/Qwen3-Embedding-0.6B` | 56.12% (78/139) | 50.00% (9/18) | 52.70% (39/74) |
| `Qwen/Qwen3-Embedding-4B` | 61.87% (86/139) | 33.33% (6/18) | 59.46% (44/74) |
| `Qwen/Qwen3-Embedding-8B` | 58.27% (81/139) | 33.33% (6/18) | 60.81% (45/74) |
| `intfloat/multilingual-e5-large-instruct` (default) | 46.76% (65/139) | 50.00% (9/18) | 52.70% (39/74) |
| `intfloat/multilingual-e5-large-instruct` (explicit truncation) | 51.80% (72/139) | 50.00% (9/18) | 54.05% (40/74) |

Score accuracy uses the highest-probability level. All 74 public Noul tasks provide true/false criteria; this suite does not test criteria-free Noul.

## Validity, calibration, and latency

| Model | Valid | Score MAE | Brier | ECE | P50 (ms) | P95 (ms) | GPU peak (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` | 231/231 | 0.6974 | 0.5485 | 0.0753 | 20.38 | 168.36 | 2.64 |
| `Qwen/Qwen3-Embedding-0.6B` | 231/231 | 0.5794 | 0.5295 | 0.0450 | 25.65 | 306.87 | 4.49 |
| `Qwen/Qwen3-Embedding-4B` | 231/231 | 0.6551 | 0.4950 | 0.0823 | 38.47 | 879.68 | 12.62 |
| `Qwen/Qwen3-Embedding-8B` | 231/231 | 0.5897 | 0.5035 | 0.0441 | 66.05 | 910.58 | 17.68 |
| `intfloat/multilingual-e5-large-instruct` (default) | 178/231 | 0.9166 | 0.5984 | 0.2369 | 14.13 | 20.94 | 1.14 |
| `intfloat/multilingual-e5-large-instruct` (explicit truncation) | 231/231 | 0.8252 | 0.6274 | 0.1484 | 16.95 | 33.03 | 1.15 |

Score MAE, Brier, and ECE use only valid returned answers, so E5's default-policy values cover fewer tasks. Lower values are better. Latency includes rejected attempts, so E5's fast length rejections affect its timing. It is serial in-process inference with CUDA synchronization and excludes loading, warmup, weight hashing, and disk writes.

## Protocol

The original JevBench states, instructions, criteria, and upstream scoring are unchanged. The model inputs use each shipped configuration's templates. No calibration parameters were fitted on benchmark tasks. The 8B run used a later JevEmbed revision that changed HTTP resource limits; the prompt, compiler, and scoring source hashes match the earlier runs. Both source revisions are recorded in the aggregate JSON. Raw local evidence may contain machine paths and is kept outside Git.

[Aggregated metrics and hashes](jevbench-public.json) are available without local paths. With an existing JevBench checkout and model weights available to the selected config, run:

```bash
python scripts/run_jevbench.py --benchmark-root ../jevbench-main \
  --config configs/kalm-embedding-v2.5.yaml --device cuda --dtype bfloat16 \
  --output artifacts/jevbench/kalm
```

Substitute another shipped `configs/<model-id>.yaml` and a new output directory for each model. For the E5 supplemental run, copy its config and set `overflow_policy: truncate`.
