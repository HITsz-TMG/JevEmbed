# JevBench public evaluation

Evaluated 2026-09-24 with the shipped model configurations: Choice/Score temperature **0.1** and Noul slopes **10**. All runs use BF16, disabled caching, and the same 231 public tasks. These are public-subset results, not an official full-benchmark score.

## Accuracy

| Model | Easy (48) | Standard (72) | Hard (111) | Overall (231) |
| --- | ---: | ---: | ---: | ---: |
| `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` | 91.67% (44/48) | 50.00% (36/72) | 35.14% (39/111) | 51.52% (119/231) |
| `Qwen/Qwen3-Embedding-0.6B` | 93.75% (45/48) | 61.11% (44/72) | 36.94% (41/111) | 56.28% (130/231) |
| `Qwen/Qwen3-Embedding-4B` | 89.58% (43/48) | 62.50% (45/72) | 39.64% (44/111) | 57.14% (132/231) |
| `Qwen/Qwen3-Embedding-8B` | 93.75% (45/48) | 69.44% (50/72) | 36.04% (40/111) | 58.44% (135/231) |
| `intfloat/multilingual-e5-large-instruct` (default) | 95.83% (46/48) | 55.56% (40/72) | 27.03% (30/111) | 50.22% (116/231) |
| `intfloat/multilingual-e5-large-instruct` (explicit truncation) | 95.83% (46/48) | 55.56% (40/72) | 36.04% (40/111) | 54.55% (126/231) |

E5's default 512-token policy rejected **53** tasks, counted as incorrect. The explicit truncation run processes them at the same 512-token limit; the other models return all 231 answers.

## By primitive

| Model | Choice (139) | Score (18) | Noul (74) |
| --- | ---: | ---: | ---: |
| `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` | 49.64% (69/139) | 61.11% (11/18) | 52.70% (39/74) |
| `Qwen/Qwen3-Embedding-0.6B` | 56.12% (78/139) | 50.00% (9/18) | 58.11% (43/74) |
| `Qwen/Qwen3-Embedding-4B` | 61.87% (86/139) | 33.33% (6/18) | 54.05% (40/74) |
| `Qwen/Qwen3-Embedding-8B` | 58.27% (81/139) | 33.33% (6/18) | 64.86% (48/74) |
| `intfloat/multilingual-e5-large-instruct` (default) | 46.76% (65/139) | 50.00% (9/18) | 56.76% (42/74) |
| `intfloat/multilingual-e5-large-instruct` (explicit truncation) | 51.80% (72/139) | 50.00% (9/18) | 60.81% (45/74) |

Score accuracy uses the highest-probability level. All 74 public Noul tasks provide true/false criteria; this suite does not test criteria-free Noul.

## Validity, calibration, and latency

| Model | Valid | Score MAE | Brier | ECE | P50 (ms) | P95 (ms) | GPU peak (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` | 231/231 | 0.6974 | 0.5521 | 0.0868 | 24.65 | 181.72 | 2.64 |
| `Qwen/Qwen3-Embedding-0.6B` | 231/231 | 0.5794 | 0.5255 | 0.0485 | 30.16 | 309.69 | 4.49 |
| `Qwen/Qwen3-Embedding-4B` | 231/231 | 0.6551 | 0.5053 | 0.0701 | 46.87 | 878.23 | 12.62 |
| `Qwen/Qwen3-Embedding-8B` | 231/231 | 0.5897 | 0.5075 | 0.0585 | 74.91 | 887.41 | 17.68 |
| `intfloat/multilingual-e5-large-instruct` (default) | 178/231 | 0.9166 | 0.5966 | 0.2500 | 15.62 | 24.15 | 1.14 |
| `intfloat/multilingual-e5-large-instruct` (explicit truncation) | 231/231 | 0.8252 | 0.6274 | 0.1689 | 18.72 | 36.41 | 1.15 |

Score MAE, Brier, and ECE use only valid returned answers, so E5's default-policy values cover fewer tasks. Lower values are better. Latency includes rejected attempts, so E5's fast length rejections affect its timing. It is serial in-process inference with CUDA synchronization and excludes loading, warmup, weight hashing, and disk writes.

## Protocol

The original JevBench states, instructions, criteria, and upstream scoring are unchanged. The model inputs use each shipped configuration's templates. No calibration parameters were fitted on benchmark tasks. All six runs use the same JevEmbed source revision; its hashes are recorded in the aggregate JSON.

[Aggregated metrics and hashes](jevbench-public.json) are available without local paths. With an existing JevBench checkout and model weights available to the selected config, run:

```bash
python scripts/run_jevbench.py --benchmark-root ../jevbench-main \
  --config configs/kalm-embedding-v2.5.yaml --device cuda --dtype bfloat16 \
  --output artifacts/jevbench/kalm
```

Substitute another shipped `configs/<model-id>.yaml` and a new output directory for each model. For the E5 supplemental run, copy its config and set `overflow_policy: truncate`.
