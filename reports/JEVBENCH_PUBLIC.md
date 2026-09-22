# JevBench public evaluation

Date: 2026-09-22. Four models were evaluated on all **231 available public tasks** using the unchanged JevEmbed input mappings and JevBench scoring implementation. E5 was also evaluated with explicit truncation. This is a public-subset evaluation, not the complete 534-task benchmark or an official leaderboard score.

Aggregated results, dependency versions, dataset hashes, and weight hashes are available in [jevbench-public.json](jevbench-public.json).

## Accuracy with default overflow policies

| Split | Tasks | KaLM v2.5 | Qwen3 0.6B | Qwen3 4B | E5 large instruct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy | 48 | 83.33% (40/48) | 91.67% (44/48) | 95.83% (46/48) | 95.83% (46/48) |
| Standard (original) | 72 | 55.56% (40/72) | 66.67% (48/72) | 59.72% (43/72) | 55.56% (40/72) |
| Hard | 111 | 38.74% (43/111) | 37.84% (42/111) | 42.34% (47/111) | 22.52% (25/111) |
| **Pooled public total** | **231** | 53.25% (123/231) | 58.01% (134/231) | 58.87% (136/231) | 48.05% (111/231) |

All four default configurations use `overflow_policy: error`. E5 accepts at most 512 tokens and rejected 53 Hard tasks; JevBench counts these refusals as incorrect. The other models support 32768 tokens and returned valid answers for all tasks. E5's default total therefore includes its input-length limitation.

| Primitive | Tasks | KaLM v2.5 | Qwen3 0.6B | Qwen3 4B | E5 large instruct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Choice | 139 | 49.64% (69/139) | 56.12% (78/139) | 61.87% (86/139) | 46.76% (65/139) |
| Score | 18 | 61.11% (11/18) | 50.00% (9/18) | 33.33% (6/18) | 50.00% (9/18) |
| Noul | 74 | 58.11% (43/74) | 63.51% (47/74) | 59.46% (44/74) | 50.00% (37/74) |

Score accuracy uses the highest-probability level, following JevBench. Its expected level is evaluated separately through MAE. Noul maps to `yes=p` and `no=1-p`; all 74 public Noul tasks provide true/false criteria. This suite therefore does not evaluate the criteria-free Noul path.

## E5 length-limit comparison

| Metric | E5 default: reject | E5 explicit: truncate |
| --- | ---: | ---: |
| Easy | 95.83% (46/48) | 95.83% (46/48) |
| Standard | 55.56% (40/72) | 55.56% (40/72) |
| Hard | 22.52% (25/111) | 30.63% (34/111) |
| Choice | 46.76% (65/139) | 51.80% (72/139) |
| Score | 50.00% (9/18) | 50.00% (9/18) |
| Noul | 50.00% (37/74) | 52.70% (39/74) |
| Pooled public total | 48.05% (111/231) | 51.95% (120/231) |
| Valid responses | 178/231 | 231/231 |
| Length-limit refusals | 53 | 0 |
| Tasks with truncated inputs | 0 | 53 |

The supplemental run changes only `overflow_policy` to `truncate`, retaining the 512-token limit. It truncates one input in each of 53 Hard tasks; the longest original input contains 3819 tokens. Truncation may remove decision-relevant evidence. The original requests are preserved in the artifacts, and the model receives truncated tokens. Results on the other 178 tasks are identical between the two E5 runs. The shipped E5 configuration remains unchanged.

| Model | Accuracy on the same 178 tasks within E5's limit |
| --- | ---: |
| KaLM v2.5 | 58.43% (104/178) |
| Qwen3 0.6B | 65.17% (116/178) |
| Qwen3 4B | 66.85% (119/178) |
| E5 large instruct | 62.36% (111/178) |

This matched subset contains all Easy and Standard tasks and 58 Hard tasks. It separates coverage from answer accuracy, but excludes many long Hard inputs and must not replace the 231-task headline result.

## Calibration, latency, and validity

| Metric | KaLM v2.5 | Qwen3 0.6B | Qwen3 4B | E5 large instruct | E5 truncate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Valid responses | 231/231 | 231/231 | 231/231 | 178/231 | 231/231 |
| Length-limit refusals | 0 | 0 | 0 | 53 | 0 |
| Infrastructure failures | 0 | 0 | 0 | 0 | 0 |
| Calibration samples | 231 | 231 | 231 | 178 | 231 |
| Score MAE samples | 18 | 18 | 18 | 15 | 18 |
| Score expected-level MAE | 0.8212 | 0.8030 | 0.8144 | 0.9570 | 0.8542 |
| Mean multiclass Brier score | 0.6428 | 0.6354 | 0.6256 | 0.6575 | 0.6711 |
| Top-label ECE, 10 bins | 0.1962 | 0.2320 | 0.2353 | 0.3053 | 0.2132 |
| Hard gold-distribution samples | 10 | 10 | 10 | 5 | 10 |
| Hard gold-distribution mean TVD | 0.2878 | 0.2786 | 0.2889 | 0.2770 | 0.2857 |
| P50 latency, all attempts (ms) | 21.87 | 27.53 | 40.94 | 14.39 | 17.15 |
| P95 latency, all attempts (ms) | 173.86 | 308.28 | 885.18 | 24.26 | 33.74 |
| Peak PyTorch allocated GPU memory (GiB) | 2.64 | 4.49 | 12.62 | 1.14 | 1.15 |

Lower MAE, Brier, ECE, and TVD are better. Calibration metrics include only returned distributions: E5 default has 178 samples, 15 Score items, and 5 gold-distribution items; the full-coverage runs have 231 samples, 18 Score items, and 10 gold-distribution items. These unequal-sample metrics are not direct comparisons.

Latency measures serial in-process `evaluate_with_trace`, with CUDA synchronization, including tokenization, embedding, scoring, and trace construction. Loading, three synthetic warmups, weight hashing, and evidence writes are excluded. These are one pass per task over varied inputs, not repeated measurements of one fixed request. E5 default latency includes 53 fast rejections; truncated E5 inputs are shorter than those processed by the other models. No HTTP/network latency or estimated provider cost is included.

## Protocol

- Hardware: NVIDIA H100 MIG, approximately 40 GB; models run sequentially on the same device with BF16 and four CPU threads.
- Choice and Score temperatures: **0.5**. Both Noul paths retain slope **1.0** and intercept **0.0**. Parameters were fixed before evaluation; no fitting was performed on benchmark tasks.
- Cache disabled; concurrency 1; no retries. The four default runs cover 924 attempts, and the supplemental E5 run adds 231. Only the supplemental E5 run truncates inputs.
- Original states, instructions, and criteria are preserved. Gold labels and provenance are used only by benchmark scoring.
- Dataset file and canonical hashes match the frozen JevBench manifest. Per-task raw evidence hashes and aggregate results were independently recomputed after the runs.
- The separate Judge tier and held-out items are unavailable. No full JevBench composite score is reported; pooled accuracy weights every available public task equally.
- Dependencies: Python 3.10.20, PyTorch 2.8.0+cu129, Transformers 4.51.0, sentence-transformers 5.3.0.

## Per-family accuracy

| Family | Tasks | KaLM v2.5 | Qwen3 0.6B | Qwen3 4B | E5 large instruct | E5 truncate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adequacy | 12 | 58.33% (7/12) | 50.00% (6/12) | 41.67% (5/12) | 50.00% (6/12) | 50.00% (6/12) |
| adversarial | 6 | 33.33% (2/6) | 33.33% (2/6) | 50.00% (3/6) | 50.00% (3/6) | 50.00% (3/6) |
| ambiguous | 7 | 57.14% (4/7) | 14.29% (1/7) | 0.00% (0/7) | 28.57% (2/7) | 28.57% (2/7) |
| extraction | 24 | 62.50% (15/24) | 70.83% (17/24) | 87.50% (21/24) | 79.17% (19/24) | 79.17% (19/24) |
| fact | 12 | 58.33% (7/12) | 83.33% (10/12) | 91.67% (11/12) | 91.67% (11/12) | 91.67% (11/12) |
| intent | 24 | 79.17% (19/24) | 91.67% (22/24) | 91.67% (22/24) | 83.33% (20/24) | 83.33% (20/24) |
| judge_hard | 17 | 41.18% (7/17) | 47.06% (8/17) | 47.06% (8/17) | 41.18% (7/17) | 41.18% (7/17) |
| long_policy | 19 | 36.84% (7/19) | 47.37% (9/19) | 36.84% (7/19) | 0.00% (0/19) | 10.53% (2/19) |
| multi_hop | 18 | 33.33% (6/18) | 16.67% (3/18) | 22.22% (4/18) | 5.56% (1/18) | 16.67% (3/18) |
| ordinal | 12 | 83.33% (10/12) | 66.67% (8/12) | 41.67% (5/12) | 66.67% (8/12) | 66.67% (8/12) |
| policy | 12 | 66.67% (8/12) | 75.00% (9/12) | 50.00% (6/12) | 66.67% (8/12) | 66.67% (8/12) |
| probability | 10 | 40.00% (4/10) | 50.00% (5/10) | 40.00% (4/10) | 20.00% (2/10) | 40.00% (4/10) |
| routing | 12 | 16.67% (2/12) | 66.67% (8/12) | 58.33% (7/12) | 16.67% (2/12) | 16.67% (2/12) |
| routing_hard | 5 | 60.00% (3/5) | 60.00% (3/5) | 100.00% (5/5) | 80.00% (4/5) | 80.00% (4/5) |
| temporal_numeric | 15 | 20.00% (3/15) | 26.67% (4/15) | 40.00% (6/15) | 33.33% (5/15) | 46.67% (7/15) |
| tool_selection | 12 | 100.00% (12/12) | 100.00% (12/12) | 100.00% (12/12) | 100.00% (12/12) | 100.00% (12/12) |
| tradeoff | 6 | 50.00% (3/6) | 33.33% (2/6) | 50.00% (3/6) | 0.00% (0/6) | 16.67% (1/6) |
| trap | 8 | 50.00% (4/8) | 62.50% (5/8) | 87.50% (7/8) | 12.50% (1/8) | 12.50% (1/8) |

## Reproduction

Use the [public-suite runner](../scripts/run_jevbench.py) with an existing JevBench checkout and the model configuration:

```bash
python scripts/run_jevbench.py --benchmark-root ../jevbench-main \
  --config configs/kalm-embedding-v2.5.yaml --device cuda --dtype bfloat16 \
  --output artifacts/jevbench/kalm

python scripts/run_jevbench.py --benchmark-root ../jevbench-main \
  --config configs/qwen3-embedding-0.6b.yaml --device cuda --dtype bfloat16 \
  --output artifacts/jevbench/qwen3-0.6b

python scripts/run_jevbench.py --benchmark-root ../jevbench-main \
  --config configs/qwen3-embedding-4b.yaml --device cuda --dtype bfloat16 \
  --output artifacts/jevbench/qwen3-4b

python scripts/run_jevbench.py --benchmark-root ../jevbench-main \
  --config configs/multilingual-e5-large-instruct.yaml --device cuda --dtype bfloat16 \
  --output artifacts/jevbench/e5

```

For the supplemental E5 run, copy its configuration to an ignored local configuration file, change `overflow_policy: error` to `overflow_policy: truncate`, and run with that file and a new output directory. The default E5 run exits with status 3 because it contains rejected inputs; it still attempts all 231 tasks and writes the complete summary.

For offline weights, add `--model-path ./models/<snapshot-directory>`. Each output directory contains `metadata.json`, `summary.json`, `predictions.jsonl`, a local accounting ledger, and raw request/response evidence. Choose a new output directory for every run. Runtime metadata can contain local paths, so these files belong in the ignored `artifacts/` directory.

| Model | Resolved upstream revision |
| --- | ---: |
| KaLM v2.5 | `753c6fe26abc20a32aeb162003aa03457d15db2f` |
| Qwen3 0.6B | Unknown; identified by file hashes |
| Qwen3 4B | `5cf2132abc99cad020ac570b19d031efec650f2b` |
| E5 large instruct | `274baa43b0e13e37fafa6428dbc7938e62e5c439` |

| Model / weight file | SHA-256 |
| --- | ---: |
| kalm-embedding-v2.5 / `model.safetensors` | `375fa304cee0e020c98a96ddff5a28a371891c5ea3f707dc7592e113342fce20` |
| qwen3-embedding-0.6b / `model.safetensors` | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |
| qwen3-embedding-4b / `model-00001-of-00002.safetensors` | `e70bfe3c970523fb7ef4eddffed2254ce3f1e7150c3de2af4342de129dd756f8` |
| qwen3-embedding-4b / `model-00002-of-00002.safetensors` | `ed1b87c8e9eb7e535a1a155e4fd00d9f4dba80e58a6db48a4c9f82cede7079c1` |
| multilingual-e5-large-instruct / `model.safetensors` | `dd6b6e4f52db0a7aff83a13d10e6c5342ef9f6ab799bad3221f4b35ef390fa85` |
