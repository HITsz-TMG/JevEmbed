# JevEmbed-Data test results

These are **base-model results without LoRA** on all 66,482 questions in the [JevEmbed-Data](https://huggingface.co/datasets/HIT-TMG/JevEmbed-Data) `test` split. Each run used BF16, its model configuration's prompts and scoring, and `overflow_policy: truncate`. KaLM and Qwen3 inputs were capped at 1,024 tokens, E5 at 512, and CLM at 2,048. KaLM, Qwen3, and E5 used Choice/Score temperature 0.1 and Noul slope 10; CLM used 0.01 and 100.

## Accuracy

Accuracy uses only hard targets: 17,487 Choice, 24,260 Score, and 22,363 Noul questions. Overall accuracy pools correct predictions across those 64,110 questions; it is not the mean of the three percentages.

| Model | Choice (17,487) | Score (24,260) | Noul (22,363) | Overall (64,110) |
| --- | ---: | ---: | ---: | ---: |
| `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` | 28.61% | 27.45% | 40.52% | 32.33% |
| `Qwen/Qwen3-Embedding-0.6B` | 33.02% | 28.56% | 40.05% | 33.79% |
| `Qwen/Qwen3-Embedding-4B` | 38.11% | 30.55% | 41.09% | 36.29% |
| `Qwen/Qwen3-Embedding-8B` | 43.20% | 33.49% | 42.57% | **39.30%** |
| `intfloat/multilingual-e5-large-instruct` | 32.07% | 24.27% | 40.54% | 32.07% |
| `Contrastive-LM/CLM-v0.1-8B` | 28.59% | 25.85% | 53.74% | 36.33% |

## Probability and score error

The split has 18,191 Choice, 24,287 Score, and 24,004 Noul questions. Accuracy excludes 704 Choice, 27 Score, and 1,641 Noul questions with non-hard targets. The error metrics include them. Score MAE compares the predicted expected level with its target; Noul MAE compares the predicted probability with its target probability. Brier and total variation distance (TVD) measure distribution error across all questions. Lower is better for every metric below.

| Model | Score MAE (24,287) | Noul MAE (24,004) | Brier (66,482) | TVD (66,482) |
| --- | ---: | ---: | ---: | ---: |
| `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` | 0.9990 | 0.5925 | 0.8489 | 0.6867 |
| `Qwen/Qwen3-Embedding-0.6B` | 1.0107 | 0.5928 | 0.8533 | 0.6823 |
| `Qwen/Qwen3-Embedding-4B` | 0.9211 | 0.5895 | 0.8366 | 0.6681 |
| `Qwen/Qwen3-Embedding-8B` | 0.8921 | 0.5894 | 0.8220 | 0.6593 |
| `intfloat/multilingual-e5-large-instruct` | 0.9905 | 0.5917 | 0.8402 | 0.6919 |
| `Contrastive-LM/CLM-v0.1-8B` | 1.2163 | 0.4555 | 0.9145 | 0.6335 |

## Reproduce

After installing JevEmbed and its training dependencies, run the [test evaluator](../scripts/evaluate_jevembed_data.py) from the repository root. For KaLM:

```bash
python scripts/evaluate_jevembed_data.py \
  --config configs/kalm-embedding-v2.5.yaml \
  --test-data path/to/JevEmbed-Data/data/test-00000-of-00001.parquet \
  --model-path path/to/KaLM-embedding-multilingual-mini-instruct-v2.5 \
  --device cuda --dtype bfloat16 \
  --max-input-tokens 1024 --overflow-policy truncate \
  --encode-batch-size 64 --chunk-questions 500 \
  --output results/kalm-test.json
```

Use the matching configuration and model path for the other models, with the token caps stated above. CLM also requires its projection checkpoint via `--projection-path`. The evaluated test Parquet shard has SHA-256 `b8c2660c9b1d7e1d298ec943ecc273337c46a724e10b8e7700712da3dd9f91c4`.
