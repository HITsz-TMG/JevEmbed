# Qwen3-Embedding-0.6B LoRA reference run

Qwen3-Embedding-0.6B was fine-tuned for one epoch on [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) `release-v2-redistributable`: 79,116 training questions and 3,723 validation questions. It used the same data and [one-epoch training recipe](../configs/training/lora.yaml) as the [KaLM reference run](OPEN_JEV_KALM_LORA.md): batch size 32, gradient accumulation 4, a 1,024-token training limit, LoRA rank 8 and alpha 16, Choice/Score temperature 0.1, and both Noul slopes 10. The final adapter is from step 619.

| Open-Jev validation metric | Base | Final adapter |
| --- | ---: | ---: |
| Overall hard-label accuracy (3,495 questions) | 30.73% (1,074/3,495) | 84.06% (2,938/3,495) |
| Validation objective loss (3,723 questions) | 4.180 | 0.451 |
| Choice accuracy (740 hard labels) | 40.00% | 74.46% |
| Score level accuracy (470 hard labels) | 29.36% | 80.43% |
| Score MAE (481 questions; lower is better) | 0.967 | 0.435 |
| Noul binary accuracy (2,285 questions) | 28.01% | 87.92% |

Overall accuracy pools the 740 Choice, 470 Score, and 2,285 Noul hard labels. The other 228 validation questions have soft targets or expected Score values and are excluded from this accuracy.

The base model and final adapter were also evaluated on all 231 public JevBench tasks with the [current Qwen3 configuration](../configs/qwen3-embedding-0.6b.yaml), BF16, the model's native input limit, and identical scoring and prompts. All tasks returned valid answers.

| JevBench metric | Base | Final adapter |
| --- | ---: | ---: |
| Overall accuracy | 54.55% (126/231) | 50.65% (117/231) |
| Choice accuracy | 56.12% (78/139) | 58.99% (82/139) |
| Score accuracy | 50.00% (9/18) | 38.89% (7/18) |
| Noul accuracy | 52.70% (39/74) | 37.84% (28/74) |
| Score expected-level MAE (lower is better) | 0.579 | 0.645 |

The validation gain did not carry over to the public JevBench total. Open-Jev Noul training rows have no criteria, whereas all 74 public JevBench Noul tasks provide true/false criteria; this difference may contribute to the Noul regression. These are public-subset results, not an official full-benchmark score.
