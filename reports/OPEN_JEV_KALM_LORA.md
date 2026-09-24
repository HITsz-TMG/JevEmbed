# KaLM v2.5 LoRA reference run

KaLM v2.5 was fine-tuned for one epoch on [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) `release-v2-redistributable` at revision `c67699e13d0ae25e35b77165a4b6b079bedc8aba`: 79,116 training questions and 3,723 validation questions. The run used batch size 32, gradient accumulation 4, a 1,024-token limit, LoRA rank 8 and alpha 16, Choice/Score temperature 0.1, and both Noul slopes 10. The saved run used the [experiment configuration](../configs/experiments/kalm-embedding-v2.5-temperature-0.1.yaml) and [training recipe](../configs/training/lora.yaml).

| Validation metric | Base | Final adapter |
| --- | ---: | ---: |
| Overall hard-label accuracy (3,495 questions) | 30.24% | 76.68% |
| Validation objective loss (3,723 questions) | 3.909 | 0.592 |
| Choice accuracy (740 hard labels) | 40.95% | 62.84% |
| Score level accuracy (470 hard labels) | 24.26% | 63.62% |
| Score MAE (481 questions; lower is better) | 1.024 | 0.683 |
| Noul binary accuracy (2,285 questions) | 28.01% | 83.85% |

Overall hard-label accuracy pools correct Choice, Score, and Noul predictions across 740 + 470 + 2,285 questions. The remaining 228 validation questions have soft targets or expected Score values and are excluded from this accuracy.

The base model and final adapter were also evaluated on all 231 public JevBench tasks with the [current KaLM configuration](../configs/kalm-embedding-v2.5.yaml), BF16, and the model's native input limit. All tasks returned valid answers.

| JevBench metric | Base | Final adapter |
| --- | ---: | ---: |
| Overall accuracy | 51.52% (119/231) | 54.11% (125/231) |
| Choice accuracy | 49.64% (69/139) | 53.24% (74/139) |
| Score accuracy | 61.11% (11/18) | 77.78% (14/18) |
| Noul accuracy | 52.70% (39/74) | 50.00% (37/74) |
| Score expected-level MAE (lower is better) | 0.697 | 0.564 |

The public-subset overall gain is 2.60 percentage points. These results are not an official full-benchmark score.
