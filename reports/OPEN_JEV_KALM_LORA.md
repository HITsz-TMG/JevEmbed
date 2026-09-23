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

The same base model and saved adapters were evaluated on all 231 public JevBench tasks with the [current KaLM configuration](../configs/kalm-embedding-v2.5.yaml): the updated Noul encoding (state and each `instructions + criterion` encoded as queries), Choice/Score temperature 0.1, Noul slopes 10, and the model's native input limit. Accuracy is the public-suite aggregate, not an official full-benchmark score.

| Training step | Overall accuracy | Change vs. base | Choice | Score | Noul |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (base) | 52.38% | — | 49.64% | 61.11% | 55.41% |
| 100 | 52.81% | +0.43 pp | 49.64% | 61.11% | 56.76% |
| 200 | 54.98% | +2.60 pp | 51.08% | 77.78% | 56.76% |
| 300 | 54.55% | +2.16 pp | 52.52% | 72.22% | 54.05% |
| 400 | 56.28% | +3.90 pp | 53.96% | 77.78% | 55.41% |
| 500 | 56.28% | +3.90 pp | 54.68% | 77.78% | 54.05% |
| 600 | 56.71% | +4.33 pp | 54.68% | 77.78% | 55.41% |
| 619 (final) | 55.41% | +3.03 pp | 53.24% | 77.78% | 54.05% |

All 231 tasks returned valid answers at each step. Step 600 had the highest observed public accuracy, but checkpoint selection on this same public subset is exploratory. Open-Jev Noul training rows have no criteria, while all 74 public JevBench Noul tasks supply criteria. The criteria-free legacy and retrieval encodings have the same cosine mathematically; the criteria-based JevBench path tests transfer to a different input mapping.
