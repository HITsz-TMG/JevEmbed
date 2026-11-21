# Additional embedding configurations

## Newly integrated decision checkpoints

The following two models were checked against their official Hugging Face cards and source files on 2026-11-21. They are outside the JevEmbed upstream Model list and use dedicated decision adapters because their learned heads already produce probabilities; treating them as ordinary vector encoders would discard those heads.

### mrm8488/embeddinggemma-2-ft-typed-decision

[Model card](https://huggingface.co/mrm8488/embeddinggemma-2-ft-typed-decision) · revision `3d0069b4c56d7a7f505feb53bee90694b2c07845` · Apache-2.0 · about 0.3B parameters · 768 dimensions.

The official `GemmaDecisionModel` renders a readable flattened state, a question, and every option; it encodes these texts with EmbeddingGemma 2 and scores each option with a schema-conditioned head. The public `decide(state, questions, workflow="")` method returns typed Choice, Score, and Noul answers, calibrated probabilities, and the expected score level. JevEmbed's `gemma_typed_decision` backend calls that native API, retains the typed output, and checks probability normalization. The public model card reports 76.4% Typed Decisions test accuracy but says hyperparameters were inspected against the test split; treat the score as potentially optimistic and do not compare it to JevEmbed results. The card also says to use BF16 or FP32 and never FP16 because EmbeddingGemma 2 may produce NaNs.

### qnaug/embeddinggemma2-decision

[Model card](https://huggingface.co/qnaug/embeddinggemma2-decision) · revision `8cd99aef0e71b392fff66efa1d4d870f15ee0c78` · 768 dimensions · `head.pt` plus a local `decision_model.py` implementation.

This Clef-style adapter uses the official prefix `task: classification | query: `, then one sequence containing the state, question spans and option spans. It mean-pools the marked question and option spans and applies the checkpoint's `DecisionHead`; the `typed|choice`, `typed|score`, and `typed|noul` temperatures are preserved from `head.pt`. JevEmbed reimplements this small head locally and loads the checkpoint with `weights_only=True`, so no remote model code is executed. The card reports 74.3% Typed Decisions test accuracy and 0.029 ECE after calibration; these are upstream claims and need independent JevEmbed evaluation.

Both configurations support the existing JevEmbed request schema and return Choice, Score, and Noul. mrm8488's released renderer does not accept custom Noul descriptions; qnaug's implementation accepts them as candidate spans. The source and tests are in `src/jevembed/backends/embeddinggemma_decision.py` and `tests/unit/test_embeddinggemma_decision.py`. Install them in a separate environment with `pip install -e '.[gemma-decision,server]'` because the official mrm8488 config records Transformers 5.19.0. The A100 bundle helper downloads both pinned snapshots and creates local offline YAML files.

The upstream model reports 76.4% accuracy on its Typed Decisions test and qnaug reports 74.3% with 0.029 ECE on its test. These are separate upstream evaluations with different data and are not comparable to each other or to JevEmbed benchmark results. Recalibrate and evaluate each model on a held-out Jev set before relying on probabilities.

### Other candidates screened but not integrated

`SargeDev/Solar-JEV-Gemma-embed-2` (revision `d95e42f629f1b5ecc9dae23cc54b3ff95ff57926`, updated 2026-10-10) is a frozen EmbeddingGemma-2 encoder with three fixed linear heads. The official card describes it as an experiment, trained by distilling Solar Decide Flash distributions over a corpus partly derived from Jev-labeled states. Its input is only a task/question and state; Choice candidate descriptions are not encoded. It predicts a padded 16-slot choice distribution and a fixed six-level score distribution. That makes its labels position-dependent and unsuitable for JevEmbed's general candidate schema, so it is screened but not registered. Its published 55.79% overall argmax agreement is an upstream result, not a JevEmbed result.

`Ajeet-001/embeddinggemma2-decision-router` (revision `9809b3bc520b57f29e1c2e2df74f794708796337`, updated 2026-11-09) contains custom routing code, multimodal checkpoints and several export formats. The public repository inventory does not establish a compatible state/candidate scoring contract, so it needs source-level review and a reference test before integration. `convaiinnovations/laya-typed-decisions` is a 421M ModernBERT specialist for four English synthetic workflows and uses the separate `laya` runtime. `Danny-Dasilva/embeddinggemma-2-multimodal-typed-decision` has a richer custom `gemma_decision` package and vision/media heads. `sulabhkatiyar/embedding-gemma-multimodal-jev-indic` is a 746M multilingual and multimodal research checkpoint under mixed research terms and uses its own `egjev` package. These require runtime or schema adapters beyond the existing text-only integration.

The Nemotron configuration reuses JevEmbed's existing Sentence Transformers backend. It is a retrieval baseline, not a newly verified Jev-trained decision checkpoint. Its official Hugging Face metadata and revision were refreshed on 2026-11-21. Full checkpoint inference and A100 deployment have not been run.

| ID | Scope | Dimension | Limit | Remote code |
| --- | --- | ---: | ---: | --- |
| `nemotron-embed-1b-v2` | General multilingual retrieval; 1B baseline | 2048 | 8192 | true, pinned to model revision |

Nemotron uses `query:` and `passage:` prefixes; its query also includes the Jev question so instructions are preserved. The generic backend disables Sentence Transformers' stored default prompts with `prompt=""`, preserving the YAML templates. Native repository pooling is retained. Scoring constants are JevEmbed defaults and remain uncalibrated.

Install `.[local,server]` in a separate environment from `.[kanana,server]`. Use the supplied examples (each includes Choice, Score and Noul):

```bash
python -m jevembed --config configs/nemotron-embed-1b-v2.yaml --input examples/nemotron_decide.json --explain
```

For a faster download machine, `python scripts/prepare_jev_models.py download --weights-dir <directory> --model <model-id>` downloads native safetensors/tokenizer/config/code/license files at the configuration's pinned revision without loading their code, skipping ONNX and similar export copies. Repeat `--model` for the models to deploy; omitting it selects the five A100 deployment targets: CLM-Qwen3.5-2B, Jeva-4B, both EmbeddingGemma decision models, and Nemotron. Upload that directory to the A100, then run the helper's `local-configs` action with the same model selections and server directory to create `configs/local/*.local.yaml`. Local Jeva configurations retain Hub access when the native checkpoint references an external base encoder; other complete local layouts are offline. This checks file presence only; actual inference remains required.

Remove `--explain` for actual model inference. An explain result verifies request compilation only. `scripts/smoke_a100_models.py` runs the five A100 targets sequentially and saves outputs and logs; it does not include Kanana. Local Jeva configurations allow Hub access when the native checkpoint refers to an external backbone. Successful startup or `/v1/models` is not evidence of successful inference, because loading is lazy.

Nemotron is a general retrieval baseline rather than a newly Jev-trained model. Evaluate it on held-out Jev tasks before comparing accuracy or probabilities. It has NVIDIA and Llama license terms; preserve the downloaded LICENSE and model card when moving weights.

## Specialized decision candidates

The screenshot lists Jeva-4B, EmbeddingGemma 2 Typed Decision, EmbeddingGemma2 Decision Router, CLM-Qwen3.5-2B and MetaEncoder. Jeva-4B and CLM-Qwen3.5-2B were integrated before this update and are excluded from the new additions. The Typed Decision entry is now integrated as `mrm8488/embeddinggemma-2-ft-typed-decision`. The screenshot does not identify the router or MetaEncoder checkpoints by full repository ID. `Ajeet-001/embeddinggemma2-decision-router` is a related public router candidate, but its custom routing, multimodal files and export variants still need a state/candidate contract and reference inference before it can be mapped safely. MetaEncoder's reported 30B scale would need about 60 GB for BF16 parameters alone, exceeding an A100 40GB.

The two newly integrated models in this update are `mrm8488/embeddinggemma-2-ft-typed-decision` and `qnaug/embeddinggemma2-decision`. Their fixed revisions, native head behavior, runtime requirements and JevEmbed mappings are documented above. They were selected because they preserve arbitrary candidate descriptions and expose decision probabilities through a compact text-only path. Solar-JEV remains screened out because it predicts fixed output slots instead of scoring the submitted candidate text.

The official artifacts for Solar-JEV are available. Its candidate and score slots are fixed by output index, while JevEmbed accepts arbitrary option labels and descriptions. That semantic mismatch is why it remains excluded rather than being presented as a general decision model.

Before adding another specialist, verify the official prompt, state and option handling, head dimensions, output ordering, calibration, limits, runtime, and license. Add a dedicated backend only when the published interface can be mapped to JevEmbed without changing the meaning of candidates.
