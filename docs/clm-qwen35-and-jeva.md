# CLM Qwen3.5 and Jeva decision adapters

These two configurations retain released decision heads. They accept the existing JevEmbed Choice, Score, and Noul request schema through the CLI, Python API, and HTTP service. Their upstream metrics are not JevEmbed evaluation results.

## Runtime

Use a dedicated Python 3.12 environment:

```bash
python -m pip install -e '.[clm,jeva,server]'
```

Both use Torch 2.9.1, Transformers 5.17.0, PEFT 0.21.2, and Accelerate 1.12.0. Jeva's runtime is pinned to GitHub commit `e2104e0720100f979f2cdd456a1cbbee96da0abd`. Do not combine this extra with the standard `local` stack or the Kanana or Gemma extras in the same environment. The core installation does not add optional Qwen3.5 convolution or linear-attention kernels; reference inference is slower than accelerated upstream measurements.

## CLM Qwen3.5 2B

[`clm-qwen3.5-2b.yaml`](../configs/clm-qwen3.5-2b.yaml) combines the pinned `Qwen/Qwen3.5-2B` encoder with `jayavibhav/clm-qwen3.5-2b` projection heads. It reuses `PairedProjectionBackend`. The adapter retains the checkpoint's chat template, ordered option rows, assistant answer position, two 512-dimensional projection heads, normalization, and checkpoint logit scale. Noul uses false and true candidate embeddings with the corresponding difference in logits.

The public YAML uses the upstream 131,072-token limit, BF16, and truncation. That is an input bound, not a claim that long requests fit any GPU. Start with batch size 1 and a lower token limit in a copied local configuration on A100 40GB; record truncation and configuration changes when evaluating results.

```bash
python -m jevembed --config configs/clm-qwen3.5-2b.yaml \
  --input examples/clm_qwen35_decide.json --trace
```

## Jeva 4B

[`jeva-4b.yaml`](../configs/jeva-4b.yaml) loads `LIQIIIII/Jeva-4B` through the installed native Jeva runtime. It preserves state/option encoding, the released LoRA adapter, option memory, and the set-aware matrix decision head. The backend maps native candidate probabilities to the original labels, computes Score as the expected level, and uses the true candidate probability for Noul. It disables the upstream null candidate to preserve JevEmbed's submitted candidate schema. The resulting probabilities pass shape, range, and normalization checks before they are returned.

Unlike a fixed cosine-only model, Jeva's decision head depends on the candidate set. Candidate representations and memory may be reused by the native runtime; complete decision scores cannot be assumed independent of the other candidates.

```bash
python -m jevembed --config configs/jeva-4b.yaml \
  --input examples/jeva_decide.json --trace
```

Keep `config.yaml`, `head.pt`, tokenizer files, the base encoder, and `backbone_adapter/` together when supplied by the checkpoint. Some checkpoint layouts reference an external Qwen3.5 base model. The download helper checks whether that base is present locally: fully local layouts get `local_files_only=true`, and external layouts retain network access. The first inference then needs Hub access or an already populated cache for the base. A downloaded LoRA adapter alone does not provide offline inference.

## Validation

`--explain` compiles requests without weights. Unit tests cover request rendering, typed output mapping, and cache clearing. An optional local projection-head check uses `JEVEMBED_CLM_QWEN35_HEAD`; it skips when the checkpoint is absent. Real A100 inference and accuracy evaluation remain separate checks.

Official sources: [CLM model](https://huggingface.co/jayavibhav/clm-qwen3.5-2b), [Jeva model](https://huggingface.co/LIQIIIII/Jeva-4B), [Jeva runtime](https://github.com/LiQiiiii/Jeva).
