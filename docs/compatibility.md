# Compatibility boundaries

JevEmbed v0.1 implements Jev-shaped JSON contracts based on the [API documentation](https://docs.typesafe.ai/api) and [Noul documentation](https://docs.typesafe.ai/primitives/noul). Noul criteria must be omitted or provide a complete true/false pair; empty, partial, and null criteria return HTTP 422. JevEmbed does not claim drop-in compatibility with the full official SDK or service.

| Area | JevEmbed v0.1 behavior |
| --- | --- |
| Input | State and instructions accept strings, objects, or arrays; nested JSON accepts finite numbers, booleans, and null |
| Top-level output | Only `model`, `answers`, and `usage`; diagnostics use a separate interface |
| Choice | One or more candidates, with no fixed count limit; null descriptions encode the name alone; ties follow request order. Large candidate sets increase inference work and memory use |
| Score | 2–10 levels; level numbers are excluded from encoded text; full distribution with string numeric keys |
| Structured Score legend | Original objects and arrays are retained; the official output table describes `map<string,string>`, so SDK consumers should treat this as a compatibility difference |
| Noul | Only `type` and `noul`; the default embedding configurations use retrieval queries, while CLM compares state/question text with true/false candidates |
| Model selection | Explicit registration and aliases; canonical response ID; no silent fallback |
| HTTP model field | Required; Python calls may use the default model |
| HTTP resource limits | Configurable body, question, embedding-input, and active-request caps; 413 for size/work violations and 429 when busy; Python calls are unaffected |
| Validation | Unknown fields, empty questions, invalid JSON, NaN/Infinity, and ambiguous criteria return 422 |
| Backend failures | HTTP 502; the entire request fails without partial answers |
| Confidence | Normalized entropy by default; CLM uses the top-probability margin; neither is prediction accuracy |
| Usage | Tokens actually encoded on this call; cache hits are excluded; `output_tokens=0`; not official billing usage |
| `GET /v1/models` | Registered IDs and aliases; official SDK metadata fields are not reproduced |
| HTTP embedding backend | Float vectors, index reordering, batching, per-call usage, and bounded retries; requires a compatible `/v1/embeddings` service |
| HTTP length enforcement | The deployment must reject overlong inputs; the generic adapter rejects truncation and server-owned templates |
| Local templates | Sentence Transformers models use repository modules with `include_prompt=true`; the paired-projection backend supports configurable text rendering and pooling |
| Revisions | Published traces record resolved revisions when available and file hashes; unknown upstream commits remain explicitly unknown |
| E5 | XLM-R, 1024-dimensional mean pooling; Instruct/Query queries and unprefixed documents; no remote code; 512-token limit |

The supported model configurations are listed in the [README](../README.md#model-list). NV-Embed-v2 is not supported.

## Model configuration and loading

Configurations are stored in `configs/<model-id>.yaml`. Repository IDs work as aliases, while responses use the canonical short ID. Unknown IDs, including `jev-latest`, are rejected.

The JevEmbed releases are fine-tuned for Choice, Score, and Noul and use 1,024-token truncation.

KaLM-embedding-multilingual-mini-instruct-v2.5 requires `trust_remote_code: true`. Qwen3-Embedding-0.6B, Qwen3-Embedding-4B, Qwen3-Embedding-8B, multilingual-e5-large-instruct, and CLM-v0.1-8B use `false`. Loading failures do not change trust or pooling settings. Pin revisions and dependency versions to reproduce results.

CLM-v0.1-8B uses separate state/action projections and [model-specific prompts](../configs/clm-v0.1-8b.yaml), with Choice/Score temperature 0.01 and Noul slope 100. See the [CLM-v0.1-8B guide](clm.md).

## Dependencies

| Dependency | Purpose |
| --- | --- |
| PyYAML | Load YAML model configurations |
| NumPy | Handle embedding arrays |
| PyTorch | Run CPU or CUDA inference |
| Transformers | Load models and tokenizers |
| sentence-transformers | Use repository-provided modules, encoding, and pooling |
| httpx | Call a compatible `/v1/embeddings` service |
| FastAPI | Expose the Jev-shaped HTTP API |
| Uvicorn | Run the HTTP server |

The tested stack uses PyTorch 2.8.0, Transformers 4.51.0, sentence-transformers 5.3.0, NumPy 1.26.4, and PyYAML 6.0.3. Transformers 4.51.0 is pinned for KaLM-embedding-multilingual-mini-instruct-v2.5 compatibility. Choose a PyTorch wheel for your hardware; CPU inference needs no FlashAttention.

Optional [LoRA training](training.md) supports the Sentence Transformers models on one or more GPUs; CLM's paired-projection backend is inference-only. Automatic prompt optimization and scalar calibration fitting are not implemented. The [JevEmbed-Data test results](../reports/JEVEMBED_DATA_TEST.md) compare the six base models and three fine-tuned models. Separate [KaLM](../reports/OPEN_JEV_KALM_LORA.md) and [Qwen3 0.6B](../reports/OPEN_JEV_QWEN3_0.6B_LORA.md) reports compare base and LoRA models on Open-Jev validation.

## Decision model runtimes

Use isolated environments for the following extras. Do not install their incompatible Transformers requirements together.

| Extra | Models | Runtime |
| --- | --- | --- |
| `clm,jeva` | CLM-Qwen3.5-2B and Jeva-4B | Python 3.12, Torch 2.9.1, Transformers 5.17.0 |
| `gemma-decision` | mrm8488 and qnaug EmbeddingGemma decision heads | Torch 2.10 or later, Transformers 5.19.0; BF16 or FP32 |
| `kanana` | Kanana-2-v-decide-3b-2611 | Torch 2.10.0, Transformers 5.5.3 |
| `local` with tested constraints | Original Sentence Transformers embedding configurations | Torch 2.8.0, Transformers 4.51.0, Sentence Transformers 5.3.0 |

Install the extras in their respective environments with `python -m pip install -e '.[clm,jeva,server]'`, `python -m pip install -e '.[gemma-decision,server]'`, or `python -m pip install -e '.[kanana,server]'`. Jeva's runtime is pinned to GitHub commit `e2104e0720100f979f2cdd456a1cbbee96da0abd`. The standard `local` extra uses a different Transformers version.

| Configuration | Decision behavior |
| --- | --- |
| [CLM-Qwen3.5-2B](../configs/clm-qwen3.5-2b.yaml) | Reuses the paired-projection backend with the checkpoint's chat template, assistant answer position, two normalized 512-dimensional heads, and native logit scale. Downloads the base Qwen3.5-2B encoder and the released `clm_heads.pt` separately |
| [Jeva-4B](../configs/jeva-4b.yaml) | Retains the LoRA adapter, state/option encoder, option memory, and set-aware decision head. Native probabilities map to Choice, expected Score, and the true probability for Noul; the upstream null candidate is disabled |
| [Kanana-2-v-decide-3b-2611](../configs/kanana-2-v-decide-3b-2611.yaml) | Uses the pinned native renderer, separate state/candidate embeddings, last-token pooling, normalization, and checkpoint decision temperature. Weights use the upstream [KananaOpenLicense](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611/blob/main/LICENSE) |
| [EmbeddingGemma Typed Decision](../configs/embeddinggemma-2-ft-typed-decision.yaml) | Calls the checkpoint's `decide` API, which separately pools state, question, and options and applies a schema-conditioned head. Custom Noul criteria are rejected |
| [EmbeddingGemma2 Decision](../configs/qnaug-embeddinggemma2-decision.yaml) | Pools question and option spans from one sequence, then applies the local implementation of the released head and its typed temperatures. Loads `head.pt` with `weights_only=True` |

Kanana and mrm8488 use remote modeling code pinned to the same revision as their weights. Jeva uses the pinned installed runtime; qnaug's head is implemented locally without remote code. These integrations accept text/JSON; checkpoint image or multimedia support is not exposed by the JevEmbed schema. Use BF16 or FP32 for the EmbeddingGemma models; FP16 is rejected because it can produce NaNs. Native heads own their probability and cache behavior, as described in [runtime behavior](runtime.md#native-decision-backends).

The configurations pin the released checkpoints. In particular, CLM uses the release heads, rather than the separately published Typed Decisions specialist. Upstream benchmark scores do not establish accuracy on JevEmbed tasks. Unit tests and `--explain` verify adapter behavior and request compilation; full-checkpoint inference and accuracy evaluation are separate checks.

### Use the shared task examples

Choose a configuration from the table and set the request's model ID, as with the existing models:

```python
import json
from pathlib import Path
from jevembed import JevEmbed, ModelConfig

config = ModelConfig.load("configs/qnaug-embeddinggemma2-decision.yaml")
request = json.loads(Path("examples/official_choice_exchange.json").read_text(encoding="utf-8"))
request["model"] = config.model_id
client = JevEmbed(config=config)
print(client.evaluate(request))
```

The same pattern works with `official_score_safari.json` and `official_noul_escalation.json`. For mrm8488, submit only `is_human_escalation` from the Noul example; `is_repeat_contact` has custom criteria that its native API does not support. Keep those criteria for models that support them.

For CLI use, save the modified request under the existing ignored `artifacts/` directory, then run `python -m jevembed --config configs/qnaug-embeddinggemma2-decision.yaml --input artifacts/request.json`. Replace the configuration and request model ID together. Local weight paths belong in an ignored YAML copy; see [offline loading](runtime.md#local-weights-and-offline-use).
