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

The supported model configurations are listed in the [README](../README.md#supported-models). NV-Embed-v2 is not supported.

Optional [LoRA training](training.md) supports the Sentence Transformers models on one device; CLM's paired-projection backend is inference-only. Automatic prompt optimization and scalar calibration fitting are not implemented. The [JevBench evaluation](../reports/JEVBENCH_PUBLIC.md) reports public-subset accuracy, distinguishing E5's default length-limit refusals from explicit truncation. Separate [KaLM](../reports/OPEN_JEV_KALM_LORA.md) and [Qwen3 0.6B](../reports/OPEN_JEV_QWEN3_0.6B_LORA.md) reports compare base and LoRA models. None of these results covers the full benchmark suite.
