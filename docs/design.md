# Architecture and design

Version: 0.1. Updated: 2026-09-22.

This document consolidates the original implementation specification into an English developer reference. The implementation, public configurations, and compatibility report describe the delivered MVP; future work is identified separately.

## Purpose and scope

JevEmbed converts Choice, Score, and Noul requests into embedding comparisons using interchangeable instruction embedding models. It preserves original instructions and returns Jev-shaped JSON fields. Compatibility covers data structures and field constraints, not Jev's model architecture, training, decision quality, confidence formula, or calibration.

The MVP includes a Python API, local and HTTP embedding backends, configurable templates, batching, caching, scoring parameters, diagnostics, a CLI, an optional HTTP API, tests, and documentation. Model training, automatic prompt optimization, generated criteria, agent execution, authorization decisions, and a frontend are outside its scope. Calibration fitting and larger labeled evaluations are future work.

## Request contract

```json
{
  "model": "kalm-embedding-v2.5",
  "state": "Content to evaluate",
  "questions": {
    "question_id": {
      "type": "choice",
      "instructions": "Which option fits?",
      "criteria": {"a": "Description A", "b": "Description B"}
    }
  }
}
```

State and instructions accept strings, JSON objects, or arrays. Nested JSON may contain finite numbers, booleans, and null. Unknown fields, non-JSON values, NaN, and Infinity are rejected. Questions must be a nonempty mapping and may mix primitives. Question IDs associate answers with requests and never enter model inputs.

The model must be registered. HTTP requires an explicit model; Python calls may use the client's default. Aliases must be registered explicitly. Responses use the canonical model ID, and unknown models never fall back silently.

| Primitive | Criteria |
| --- | --- |
| Choice | Mapping of 1–255 names to strings, objects, arrays, or null |
| Score | Ordered array of 2–10 strings, objects, or arrays |
| Noul | Omitted, or a complete mapping with `true` and `false` descriptions |

Empty, partial, and null Noul criteria are rejected. No missing counterexample is generated. A single Choice candidate has probability and confidence equal to 1.

## Response contract

The standard response has exactly `model`, `answers`, and `usage` at the top level. Each answer contains these fields:

| Primitive | Answer fields |
| --- | --- |
| Choice | `type`, `choice`, `probabilities`, `confidence` |
| Score | `type`, `score`, `legend`, `probabilities`, `confidence` |
| Noul | `type`, `noul` |

Choice retains every original candidate name. Score uses string keys `"0"`, `"1"`, and so on for the complete legend and probability distribution. Structured legend values retain their original types. Noul adds no confidence, unknown class, or probability map.

Usage contains `input_tokens` and `output_tokens`; the latter is zero because no generative decoding occurs. Similarities, calibration metadata, and debug information are available through `explain()` and `evaluate_with_trace()`, rather than additional standard response fields. Evaluation returns ordinary Python dictionaries.

## Compilation and serialization

Strings are preserved verbatim. Structured values use deterministic compact JSON with sorted object keys, preserved Unicode, and unchanged array order. Field references such as `ticket.message` remain literal instruction text; the compiler does not extract fields automatically.

The default templates are:

```text
Query:    Instruct: {instruction}\nQuery: {text}
Document: {text}
```

Each question becomes an independent task plan. The compiler collects embedding inputs, deduplicates them, and maps returned vectors back to tasks. An unrelated question must not change an existing question's candidates or normalization range. Numerical comparisons across different batches allow a documented floating-point tolerance.

### Choice

Encode the original instructions and state as a query. Each document is `name: description`; a null description produces the name alone. Do not append the full candidate list to the query.

For the exchange fixture, the query is:

```text
Instruct: Which team should handle this?
Query: My running shoes arrived in the wrong size. Can I swap them for a size 10?
```

The documents are encoded separately:

```text
returns: Exchanges, wrong or damaged items
shipping: Delivery status, delays, lost packages
billing: Charges, invoices, payment problems
```

### Score

Encode the original instructions and state as a query. Encode each level description separately, without adding its array index. The index is used only in the output and expectation calculation.

For the Safari fixture, the query is:

```text
Instruct: How severe is the reported issue?
Query: The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari.
```

The documents are:

```text
Cosmetic; no impact to functionality
Broken or degraded feature, but workaround exists
Blocking issue; no workaround exists
```

The complete distribution is retained because different distributions can have the same expected score. There is no ordinal smoothing, enforced unimodality, or score rounding.

### Noul with criteria

Encode the original instructions and state as a query. Encode the true and false descriptions as separate documents in that fixed order, regardless of the incoming JSON key order. The keys themselves are not prepended to the descriptions.

```text
Instruct: Has the customer contacted support about this before?
Query: I have asked three times now. Can I please just talk to a real person?
```

Documents:

```text
Mentions a prior attempt, ticket, or that they have asked before
No sign of any previous contact
```

### Noul without criteria

Both inputs use the query role and the fixed instruction `Retrieve semantically similar text.`. The original state and original instructions become the two query texts:

```text
Instruct: Retrieve semantically similar text.
Query: I have asked three times now. Can I please just talk to a real person?
```

```text
Instruct: Retrieve semantically similar text.
Query: Is the customer asking for a human agent?
```

The second input does not use the document template. The same state embedding can be reused across multiple questions on this path. No true/false propositions or unknown category are generated. This path measures semantic similarity and cannot inherently distinguish negation, contradiction, and support.

## Scoring

Normalize vectors with L2 normalization before computing cosine. Reject zero, empty, missing, non-finite, or dimensionally inconsistent vectors.

```text
s_i = cosine(query, document_i)
p_i = stable_softmax(s_i / temperature)
choice = argmax(p_i)
score = sum(i * p_i)
confidence = 1 - entropy(p) / log(K)
```

Temperatures are finite and positive, with independent Choice and Score defaults of 0.5. Ties in Choice follow request order. Confidence is clamped to [0,1] and is 1 for K=1. It measures concentration, not accuracy; callers can inject a custom estimator.

```text
Noul with criteria:
  sigmoid(a_criteria * (cosine(q, true) - cosine(q, false)) + b_criteria)
Noul without criteria:
  sigmoid(a_similarity * cosine(query_state, query_instruction) + b_similarity)
```

Both Noul parameter pairs default to slope 1 and intercept 0. All default mappings are uncalibrated. A value in [0,1] is not evidence of calibration. See [calibration](calibration.md) for configuration, fitting records, and evaluation requirements.

## Backend contract

```python
class EmbeddingBackend(Protocol):
    def encode(self, items: list[EmbeddingInput]) -> EmbeddingBatch: ...
```

An input carries `role`, `instruction`, and `text`. A batch carries vectors, token usage, and usage provenance. Model-specific rendering belongs in the adapter; the compiler and scorer remain independent of model libraries.

The local backend uses `sentence_transformers.SentenceTransformer` to load repository-provided modules. It does not replace KaLM or Qwen3 pooling with a separate hand-written AutoModel pipeline. It accepts a local model directory or repository ID, revision, device, batch size, supported precision, and explicit remote-code trust. Loading is lazy; importing the core and compiling requests require no weights or GPU.

The HTTP backend connects to a compatible `/v1/embeddings` endpoint with a configured model, base URL, API-key environment variable, timeout, batch size, and retry limit. The client owns template rendering. The service must reject overlong input before the deployment declares `server_enforces_length: true`; the generic adapter does not implement client-side truncation or server-owned templates. Response indices restore vector order, and missing or duplicate indices are errors. Usage belongs to each call and is never stored as shared mutable `last_usage` state.

Custom Python backends can integrate other runtimes or fixed vectors for tests.

## Models, prompts, and loading

The four delivered configurations are KaLM v2.5, Qwen3 0.6B, Qwen3 4B, and multilingual E5 large instruct. Their repository IDs, dimensions, pooling, and limits are listed in the [README](../README.md#supported-models). All use the same four compilation paths. Changing the model does not require rewriting business questions or criteria.

KaLM uses repository-defined bidirectional attention and mean pooling. Qwen3 uses last-token pooling. E5 uses XLM-R and mean pooling. E5 accepts 512 tokens including prompts and special tokens; the other public configurations use a 32768-token limit. Adapter validation checks dimensions and loaded metadata.

Supported models use `include_prompt=true`. The backend renders complete strings and calls `encode(..., prompt="")` so default retrieval prompts cannot replace or duplicate original instructions. A model with excluded prompt tokens or structured instruction/text inputs requires a dedicated adapter. Text equality alone does not establish pooling equivalence.

Remote code is disabled by default and explicitly enabled in the KaLM configuration. A load failure must not enable trust automatically. Remote-code trust does not install missing dependencies or resolve library incompatibilities. The tested library versions are recorded in the dependency constraints. Pin model and code revisions when supported; a weight revision alone must not be described as pinning independently referenced code. Unknown revisions remain unknown in diagnostics, with file hashes provided when available.

Public configurations use repository IDs. Local paths belong in ignored user configuration files. `device: auto`, `dtype: auto`, and `model_default` are framework settings resolved by the adapter. CPU uses float32; supported CUDA devices can use bfloat16. FlashAttention is not required. Failed loading must not silently change attention or pooling semantics.

## Length, caching, and usage

The default length policy rejects overlong inputs. Explicit truncation is recorded with original and encoded token counts, including when a result comes from cache. Silent loss of business rules is not allowed.

The bounded in-memory LRU cache defaults to 4096 entries and exposes a clear operation. Identity incorporates model and revision, role, templates, input text, truncation and encoding settings, and loaded metadata. A per-model lock protects encoding, cache access, and token accounting. Create a new client after changing model weights or configuration.

Usage counts only deduplicated inputs actually encoded. Cache hits are excluded. Counts can differ from Jev billing because templates, repeated context, deduplication, and caching differ. Strict mode requires exact backend/tokenizer counts. Optional estimate mode uses `ceil(UTF-8 bytes / 4)` per text and labels estimates in logs and traces.

## Interfaces and errors

The Python API supports registration, evaluation, compilation inspection, traced evaluation, and cache clearing. The CLI accepts JSON input, repeated model configurations, output files, `--explain`, `--trace`, and `--serve`. Scoring and backend parameters belong in model configuration, not extra Jev question fields.

The HTTP API provides `POST /v1/systemone` and `GET /v1/models`, binding to loopback by default. An explicitly enabled debug route exposes compilation. Validation failures return 422 and backend failures return 502. Requests fail atomically without partial answers. Retries are bounded and apply only to transient failures. See [compatibility](compatibility.md) for unverified SDK boundaries.

## Source layout

| Component | Responsibility |
| --- | --- |
| `schemas.py`, `serialization.py` | Request validation and deterministic JSON |
| `compiler.py`, `prompts.py` | Task plans and rendered model inputs |
| `config.py`, `client.py` | Configuration, registration, evaluation, and usage |
| `backends/` | Local, HTTP, and custom backend contracts |
| `scoring.py`, `cache.py` | Probability mappings, confidence, and bounded caching |
| `server.py`, `cli.py` | HTTP and command-line interfaces |
| `configs/`, `examples/` | Portable configurations and shared task examples |
| `tests/`, `reports/` | Fixtures, automated checks, and published validation results |

Core dependencies remain lightweight. Local inference, HTTP clients, and servers have separate optional dependency groups.

## Validation contract

Validation separates contract correctness, model integration, and semantic observations. Fixed vectors can verify formulas and fields but cannot establish model decision quality.

Automated checks cover exact input mappings; both Noul paths; unchanged instructions and question-ID independence; null and structured criteria; candidate bounds and ties; Score indices and legends; mixed-question independence; stable probability formulas; finite parameters and vectors; Unicode; length policies; model aliases; cache isolation; concurrency and usage; HTTP response indices, retries, and errors; matching Python and HTTP fields; lazy imports; and CLI operation.

Real-model validation records revisions or explicit unknown status, artifact hashes, templates, scoring settings, raw cosine values, outputs, device, precision, dependencies, timing, and usage provenance. Single-question and mixed-question Noul runs are compared with an explicit tolerance. Length validation must prove acceptance at the limit, rejection above it, and reported truncation when enabled.

### Official fixtures

The original requests and reference responses are retained separately under `tests/fixtures/`:

| Fixture stem | Source | Required observations |
| --- | --- | --- |
| `official_choice_exchange` | [Choice](https://docs.typesafe.ai/primitives/choice) | Exact named documents; all candidate probabilities; selected argmax; record whether the model selects returns |
| `official_score_safari` | [Score](https://docs.typesafe.ai/primitives/score) | Unnumbered documents; complete string-keyed legend; expected score; record the ranking of levels 1 and 2 relative to 0 |
| `official_noul_escalation` | [Noul](https://docs.typesafe.ai/primitives/noul) | Both paths in one request; finite Noul values; compare individual and mixed execution |

Only the request model changes during JevEmbed evaluation. Original reference JSON remains unchanged. The Score reference satisfies `0 * 0.0 + 1 * 0.57 + 2 * 0.43 = 1.43`; JevEmbed is not required to reproduce that value or official probabilities, confidence, or usage.

These three requests cover four questions and four compilation paths for each model. They are smoke tests, not an accuracy benchmark. Calibration fitting and generalization evaluation require separate labeled data.

See the [JevBench public evaluation](../reports/JEVBENCH_PUBLIC.md) for measured accuracy, calibration, latency, and limitations. Future work includes calibration fitting, reliability plots, larger labeled evaluations, and additional dedicated adapters.
