# Architecture and design

Version: 0.1. Updated: 2026-09-24.

This document describes JevEmbed's architecture, public configurations, and compatibility boundaries.

## Purpose and scope

JevEmbed converts Choice, Score, and Noul requests into embedding comparisons using interchangeable instruction embedding models. It preserves original instructions and returns Jev-shaped JSON fields. Compatibility covers data structures and field constraints, not Jev's model architecture, training, decision quality, confidence formula, or calibration.

JevEmbed v0.1 includes a Python API, local and HTTP embedding backends, configurable templates, batching, caching, scoring parameters, diagnostics, a CLI, an optional HTTP API, and [LoRA fine-tuning](training.md) for supervised Jev tasks. Automatic prompt optimization, generated criteria, agent execution, authorization decisions, and a frontend are outside its scope. Calibration fitting and broader independent evaluations remain future work.

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
| Choice | Nonempty mapping of names to strings, objects, arrays, or null; no fixed candidate count limit |
| Score | Ordered array of 2–10 strings, objects, or arrays |
| Noul | Omitted, or a complete mapping with `true` and `false` descriptions |

Empty, partial, and null Noul criteria are rejected. When criteria are omitted, the default embedding configurations compare the state and question as separate queries; CLM compares the state/question input with true/false candidates. A single Choice candidate has probability and confidence equal to 1.

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

Strings are preserved verbatim. The default configurations serialize structured values as deterministic compact JSON with sorted object keys and preserved Unicode. CLM uses prose serialization configured in its YAML file. Field references such as `ticket.message` remain literal instruction text; the compiler does not extract fields automatically.

The default Sentence Transformers templates are:

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

Encode the original question and state together as one query under the fixed retrieval instruction:

```text
Instruct: Retrieve semantically similar text.
Query: Has the customer contacted support about this before?
I have asked three times now. Can I please just talk to a real person?
```

Encode each criterion as a separate query, prefixed with its true/false key and ordered true then false regardless of JSON key order:

```text
Instruct: Retrieve semantically similar text.
Query: true: Mentions a prior attempt, ticket, or that they have asked before
```

```text
Instruct: Retrieve semantically similar text.
Query: false: No sign of any previous contact
```

### Noul without criteria

Encode the original question and state separately as queries under the same fixed retrieval instruction:

```text
Instruct: Retrieve semantically similar text.
Query: Is the customer asking for a human agent?
```

```text
Instruct: Retrieve semantically similar text.
Query: I have asked three times now. Can I please just talk to a real person?
```

This default path uses one cosine similarity and generates no true/false candidates or unknown category. The retrieval mapping does not inherently distinguish semantic relevance from agreement. CLM instead uses two candidates for both Noul forms; see the [CLM guide](clm.md).

## Scoring

Normalize vectors with L2 normalization before computing cosine. Reject zero, empty, missing, non-finite, or dimensionally inconsistent vectors.

```text
s_i = cosine(query, document_i)
p_i = stable_softmax(s_i / temperature)
choice = argmax(p_i)
score = sum(i * p_i)
confidence = 1 - entropy(p) / log(K)  # default
```

Temperatures are finite and positive, with independent Choice and Score defaults of 0.1; CLM configures 0.01. Ties in Choice follow request order. Confidence is clamped to [0,1] and is 1 for K=1. CLM selects the top-probability margin instead of entropy. Both measure concentration, not accuracy; callers can inject a custom estimator.

```text
Noul with criteria:
  sigmoid(a_with * (cosine(q_question_state, q_true_criterion) - cosine(q_question_state, q_false_criterion)) + b_with)
Noul without criteria:
  sigmoid(a_without * cosine(q_question, q_state) + b_without)
```

Both Noul parameter pairs default to slope 10 and intercept 0. CLM uses a true-minus-false difference for both Noul forms and configures slope 100. All mappings are uncalibrated. A value in [0,1] is not evidence of calibration. See [calibration](calibration.md) for configuration, fitting records, and evaluation requirements.

## Backend contract

```python
class EmbeddingBackend(Protocol):
    def encode(self, items: list[EmbeddingInput]) -> EmbeddingBatch: ...
```

An input carries `role`, `instruction`, and `text`. A batch carries vectors, token usage, and usage provenance. Model-specific rendering belongs in the adapter; the compiler and scorer remain independent of model libraries.

The Sentence Transformers backend loads repository-provided modules, preserving KaLM and Qwen3-Embedding pooling. The paired-projection backend loads a base encoder and a two-head checkpoint, with pooling and prompts selected by configuration. Both accept local directories or repository IDs, device, precision, and explicit remote-code trust. Loading is lazy; importing the core and compiling requests require no weights or GPU.

The HTTP backend connects to a compatible `/v1/embeddings` endpoint with a configured model, base URL, API-key environment variable, timeout, batch size, and retry limit. The client owns template rendering. The service must reject overlong input before the deployment declares `server_enforces_length: true`; the generic adapter does not implement client-side truncation or server-owned templates. Response indices restore vector order, and missing or duplicate indices are errors. Usage belongs to each call and is never stored as shared mutable `last_usage` state.

Custom Python backends can integrate other runtimes or fixed vectors for tests.

## Models, prompts, and loading

The delivered configurations are KaLM v2.5, Qwen3-Embedding 0.6B/4B/8B, multilingual E5 large instruct, and CLM v0.1-8B. Their repository IDs, dimensions, pooling, and limits are listed in the [README](../README.md#supported-models). Changing the model does not require rewriting business questions or criteria.

KaLM uses repository-defined bidirectional attention and mean pooling. Qwen3-Embedding uses last-token pooling. E5 uses XLM-R and mean pooling. CLM uses Qwen3-8B last-token pooling followed by 512-dimensional state/action projections. E5 accepts 512 tokens, CLM 2048, and the remaining public configurations use a 32768-token limit. Adapter validation checks dimensions and loaded metadata.

The Sentence Transformers models use `include_prompt=true`. That backend renders complete strings and calls `encode(..., prompt="")` so default retrieval prompts cannot replace or duplicate original instructions. CLM uses its configured raw-text encoder and projection heads.

Remote code is disabled by default and explicitly enabled in the KaLM configuration. A load failure must not enable trust automatically. Remote-code trust does not install missing dependencies or resolve library incompatibilities. The tested library versions are recorded in the dependency constraints. Pin model and code revisions when supported; a weight revision alone must not be described as pinning independently referenced code. Unknown revisions remain unknown in diagnostics, with file hashes provided when available.

Public configurations use repository IDs. Local paths belong in ignored user configuration files. `device: auto`, `dtype: auto`, and `model_default` are framework settings resolved by the adapter. CPU uses float32; supported CUDA devices can use bfloat16. FlashAttention is not required. Failed loading must not silently change attention or pooling semantics.

## Length, caching, and usage

The default length policy rejects overlong inputs. Explicit truncation is recorded with original and encoded token counts, including when a result comes from cache. Silent loss of business rules is not allowed.

The bounded in-memory LRU cache defaults to 4096 entries and exposes a clear operation. Identity incorporates model and revision, role, templates, input text, truncation and encoding settings, and loaded metadata. A per-model lock protects encoding, cache access, and token accounting. Create a new client after changing model weights or configuration.

Usage counts only deduplicated inputs actually encoded. Cache hits are excluded. Counts can differ from Jev billing because templates, repeated context, deduplication, and caching differ. Strict mode requires exact backend/tokenizer counts. Optional estimate mode uses `ceil(UTF-8 bytes / 4)` per text and labels estimates in logs and traces.

## Interfaces and errors

The Python API supports registration, evaluation, compilation inspection, traced evaluation, and cache clearing. The CLI accepts JSON input, repeated model configurations, output files, `--explain`, `--trace`, and `--serve`. Scoring and backend parameters belong in model configuration, not extra Jev question fields.

The HTTP API provides `POST /v1/systemone` and `GET /v1/models`, binding to loopback by default. An explicitly enabled debug route exposes compilation. Validation failures return 422 and backend failures return 502. Requests fail atomically without partial answers. Retries are bounded and apply only to transient failures. See [compatibility](compatibility.md) for SDK boundaries.

## Source layout

| Component | Responsibility |
| --- | --- |
| `schemas.py`, `serialization.py` | Request validation and configurable text serialization |
| `compiler.py`, `prompts.py` | Task plans and rendered model inputs |
| `config.py`, `client.py` | Configuration, registration, evaluation, and usage |
| `backends/` | Local, HTTP, and custom backend contracts |
| `scoring.py`, `cache.py` | Probability mappings, confidence, and bounded caching |
| `server.py`, `cli.py` | HTTP and command-line interfaces |
| `configs/`, `examples/` | Portable configurations and shared task examples |
| `tests/`, `reports/` | Regression tests and published test-set results |

Core dependencies remain lightweight. Local inference, HTTP clients, and servers have separate optional dependency groups.

## Examples and test results

The [shared examples](../examples/README.md) show Choice, Score, and both Noul forms. The [JevEmbed-Data test evaluation](../reports/JEVEMBED_DATA_TEST.md) reports model accuracy and error metrics with their denominators.
