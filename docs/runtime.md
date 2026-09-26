# Encoding, caching, and length limits

[Back to README](../README.md)

## Model encoding

The Sentence Transformers backend loads repository modules, preserving their attention and pooling. Those models use `include_prompt=true`; `encode(rendered_texts, prompt="")` prevents automatic prompts from replacing or duplicating instructions. The paired-projection backend encodes raw text with a base model and routes query and candidate vectors through separate checkpointed heads. Its prompts and pooling are configured per model.

## Device, precision, and length limits

`device: auto` prefers CUDA and otherwise uses CPU. `dtype: auto` selects bfloat16 on supported CUDA devices and float32 on CPU. E5 uses a 512-token limit, CLM uses 2048, and the other supported configurations use 32768. Overlong inputs fail by default. Explicit `overflow_policy: truncate` enables truncation, recorded in the trace, including cache hits. The paired-projection backend supports left or right truncation separately for queries and documents, as CLM's configuration uses. The sentence-transformers and HTTP backends accept only the default right-side setting; custom backends may implement either side.

## Caching

The in-memory LRU cache defaults to 4096 entries. Cache identity includes the model, revision, templates, encoding settings, and loaded model metadata. Cache access is locked per model. The built-in local backends serialize model encoding for thread safety; requests may overlap outside encoding. The built-in HTTP embedding backend permits concurrent encoding. Unknown custom backends retain whole-request serialization. Concurrent requests may encode the same uncached input more than once; usage reports each request's actual work. Create a new client after changing model weights or configuration.

## Token usage

Usage counts only inputs actually encoded after deduplication; cache hits add no tokens and `output_tokens=0`. Missing exact counts cause an error by default. Optional `usage_mode: estimate` uses `ceil(UTF-8 bytes / 4)` per text and marks estimates in logs and traces.

## HTTP serving limits

The optional HTTP server applies these limits per process to both `/v1/systemone` and, when enabled, `/debug/explain`:

| Limit | Default | CLI option |
| --- | ---: | --- |
| JSON request body | 2 MiB | `--max-request-bytes` |
| Questions per request | 64 | `--max-questions` |
| Compiled embedding inputs per request | 4096 | `--max-embedding-inputs` |
| Active inference requests | 4 | `--max-concurrent-requests` |
| Overall request body read timeout | 30 seconds | `--body-read-timeout-seconds` |

Body bytes are checked while streaming, including when `Content-Length` is missing or incorrect. The positive, finite body read timeout covers the whole upload, including pauses between chunks; an incomplete upload returns HTTP 408 and releases its request slot. It does not limit inference time. Embedding input counts include repeated inputs across questions, so they bound the work requested before cache lookup. Size and work violations return HTTP 413. When all inference slots are busy, the server returns HTTP 429 immediately with `Retry-After: 1`; it does not queue more work. Raising `--max-embedding-inputs` permits larger Choice sets without changing the Python API or its schema.

For an embedded FastAPI application, pass the same limits explicitly:

```python
from jevembed.server import HTTPServiceLimits, create_app

app = create_app(client, limits=HTTPServiceLimits(max_body_bytes=4 * 1024 * 1024,
                                                  max_embedding_inputs=8192,
                                                  max_concurrent_requests=8,
                                                  body_read_timeout_seconds=15))
```

Each worker process has its own limit and in-memory cache; multiply the concurrency limit by the number of workers when estimating total model load.
