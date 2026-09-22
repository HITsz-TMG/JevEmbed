# Encoding, caching, and length limits

[Back to README](../README.md)

## Model encoding

The local backend loads repository modules through SentenceTransformer, preserving attention and pooling. Supported models use `include_prompt=true`; `encode(rendered_texts, prompt="")` prevents automatic prompts from replacing or duplicating instructions. Other pooling conventions or structured model inputs require a dedicated adapter.

## Device, precision, and length limits

`device: auto` prefers CUDA and otherwise uses CPU. `dtype: auto` selects bfloat16 on supported CUDA devices and float32 on CPU. E5's 512-token limit includes instructions, content, and special tokens; the other supported configurations use 32768. Overlong inputs fail by default. Explicit `overflow_policy: truncate` enables truncation, recorded in the trace, including cache hits.

## Caching

The in-memory LRU cache defaults to 4096 entries. Cache identity includes the model, revision, templates, encoding settings, and loaded model metadata. A per-model lock protects encoding, cache access, and usage accounting. Create a new client after changing model weights or configuration.

## Token usage

Usage counts only inputs actually encoded after deduplication; cache hits add no tokens and `output_tokens=0`. Missing exact counts cause an error by default. Optional `usage_mode: estimate` uses `ceil(UTF-8 bytes / 4)` per text and marks estimates in logs and traces.
