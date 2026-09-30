# Python API

[Back to README](../README.md#quick-start)

## Evaluate a request

Load one model configuration and evaluate a Jev request:

```python
import json
from pathlib import Path
from jevembed import JevEmbed, ModelConfig

config = ModelConfig.load("configs/kalm-embedding-v2.5.yaml")
client = JevEmbed(config=config)
request = json.loads(Path("examples/official_noul_escalation.json").read_text(encoding="utf-8"))
response = client.evaluate(request)
```

`evaluate()` returns a plain dictionary containing the model ID, answers, and token usage.

## Register multiple models

Register each configuration, then select a model with the request's `model` field:

```python
client = JevEmbed()
client.register(ModelConfig.load("configs/kalm-embedding-v2.5.yaml"))
client.register(ModelConfig.load("configs/qwen3-embedding-0.6b.yaml"))

request["model"] = "qwen3-embedding-0.6b"
response = client.evaluate(request)
```

Python calls may omit `model` and use the first registered model. HTTP requests must specify a registered model ID or alias.

## Build a request in Python

Use `system_one()` when the state and questions are already Python values:

```python
response = client.system_one(
    state="hello",
    questions={
        "greeting": {
            "type": "choice",
            "instructions": "Classify the message.",
            "criteria": {"greeting": "A greeting", "other": "Other content"},
        }
    },
    model="kalm-embedding-v2.5",
)
```

## Inspect inputs and traces

`explain()` compiles and renders inputs without loading model weights. `evaluate_with_trace()` runs inference and returns both the standard response and diagnostics:

```python
compiled = client.explain(request)
result = client.evaluate_with_trace(request)
response = result["response"]
trace = result["trace"]
```

Traces include rendered inputs, similarities, cache activity, truncation records, backend metadata, token usage sources, and elapsed time. Keep traces outside standard Jev responses.

## Cache and custom backends

Clear every in-memory embedding cache, or only one model's cache:

```python
client.clear_cache()
client.clear_cache("kalm-embedding-v2.5")
```

Custom backends implement `encode(list[EmbeddingInput]) -> EmbeddingBatch` and can be injected with `JevEmbed(backend=backend, model="my-model")`. Pass `confidence_estimator` to provide a custom confidence function. See the [backend contract](design.md#backend-contract) for the complete interface and the [runtime guide](runtime.md) for caching, token usage, and local weights.
