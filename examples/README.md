# Request examples

These examples are organized by task and shared by all supported models.

| File | Scenario | Paths covered |
| --- | --- | --- |
| `official_choice_exchange.json` | Route a shoe exchange request | Choice |
| `official_score_safari.json` | Assess a Safari bug's severity | Score |
| `official_noul_escalation.json` | Detect human escalation and repeat contact | Noul with and without criteria |

The JSON files default to `model: kalm-embedding-v2.5`. To switch models, change only `model` and load the corresponding configuration. Keep state, instructions, and criteria unchanged.

Run this example from the project root:

```python
import json
from jevembed import JevEmbed, ModelConfig

config = ModelConfig.load("configs/qwen3-embedding-4b.yaml")
with open("examples/official_noul_escalation.json", encoding="utf-8") as handle:
    request = json.load(handle)
request["model"] = config.model_id

client = JevEmbed(config=config)
print(client.explain(request))  # Inspect compiled inputs without loading weights
print(client.evaluate(request))
```

Available configurations are listed under [supported models](../README.md#supported-models). The CLI uses the same request fields. Save modified requests under the ignored `artifacts/` directory.

Original requests and reference responses live in `tests/fixtures/`. The same examples apply to all supported models.
