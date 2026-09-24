# Request examples

These examples are organized by task and shared by all supported models.

| File | Scenario | Paths covered |
| --- | --- | --- |
| `official_choice_exchange.json` | Route a shoe exchange request | Choice |
| `official_score_safari.json` | Assess a Safari bug's severity | Score |
| `official_noul_escalation.json` | Detect human escalation and repeat contact | Noul with and without criteria |

The JSON files default to `model: kalm-embedding-v2.5`. To switch models, change only `model` and load the corresponding configuration. Keep state, instructions, and criteria unchanged.

The following example predictions use the shipped configurations, CUDA BF16, and disabled caching. Values are rounded to six decimals. They are uncalibrated predictions, not accuracy measurements.

| Model | Choice: `returns` probability | Score: expected severity | Noul: human escalation | Noul: repeat contact |
| --- | ---: | ---: | ---: | ---: |
| KaLM v2.5 | 0.792899 | 1.257043 | 0.999816 | 0.521023 |
| Qwen3 0.6B | 0.542119 | 1.090709 | 0.999773 | 0.738244 |
| Qwen3 4B | 0.562020 | 0.979632 | 0.999852 | 0.699510 |
| Qwen3 8B | 0.460480 | 1.226909 | 0.999863 | 0.680314 |
| E5 large instruct | 0.439349 | 1.096561 | 0.999778 | 0.627429 |

All five models select `returns` for Choice. The full KaLM CPU FP32 responses and Jev reference outputs appear in the [main README](../README.md#official-jev-examples-with-kalm-results). Small differences are possible with another device, precision, or model revision.

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
