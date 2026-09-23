# Request examples

These examples are organized by task and shared by all supported models.

| File | Scenario | Paths covered |
| --- | --- | --- |
| `official_choice_exchange.json` | Route a shoe exchange request | Choice |
| `official_score_safari.json` | Assess a Safari bug's severity | Score |
| `official_noul_escalation.json` | Detect human escalation and repeat contact | Noul with and without criteria |

The JSON files default to `model: kalm-embedding-v2.5`. To switch models, change only `model` and load the corresponding configuration. Keep state, instructions, and criteria unchanged.

With the shipped models' retrieval Noul format, Choice/Score temperature 0.1, and Noul slope 10/intercept 0, the unadapted models returned the following values on the three requests. These are CPU FP32 inference results with caching disabled, rounded to six decimals. They are example predictions, not accuracy measurements or calibrated probabilities.

| Model | Choice: `returns` probability | Score: expected severity | Noul: human escalation | Noul: repeat contact |
| --- | ---: | ---: | ---: | ---: |
| KaLM v2.5 | 0.793768 | 1.256149 | 0.999815 | 0.517449 |
| Qwen3 0.6B | 0.551758 | 1.096047 | 0.999774 | 0.625097 |
| Qwen3 4B | 0.553991 | 0.979757 | 0.999854 | 0.547160 |
| E5 large instruct | 0.438287 | 1.097949 | 0.999777 | 0.523098 |

All four models select `returns` for Choice. In Noul, every input uses the fixed retrieval instruction; with criteria, the original question is joined to each supplied criterion with a newline. Criteria-free scores near 1 can also occur for semantically related negative cases. The full KaLM responses and Jev reference outputs appear in the [main README](../README.md#official-jev-examples-with-kalm-results). Small differences are possible with another device, precision, or model revision.

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
