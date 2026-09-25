# CLM v0.1-8B

The [CLM v0.1-8B checkpoint](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) uses the base [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) model as its encoder. It is different from Qwen3-Embedding-8B. The configuration in [`configs/clm-v0.1-8b.yaml`](../configs/clm-v0.1-8b.yaml) downloads both weights on first use, or you can copy it and set `model_name_or_path` and `projection_name_or_path` to local directories.

CLM encodes the state followed by a blank line and the question instructions. Choice and Score encode each candidate description separately. Noul encodes `true: ...` and `false: ...` candidates, including when the request has no criteria. Structured inputs are rendered as plain text. The state text uses the state projection head; candidates use the action projection head. The encoder uses last-token pooling, and both projected vectors are normalized before cosine scoring.

These text layouts and fallback descriptions are configuration values. The `paired_projection` backend loads compatible two-head checkpoints, while JevEmbed's existing Choice, Score, and Noul scoring code produces the answers. For this checkpoint, the capped logit scale is 100: Choice/Score use temperature 0.01 and Noul uses slope 100 with zero intercept. Confidence uses the top probability minus the mean of the others. These values are uncalibrated.

To run a bundled example with CLM, select the configuration and set the request's model ID:

```python
import json
from jevembed import JevEmbed, ModelConfig

config = ModelConfig.load("configs/clm-v0.1-8b.yaml")
with open("examples/official_choice_exchange.json", encoding="utf-8") as handle:
    request = json.load(handle)
request["model"] = config.model_id

client = JevEmbed(config=config)
print(client.evaluate(request))
```

The [example results](../examples/README.md) and [JevEmbed-Data test results](../reports/JEVEMBED_DATA_TEST.md) provide measured outputs.
