# HTTP interfaces

[Back to README](../README.md#http-interfaces)

JevEmbed can **serve decisions** through its own HTTP API or **call an embeddings service** as a backend. The first accepts Jev Choice, Score, and Noul requests at `/v1/systemone`. The second sends rendered text to an upstream `/v1/embeddings` endpoint; that upstream endpoint is not provided by JevEmbed's decision server.

## Serve JevEmbed decisions

Install the project as described in the [README](../README.md#installation), then start a server from the repository root:

```bash
python -m jevembed --config configs/kalm-embedding-v2.5.yaml --serve
```

It listens on `127.0.0.1:8000` by default. Use `--host` and `--port` to change the bind address and port. Model weights load when first needed, so a successful model-list response does not mean inference has already loaded the weights.

List the registered model IDs and aliases:

```bash
curl -sS http://127.0.0.1:8000/v1/models
```

The response has `"object": "list"` and a `data` array of objects with `id` and `aliases`. To register several models, repeat `--config` when starting the server:

```bash
python -m jevembed \
  --config configs/kalm-embedding-v2.5.yaml \
  --config configs/qwen3-embedding-0.6b.yaml \
  --serve
```

Every HTTP decision request must include a registered `model` ID or alias, even when only one model is loaded. Responses use the canonical short ID. Send a Choice request with `curl`:

```bash
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "kalm-embedding-v2.5",
  "state": "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "returns": "Exchanges, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
        "billing": "Charges, invoices, payment problems"
      }
    }
  }
}
JSON
```

The response contains `model`, `answers.department` (including `choice`, `probabilities`, and `confidence`), and `usage`. The [README example](../README.md#choice-route-an-exchange-request) shows a complete response. The same endpoint accepts Score and Noul requests using the [shared examples](../examples/README.md).

Python callers can send the same request with `httpx`:

```python
import json
from pathlib import Path
import httpx

request = json.loads(Path("examples/official_choice_exchange.json").read_text(encoding="utf-8"))
response = httpx.post("http://127.0.0.1:8000/v1/systemone", json=request, timeout=300.0)
response.raise_for_status()
print(response.json()["answers"]["department"]["choice"])
```

The `httpx` timeout is the caller's wait limit; it is independent of the server's request-body read timeout. Install `httpx` with the project's standard requirements or `pip install -e '.[http]'`.

## Limits and responses

The POST endpoint applies these defaults per server process:

| Limit | Default | CLI option |
| --- | ---: | --- |
| JSON body | 2 MiB | `--max-request-bytes` |
| Questions | 64 | `--max-questions` |
| Compiled embedding inputs | 4096 | `--max-embedding-inputs` |
| Active requests | 4 | `--max-concurrent-requests` |
| Entire body read | 30 seconds | `--body-read-timeout-seconds` |

Invalid requests return **422**; backend failures return **502**. Size or work violations return **413**. When all request slots are busy, the server returns **429** immediately with `Retry-After: 1`. An upload that exceeds the overall body read deadline returns **408** and frees its slot. That deadline covers only reading the body, not model inference. Adjust the limits with the CLI options above or `HTTPServiceLimits` when using `create_app`; see [HTTP serving limits](runtime.md#http-serving-limits). The [compatibility guide](compatibility.md) describes the request and response contract.

## Call an external embeddings service

To use an upstream embedding model while keeping JevEmbed's decision logic, start from [http-example.yaml](../configs/http-example.yaml). Set `base_url` to the upstream `/v1` root and `model_name_or_path` to the model ID that service accepts. JevEmbed sends batches to `base_url + "/embeddings"`; it does not send Jev decisions to that endpoint. The config's `model_id` (`embedding-service` in the example) is the ID to put in JevEmbed requests.

The generic backend renders prompts on the client and requires the upstream service to reject overlong inputs. Set `server_enforces_length: true` only after enabling that behavior upstream; `overflow_policy: error` is required. With `usage_mode: strict`, responses must include exact `usage.prompt_tokens` and indexed float embeddings. Optional `api_key_env` names an environment variable containing the upstream API key. Its `timeout` and `max_retries` control upstream embedding calls, independently of the decision server's upload deadline.

Copy the example, edit its upstream settings, and set the request's `model` to `embedding-service`. Then use it through the CLI or JevEmbed decision server:

```bash
mkdir -p configs/local
cp configs/http-example.yaml configs/local/embeddings.yaml
# Edit configs/local/embeddings.yaml before running either command.
python -m jevembed --config configs/local/embeddings.yaml --input your-request.json
# Or expose decisions that use the upstream embeddings service:
python -m jevembed --config configs/local/embeddings.yaml --serve
```

The upstream service must provide `/v1/embeddings`; JevEmbed's own `--serve` mode provides `/v1/systemone` and `/v1/models` instead.
