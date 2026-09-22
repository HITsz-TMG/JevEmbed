# JevEmbed

**Meet JevEmbed — turn embeddings into decisions.**

*Choose, score, and judge with your choice of embedding model.*

JevEmbed is a Python framework for embedding-based Choice, Score, and Noul decisions. It provides a Python API, CLI, and optional HTTP server using Jev-style request and response schemas. JevEmbed is an independent implementation.

## Installation

Use Python 3.10–3.12 for the tested model stack. Run these commands from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` installs dependencies for local inference, the HTTP client, and the HTTP server. It automatically applies the version constraints in `requirements-models-tested.txt`; no separate installation is needed. Contributors can additionally install `requirements-dev.txt` for testing and packaging.

| Dependency | Purpose |
| --- | --- |
| PyYAML | Load YAML model configurations |
| NumPy | Handle embedding arrays |
| PyTorch | Run CPU or CUDA inference |
| Transformers | Load models and tokenizers |
| sentence-transformers | Use repository-provided modules, encoding, and pooling |
| httpx | Call a compatible `/v1/embeddings` service |
| FastAPI | Expose the Jev-shaped HTTP API |
| Uvicorn | Run the HTTP server |

The tested model stack uses PyTorch 2.8.0, Transformers 4.51.0, sentence-transformers 5.3.0, NumPy 1.26.4, and PyYAML 6.0.3. The Transformers version is pinned for compatibility with KaLM's custom implementation. Select a PyTorch wheel suitable for your hardware and driver. CPU inference is supported without FlashAttention.

For a minimal installation, select the required optional dependencies:

```bash
python -m pip install -e .                        # Core, configuration, and explain
python -m pip install -e '.[http,server]'          # HTTP backend and API
python -m pip install -e '.[local]' -c requirements-models-tested.txt
python -m pip install -r requirements-dev.txt     # Tests and builds, without model libraries
```

Importing the core package and running `--explain` do not load weights, access a GPU, or connect to a model service.

## Supported models

The provided configurations use Hugging Face repository IDs. Weights are downloaded on first inference if unavailable in the local cache; they are distributed separately from this project.

| JevEmbed ID / configuration filename | Weight repository | Dimensions | Token limit | Pooling |
| --- | --- | --- | --- | --- |
| `kalm-embedding-v2.5` | `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` | 896 | 32768 | Mean |
| `qwen3-embedding-0.6b` | `Qwen/Qwen3-Embedding-0.6B` | 1024 | 32768 | Last-token |
| `qwen3-embedding-4b` | `Qwen/Qwen3-Embedding-4B` | 2560 | 32768 | Last-token |
| `multilingual-e5-large-instruct` | `intfloat/multilingual-e5-large-instruct` | 1024 | 512 | Mean |

Configurations are stored in `configs/<model-id>.yaml`. Each full repository ID is registered as an alias, while responses use the canonical short ID. Unknown model IDs are rejected. `jev-latest` is not registered automatically.

KaLM explicitly enables `trust_remote_code: true` to load its repository-provided Python implementation. Qwen3 and E5 use `false`. A loading failure never enables trust or changes pooling automatically. Pin a revision and preserve dependency and template versions when reproducing results.

## Quick start

```bash
# Inspect the compiled inputs without loading weights
python -m jevembed --config configs/kalm-embedding-v2.5.yaml \
  --input examples/official_noul_escalation.json --explain

# Run inference and print a standard JSON response
python -m jevembed --config configs/kalm-embedding-v2.5.yaml \
  --input examples/official_choice_exchange.json

# Save the response and a separate diagnostic trace
mkdir -p artifacts
python -m jevembed --config configs/kalm-embedding-v2.5.yaml \
  --input examples/official_noul_escalation.json --trace --output artifacts/noul-trace.json
```

Repeat `--config` to register multiple models. When switching models, also change the request's `model` field. `--input -` reads JSON from standard input; output goes to standard output unless a file is specified.

The shared [examples](examples/README.md) cover Choice, Score, and both Noul paths. They default to KaLM and work with every supported model by changing only `model` and the selected configuration.

On POSIX systems, `bash scripts/run_local.sh ...` runs from a source checkout using the current environment's `python3`. Set `JEVEMBED_PYTHON=python` to choose another interpreter. The script sets the project import path and runs the CLI.

## Official Jev examples with KaLM results

The following requests reproduce the Jev documentation examples, changing only `model` to `kalm-embedding-v2.5`. The responses are JevEmbed predictions using KaLM. Outputs from the Jev documentation are included for reference.

### Evaluation settings

Model inference used an H100 MIG GPU with BF16 and caching disabled. The results below use these scoring settings from the [KaLM configuration](configs/kalm-embedding-v2.5.yaml):

```yaml
scoring:
  choice_temperature: 0.5
  score_temperature: 0.5
  noul_criteria: {slope: 1.0, intercept: 0.0}
  noul_similarity: {slope: 1.0, intercept: 0.0}
  confidence: normalized_entropy
  calibration_status: uncalibrated
```

| Setting | Applies to | Scoring behavior |
| --- | --- | --- |
| `choice_temperature: 0.5` | Exchange routing | Probabilities are `softmax(cosine / T)`. A smaller positive T sharpens the distribution without changing the similarity ranking or winning choice. |
| `score_temperature: 0.5` | Safari severity | Uses the same softmax with an independent T. Changing T changes the distribution and can change its expected score; level rankings stay the same. |
| `noul_criteria` | Repeat contact | Uses `sigmoid(slope * (s_true - s_false) + intercept)`. Choice/Score temperatures have no effect on this path. |
| `noul_similarity` | Human escalation | Uses `sigmoid(slope * cosine + intercept)`, independently of the criteria path and the softmax temperatures. |
| `confidence: normalized_entropy` | Choice and Score | Measures distribution concentration. A sharper distribution can raise confidence without improving correctness. |

**These results are uncalibrated.** At T=0.5, similar cosine scores produce similar probabilities. Lower temperatures sharpen the distribution while preserving its ranking. Noul uses independent slope and intercept parameters to control the mapping and decision threshold. No parameters were fitted to these examples.

Small numerical differences can occur with different hardware, precision, or model revisions. See [input mapping and scoring](#input-mapping-and-scoring) for the calculation steps and [calibration](docs/calibration.md) for fitting guidance.

### Example results

| Example | Jev reference output | JevEmbed with KaLM v2.5 |
| --- | --- | --- |
| Choice: exchange routing | `returns`, probability 1.0 | `returns`, probability 0.430626 |
| Score: Safari bug severity | 1.43 | 1.063471 |
| Noul: human escalation | 0.99 | 0.702647 |
| Noul: repeat contact | 0.93 | 0.486044 |

Reference outputs are taken from the saved Jev documentation examples, rather than new API calls. Both columns contain model predictions, not independently annotated ground truth. These examples illustrate API behavior and do not constitute an accuracy benchmark.

### Choice: route an exchange request

Source: [Jev Choice example](https://docs.typesafe.ai/primitives/choice). Saved [official reference response](tests/fixtures/official_choice_exchange.reference.json).

Run the [request file](examples/official_choice_exchange.json):

```bash
python -m jevembed --config configs/kalm-embedding-v2.5.yaml \
  --input examples/official_choice_exchange.json
```

Request:

```json
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
```

KaLM response:

```json
{
  "model": "kalm-embedding-v2.5",
  "answers": {
    "department": {
      "type": "choice",
      "probabilities": {
        "returns": 0.43062562335000293,
        "shipping": 0.3013149132578033,
        "billing": 0.2680594633921939
      },
      "confidence": 0.019509453551041944,
      "choice": "returns"
    }
  },
  "usage": {
    "input_tokens": 61,
    "output_tokens": 0
  }
}
```

KaLM selects `returns` and retains the probability distribution over all candidates.

### Score: assess a Safari bug

Source: [Jev Score example](https://docs.typesafe.ai/primitives/score). Saved [official reference response](tests/fixtures/official_score_safari.reference.json).

Run the [request file](examples/official_score_safari.json):

```bash
python -m jevembed --config configs/kalm-embedding-v2.5.yaml \
  --input examples/official_score_safari.json
```

Request:

```json
{
  "model": "kalm-embedding-v2.5",
  "state": "The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari.",
  "questions": {
    "bug_severity": {
      "type": "score",
      "instructions": "How severe is the reported issue?",
      "criteria": [
        "Cosmetic; no impact to functionality",
        "Broken or degraded feature, but workaround exists",
        "Blocking issue; no workaround exists"
      ]
    }
  }
}
```

KaLM response:

```json
{
  "model": "kalm-embedding-v2.5",
  "answers": {
    "bug_severity": {
      "type": "score",
      "probabilities": {
        "0": 0.28926963348051227,
        "1": 0.35798942381727,
        "2": 0.3527409427022178
      },
      "confidence": 0.004091061243655325,
      "score": 1.0634713092217056,
      "legend": {
        "0": "Cosmetic; no impact to functionality",
        "1": "Broken or degraded feature, but workaround exists",
        "2": "Blocking issue; no workaround exists"
      }
    }
  },
  "usage": {
    "input_tokens": 62,
    "output_tokens": 0
  }
}
```

KaLM ranks the levels as 1 > 2 > 0. The predicted score, 1.063471, is the probability-weighted level index on the 0–2 scale. The response retains the full distribution and legend for interpretation.

### Noul: human escalation and repeat contact

Source: [Jev Noul example](https://docs.typesafe.ai/primitives/noul). Saved [official reference response](tests/fixtures/official_noul_escalation.reference.json).

Run the [request file](examples/official_noul_escalation.json):

```bash
python -m jevembed --config configs/kalm-embedding-v2.5.yaml \
  --input examples/official_noul_escalation.json
```

Request:

```json
{
  "model": "kalm-embedding-v2.5",
  "state": "I have asked three times now. Can I please just talk to a real person?",
  "questions": {
    "is_human_escalation": {
      "type": "noul",
      "instructions": "Is the customer asking for a human agent?"
    },
    "is_repeat_contact": {
      "type": "noul",
      "instructions": "Has the customer contacted support about this before?",
      "criteria": {
        "true": "Mentions a prior attempt, ticket, or that they have asked before",
        "false": "No sign of any previous contact"
      }
    }
  }
}
```

KaLM response:

```json
{
  "model": "kalm-embedding-v2.5",
  "answers": {
    "is_human_escalation": {
      "type": "noul",
      "noul": 0.702646900209176
    },
    "is_repeat_contact": {
      "type": "noul",
      "noul": 0.48604372031182913
    }
  },
  "usage": {
    "input_tokens": 104,
    "output_tokens": 0
  }
}
```

This request covers both Noul paths: human escalation omits criteria, while repeat contact supplies true/false descriptions. For repeat contact, KaLM assigns a lower similarity to the true description despite the explicit prior-contact statement. This is a semantic limitation of the current model and mapping; Choice/Score temperature does not affect Noul.

## Local weights and offline use

Copy a public configuration into the ignored `configs/local/` directory:

```bash
mkdir -p configs/local
cp configs/kalm-embedding-v2.5.yaml configs/local/kalm.yaml
```

Set the model directory and offline loading policy in the copied configuration:

```yaml
model_name_or_path: ./models/KaLM-embedding-multilingual-mini-instruct-v2.5
local_files_only: true
```

Relative paths resolve from the command's working directory. Run with `--config configs/local/kalm.yaml`. You can also set `HF_HUB_OFFLINE=1` for offline operation. `models/`, `configs/local/`, `*.local.yaml`, `.env`, and `artifacts/` are ignored by Git for local weights, configuration, and outputs.

## Python API

```python
import json
from jevembed import JevEmbed, ModelConfig

client = JevEmbed()
client.register(ModelConfig.load("configs/kalm-embedding-v2.5.yaml"))
client.register(ModelConfig.load("configs/qwen3-embedding-0.6b.yaml"))

with open("examples/official_noul_escalation.json", encoding="utf-8") as handle:
    request = json.load(handle)

compiled = client.explain(request)            # No model loading
response = client.evaluate(request)           # A plain dictionary
result = client.evaluate_with_trace(request)  # {"response": ..., "trace": ...}

request["model"] = "qwen3-embedding-0.6b"
response = client.evaluate(request)

# Python calls can omit model and use the first registered model
response = client.system_one(state="hello", questions={
    "greeting": {"type": "choice", "instructions": "Classify the message.",
                 "criteria": {"greeting": "A greeting", "other": "Other content"}}
})
client.clear_cache()
```

Custom backends implement `encode(list[EmbeddingInput]) -> EmbeddingBatch` and can be injected with `JevEmbed(backend=backend, model="my-model")`. Each input carries a role, instruction, and text. The backend renders its template and returns vectors and per-call token usage. The core handles deduplication, normalization, and scoring. Use `confidence_estimator` to supply a custom confidence function.

## Input mapping and scoring

JevEmbed converts each request into embedding inputs, computes cosine similarities, and applies the scoring rule for the selected primitive. The following cases show the model inputs for the [official examples](#official-jev-examples-with-kalm-results) using the default templates.

### Choice — select a candidate

**Case: route a shoe exchange request.** Combine the original instructions and state into one query:

```text
Instruct: Which team should handle this?
Query: My running shoes arrived in the wrong size. Can I swap them for a size 10?
```

Encode each candidate line separately, including its name:

```text
returns: Exchanges, wrong or damaged items
shipping: Delivery status, delays, lost packages
billing: Charges, invoices, payment problems
```

**Scoring:** compare the query with each candidate, apply `softmax(similarities / temperature)`, and select the candidate with the highest probability. A null description uses the candidate name alone.

**KaLM result:** `returns`, probability **0.430626** at temperature **0.5**.

### Score — estimate an ordinal value

**Case: rate a Safari bug's severity.** Encode the question:

```text
Instruct: How severe is the reported issue?
Query: The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari.
```

Encode each level description separately, without adding level numbers:

```text
Cosmetic; no impact to functionality
Broken or degraded feature, but workaround exists
Blocking issue; no workaround exists
```

**Scoring:** apply softmax and compute the expected level index: `score = 0*p0 + 1*p1 + 2*p2`. For K levels, the prediction ranges from 0 to K−1.

**KaLM result:** probabilities approximately **[0.289270, 0.357989, 0.352741]**, giving a score of **1.063471** at temperature **0.5**. The response also retains the descriptions as its legend.

### Noul with criteria — compare true and false descriptions

**Case: detect repeat contact.** Encode the question:

```text
Instruct: Has the customer contacted support about this before?
Query: I have asked three times now. Can I please just talk to a real person?
```

Encode the true description first and the false description second, as separate texts:

```text
Mentions a prior attempt, ticket, or that they have asked before
No sign of any previous contact
```

**Scoring:** subtract the false similarity from the true similarity, then apply `sigmoid(slope * difference + intercept)`. The true/false labels are not added to the model input.

**KaLM result:** **0.486044** with slope **1.0** and intercept **0.0**. The true description scored below the false description in this case, despite the customer's explicit prior-contact statement.

### Noul without criteria — compare the state and the question

**Case: detect a request for a human agent.** With no candidate descriptions, encode these two texts separately using the same fixed similarity instruction:

```text
Instruct: Retrieve semantically similar text.
Query: I have asked three times now. Can I please just talk to a real person?
```

```text
Instruct: Retrieve semantically similar text.
Query: Is the customer asking for a human agent?
```

**Scoring:** compare the two query vectors and apply `sigmoid(slope * similarity + intercept)`. No true/false candidates are generated.

**KaLM result:** **0.702647** with slope **1.0** and intercept **0.0**. This measures semantic similarity; it does not inherently distinguish agreement from negation.

### Interpreting outputs

- **Temperature** defaults to **0.5** and affects Choice and Score only. Lower values sharpen probabilities without improving the similarity ranking.
- **Slope and intercept** control Noul, with independent settings for its two paths.
- **Confidence** describes how concentrated a Choice or Score distribution is, not how likely the answer is to be correct. Noul has no confidence field.

All results above are uncalibrated and rounded for display. Full responses appear in the [official examples](#official-jev-examples-with-kalm-results); detailed formulas and fitting guidance are in [scoring and calibration](docs/calibration.md). Use `--explain` to inspect the exact model inputs for your own request.

For model encoding, device selection, caching, token usage, and input limits, see the [runtime guide](docs/runtime.md).

## HTTP interfaces

```bash
python -m jevembed --config configs/qwen3-embedding-0.6b.yaml --serve
```

The server binds to `127.0.0.1:8000` by default and provides `POST /v1/systemone` and `GET /v1/models`. HTTP requests require `model`. Validation errors return 422; backend failures return 502, without partial answers. `create_app(client, enable_debug=True)` additionally enables `/debug/explain`.

Use [http-example.yaml](configs/http-example.yaml) to connect to an embeddings service. Templates are rendered by the client; the service should not add prompts again. Configure the service to reject overlong inputs before declaring `server_enforces_length: true`. The generic adapter does not support server-owned templates or client-side truncation. It restores vector order from response indices and rejects missing or duplicate indices. Retries are bounded and limited to transient failures. API keys are read from the configured environment variable.

## Development and validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q

# Build a wheel and source distribution
python -m build
```

The automated test suite uses fixed vectors, mock models, and temporary local HTTP services without downloading model weights. It covers input mapping, scoring, caching, concurrent requests, length limits, backend errors, the CLI, and HTTP interfaces.

Original requests and reference responses are stored in `tests/fixtures/`. Save generated traces under the ignored `artifacts/` directory, as diagnostics may contain local paths.

## JevBench results

All runs use temperature **0.5**, BF16, and disabled caching on the same GPU. Accuracy is measured over **231 public tasks**.

| Model | Easy (48) | Standard (72) | Hard (111) | Overall (231) |
| --- | ---: | ---: | ---: | ---: |
| KaLM v2.5 | 83.33% | 55.56% | 38.74% | 53.25% |
| Qwen3 0.6B | 91.67% | 66.67% | 37.84% | 58.01% |
| Qwen3 4B | 95.83% | 59.72% | 42.34% | **58.87%** |
| E5 large instruct, default | 95.83% | 55.56% | 22.52% | 48.05% |
| E5 large instruct, explicit truncation | 95.83% | 55.56% | 30.63% | 51.95% |

E5's default 512-token limit rejects 53 Hard tasks, which count as incorrect. The supplemental truncation run processes those inputs at 512 tokens. All other runs return valid answers without truncation.

The [JevBench public evaluation](reports/JEVBENCH_PUBLIC.md) compares all four supported models on 231 labeled public tasks, with per-tier accuracy, calibration metrics, and measured latency. E5 results distinguish its default 512-token rejection policy from an explicit truncation run. This covers the available public subset, not the complete JevBench leaderboard suite.

Aggregated metrics and evaluation settings are available in [JSON format](reports/jevbench-public.json). The report includes commands for reproducing the runs with an existing JevBench checkout.

See the [compatibility boundaries](docs/compatibility.md) and [architecture and design](docs/design.md) for the implementation contract.

## Acknowledgments

JevEmbed uses [Sentence Transformers](https://www.sbert.net/) for model loading and embedding inference. We thank its maintainers and contributors for the library and its support for the embedding community.

## Citation

If you use KaLM embeddings in your research, please cite the following papers:

```bibtex
@misc{zhao2025kalmembeddingv2,
      title={KaLM-Embedding-V2: Superior Training Techniques and Data Inspire A Versatile Embedding Model},
      author={Xinping Zhao and Xinshuo Hu and Zifei Shan and Shouzheng Huang and Yao Zhou and Xin Zhang and Zetian Sun and Zhenyu Liu and Dongfang Li and Xinyuan Wei and Youcheng Pan and Yang Xiang and Meishan Zhang and Haofen Wang and Jun Yu and Baotian Hu and Min Zhang},
      year={2025},
      eprint={2506.20923},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.20923},
}

@misc{hu2025kalmembedding,
      title={KaLM-Embedding: Superior Training Data Brings A Stronger Embedding Model},
      author={Xinshuo Hu and Zifei Shan and Xinping Zhao and Zetian Sun and Zhenyu Liu and Dongfang Li and Shaolin Ye and Xinyuan Wei and Qian Chen and Baotian Hu and Haofen Wang and Jun Yu and Min Zhang},
      year={2025},
      eprint={2501.01028},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.01028},
}
```
