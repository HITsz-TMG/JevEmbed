# JevEmbed

![JevEmbed pixel-art banner showing embeddings leading to Choice, Score, and Noul decisions](assets/jevembed-banner.png)

**Meet JevEmbed — turn embeddings into decisions.**

*Choose, score, and judge with your choice of embedding model.*

JevEmbed is a Python framework that turns embedding models into structured decision engines for Choice, Score, and Noul tasks. It offers a consistent Jev-style interface through a Python API, CLI, and optional HTTP server.

## News

- **September 26, 2026:** JevEmbed now includes ready-to-use configurations for [JevEmbed-Qwen3-Embedding-0.6B](https://huggingface.co/HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B) and [JevEmbed-KaLM-Embedding-V2.5](https://huggingface.co/HIT-TMG/JevEmbed-KaLM-Embedding-V2.5).
- **September 25, 2026:** [JevEmbed-KaLM-Embedding-V2.5](https://huggingface.co/HIT-TMG/JevEmbed-KaLM-Embedding-V2.5) is available on Hugging Face with merged weights and a LoRA adapter.
- **September 24, 2026:** JevEmbed now supports [CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) for Choice, Score, and Noul decisions. See the [CLM-v0.1-8B setup guide](docs/clm.md).
- **September 24, 2026:** [JevEmbed-Data](https://huggingface.co/datasets/HIT-TMG/JevEmbed-Data) is available for fine-tuning, with 1.67 million labeled Choice, Score, and Noul questions.

## Installation

Use Python 3.10–3.12 for the tested model stack. Run these commands from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` includes local inference and HTTP dependencies with the constraints in `requirements-models-tested.txt`. Contributors can install `requirements-dev.txt` for testing and packaging.

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

The tested stack uses PyTorch 2.8.0, Transformers 4.51.0, sentence-transformers 5.3.0, NumPy 1.26.4, and PyYAML 6.0.3. Transformers 4.51.0 is pinned for KaLM-embedding-multilingual-mini-instruct-v2.5 compatibility. Choose a PyTorch wheel for your hardware; CPU inference needs no FlashAttention.

For a minimal installation, select the required optional dependencies:

```bash
python -m pip install -e .                        # Core, configuration, and explain
python -m pip install -e '.[http,server]'          # HTTP backend and API
python -m pip install -e '.[local]' -c requirements-models-tested.txt
python -m pip install -r requirements-dev.txt     # Tests and builds, without model libraries
```

Core imports and `--explain` require no model weights or service.

## Supported models

Configurations use Hugging Face repository IDs; weights download on first inference if absent from the local cache.

| JevEmbed ID / configuration filename | Weight repository | Dimensions | Token limit | Pooling |
| --- | --- | --- | --- | --- |
| `kalm-embedding-v2.5` | `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` | 896 | 32768 | Mean |
| `qwen3-embedding-0.6b` | `Qwen/Qwen3-Embedding-0.6B` | 1024 | 32768 | Last-token |
| `qwen3-embedding-4b` | `Qwen/Qwen3-Embedding-4B` | 2560 | 32768 | Last-token |
| `qwen3-embedding-8b` | `Qwen/Qwen3-Embedding-8B` | 4096 | 32768 | Last-token |
| `multilingual-e5-large-instruct` | `intfloat/multilingual-e5-large-instruct` | 1024 | 512 | Mean |
| `clm-v0.1-8b` | `Contrastive-LM/CLM-v0.1-8B` projection heads + `Qwen/Qwen3-8B` encoder | 512 | 2048 | Last-token + paired heads |
| [`jevembed-kalm-embedding-v2.5`](configs/jevembed-kalm-embedding-v2.5.yaml) | [HIT-TMG/JevEmbed-KaLM-Embedding-V2.5](https://huggingface.co/HIT-TMG/JevEmbed-KaLM-Embedding-V2.5) | 896 | 1024 | Mean |
| [`jevembed-qwen3-embedding-0.6b`](configs/jevembed-qwen3-embedding-0.6b.yaml) | [HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B](https://huggingface.co/HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B) | 1024 | 1024 | Last-token |

Configurations are in `configs/<model-id>.yaml`. Repository IDs work as aliases; responses use the short ID. Unknown IDs, including `jev-latest`, are rejected.

The JevEmbed releases are fine-tuned for Choice, Score, and Noul and use 1,024-token truncation.

KaLM-embedding-multilingual-mini-instruct-v2.5 requires `trust_remote_code: true`. Qwen3-Embedding-0.6B, Qwen3-Embedding-4B, Qwen3-Embedding-8B, multilingual-e5-large-instruct, and CLM-v0.1-8B use `false`. Loading failures do not change trust or pooling settings. Pin revisions and dependency versions to reproduce results.

CLM-v0.1-8B uses separate state/action projections and [model-specific prompts](configs/clm-v0.1-8b.yaml), with Choice/Score temperature 0.01 and Noul slope 100. See the [CLM-v0.1-8B guide](docs/clm.md).

## Quick start

```bash
python -m jevembed --config configs/kalm-embedding-v2.5.yaml \
  --input examples/official_choice_exchange.json
```

Repeat `--config` for multiple models and set the request's `model` field to the selected ID. `--input -` reads standard input.

For JevEmbed-Qwen3-Embedding-0.6B, use `--config configs/jevembed-qwen3-embedding-0.6b.yaml` with `"model": "jevembed-qwen3-embedding-0.6b"`. JevEmbed-KaLM-Embedding-V2.5 uses its [configuration](configs/jevembed-kalm-embedding-v2.5.yaml) and `jevembed-kalm-embedding-v2.5` ID.

The shared [examples](examples/README.md) cover all three tasks. To try another model, replace `"model": "kalm-embedding-v2.5"` in the request and select the matching configuration. The [local runner](scripts/run_local.sh) is available for source checkouts.

## Official Jev examples with kalm-embedding-v2.5 results

These Jev documentation examples use `model: kalm-embedding-v2.5`. The saved Jev outputs appear alongside KaLM-embedding-multilingual-mini-instruct-v2.5 predictions for reference.

### Evaluation settings

Inference used CPU FP32 with caching disabled. The [KaLM-embedding-multilingual-mini-instruct-v2.5 configuration](configs/kalm-embedding-v2.5.yaml) sets Choice/Score temperature **0.1** and Noul slope **10**, intercept **0**, for both Noul paths. Choice/Score probabilities use softmax over cosine similarities; Noul applies a sigmoid to one similarity or a true-minus-false difference. Normalized-entropy confidence measures distribution concentration. **These predictions are uncalibrated** and may vary slightly by hardware, precision, or revision. See [input mapping and scoring](#input-mapping-and-scoring) and [calibration](docs/calibration.md).

### Example results

| Example | Jev reference output | JevEmbed with KaLM-embedding-multilingual-mini-instruct-v2.5 |
| --- | --- | --- |
| Choice: exchange routing | `returns`, probability 1.0 | `returns`, probability 0.793768 |
| Score: Safari bug severity | 1.43 | 1.256149 |
| Noul: human escalation | 0.99 | 0.999815 |
| Noul: repeat contact | 0.93 | 0.520574 |

The reference outputs are saved Jev documentation predictions. Neither column is ground truth; these examples illustrate behavior, not accuracy.

The same requests were run on the six original base models with CUDA BF16; see the [example results](examples/README.md).

### Choice: route an exchange request

Source: [Jev Choice example](https://docs.typesafe.ai/primitives/choice). Saved [official reference response](tests/fixtures/official_choice_exchange.reference.json).

Request ([file](examples/official_choice_exchange.json)):

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

KaLM-embedding-multilingual-mini-instruct-v2.5 response:

```json
{
  "model": "kalm-embedding-v2.5",
  "answers": {
    "department": {
      "type": "choice",
      "probabilities": {
        "returns": 0.7937679922689964,
        "shipping": 0.131790116809968,
        "billing": 0.07444189092103566
      },
      "confidence": 0.4139963162182463,
      "choice": "returns"
    }
  },
  "usage": {
    "input_tokens": 61,
    "output_tokens": 0
  }
}
```

### Score: assess a Safari bug

Source: [Jev Score example](https://docs.typesafe.ai/primitives/score). Saved [official reference response](tests/fixtures/official_score_safari.reference.json).

Request ([file](examples/official_score_safari.json)):

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

KaLM-embedding-multilingual-mini-instruct-v2.5 response:

```json
{
  "model": "kalm-embedding-v2.5",
  "answers": {
    "bug_severity": {
      "type": "score",
      "probabilities": {
        "0": 0.15376904118628926,
        "1": 0.43631340883021275,
        "2": 0.40991754998349794
      },
      "confidence": 0.07579551491122716,
      "score": 1.2561485087972086,
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

### Noul: human escalation and repeat contact

Source: [Jev Noul example](https://docs.typesafe.ai/primitives/noul). Saved [official reference response](tests/fixtures/official_noul_escalation.reference.json).

Request ([file](examples/official_noul_escalation.json)):

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

KaLM-embedding-multilingual-mini-instruct-v2.5 response:

```json
{
  "model": "kalm-embedding-v2.5",
  "answers": {
    "is_human_escalation": {
      "type": "noul",
      "noul": 0.9998150635615807
    },
    "is_repeat_contact": {
      "type": "noul",
      "noul": 0.5205740521193529
    }
  },
  "usage": {
    "input_tokens": 136,
    "output_tokens": 0
  }
}
```

The criteria-free path compares the question with the state; the other compares true and false criteria. Both outputs exceed 0.5 here, but high state/question similarity is not a calibrated probability of agreement.

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

JevEmbed converts each request into embedding inputs, computes cosine similarities, and applies the scoring rule for the selected primitive. The following cases show the model inputs for the [official examples](#official-jev-examples-with-kalm-embedding-v25-results) using the default templates.

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

**KaLM-embedding-multilingual-mini-instruct-v2.5 result:** `returns`, probability **0.793768** at temperature **0.1**.

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

**KaLM-embedding-multilingual-mini-instruct-v2.5 result:** probabilities approximately **[0.153769, 0.436313, 0.409918]**, giving a score of **1.256149** at temperature **0.1**. The response also retains the descriptions as its legend.

### Noul with criteria — compare the question and state with each criterion

**Case: detect repeat contact.** Encode the question and state together under the retrieval instruction:

```text
Instruct: Retrieve semantically similar text.
Query: Has the customer contacted support about this before?
I have asked three times now. Can I please just talk to a real person?
```

Encode each criterion as a separate query, prefixed with its true/false label:

```text
Instruct: Retrieve semantically similar text.
Query: true: Mentions a prior attempt, ticket, or that they have asked before
```

```text
Instruct: Retrieve semantically similar text.
Query: false: No sign of any previous contact
```

**Scoring:** compare the question-plus-state query with each criterion query, subtract the false similarity from the true similarity, and apply `sigmoid(difference / 0.1)`.

**KaLM-embedding-multilingual-mini-instruct-v2.5 result:** **0.520574** with slope **10.0** and intercept **0.0**. The true and false similarities were **0.756465** and **0.748231**; the margin is small, so this single result does not establish reliable repeat-contact detection.

### Noul without criteria — compare two queries

**Case: detect a request for a human agent.** Encode the original question and state separately under the same fixed retrieval instruction:

```text
Instruct: Retrieve semantically similar text.
Query: Is the customer asking for a human agent?
```

```text
Instruct: Retrieve semantically similar text.
Query: I have asked three times now. Can I please just talk to a real person?
```

**Scoring:** use `sigmoid(cosine(question, state) / 0.1)`. There is no negative comparison in this path, so semantic relevance alone can produce a high score.

**KaLM-embedding-multilingual-mini-instruct-v2.5 result:** **0.999815** with slope **10.0** and intercept **0.0**. The high value reflects semantic similarity between the state and question; this path has no negative comparison and can also score related negative cases highly.

### Interpreting outputs

Temperature **0.1** sharpens Choice/Score probabilities without changing their ranking. Both Noul paths use slope **10** and intercept **0**, with separate settings for a similarity difference and a single similarity. Confidence measures concentration, not correctness; Noul has no confidence field. See [scoring and calibration](docs/calibration.md) for details, `--explain` for rendered inputs, and the [runtime guide](docs/runtime.md) for model and cache behavior.

## HTTP interfaces

```bash
python -m jevembed --config configs/kalm-embedding-v2.5.yaml --serve
```

The server binds to `127.0.0.1:8000` and provides `POST /v1/systemone` and `GET /v1/models`. HTTP requests require `model`; validation errors return 422 and backend failures 502. Per process, the defaults are 2 MiB, 64 questions, 4096 embedding inputs, four active requests, and a 30-second body read timeout. Size or work violations return 413, excess concurrency 429, and slow uploads 408. The [HTTP guide](docs/http.md) has `curl` and Python examples; [serving limits](docs/runtime.md#http-serving-limits) covers options and concurrency.

Use [http-example.yaml](configs/http-example.yaml) to connect to an embeddings service. Templates are rendered by the client; the service must enforce input length before `server_enforces_length: true` is set. The [HTTP guide](docs/http.md#call-an-external-embeddings-service) explains this separate backend.

## LoRA fine-tuning

JevEmbed fine-tunes Sentence Transformers encoders on Choice, Score, and Noul supervision, including soft targets. The released [JevEmbed-KaLM-Embedding-V2.5](https://huggingface.co/HIT-TMG/JevEmbed-KaLM-Embedding-V2.5) and [JevEmbed-Qwen3-Embedding-0.6B](https://huggingface.co/HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B) models each trained for **one epoch on all 1,601,157 JevEmbed-Data training questions**, then had their LoRA weights merged into standalone models. Both used:

- BF16 and an effective batch of **512 questions**;
- LoRA rank **64**, alpha **32**, dropout **0.05**, and `q_proj`/`k_proj`/`v_proj` targets;
- learning rate **2 × 10⁻⁴** with **10% warmup** and **1,024-token truncation**;
- Choice/Score temperature **0.1** and Noul slope **10** in the [KaLM-embedding-multilingual-mini-instruct-v2.5](configs/kalm-embedding-v2.5.yaml) and [Qwen3-Embedding-0.6B](configs/qwen3-embedding-0.6b.yaml) base configurations.

Install the training extra with `python -m pip install -e '.[train]' -c requirements-models-tested.txt`. The trainer reads supervised JSONL, so convert the dataset's `train` Parquet rows to the [documented JSONL schema](docs/training.md#supervision-format-and-objectives) and reserve `test` for final evaluation. The [training guide](docs/training.md) covers commands, validation, and adapter inference; the model cards give each release's exact settings.

On the held-out JevEmbed-Data `test` split, overall hard-label accuracy covers **64,110** of **66,482** questions:

| Model | Base | After JevEmbed-Data fine-tuning | Change |
| --- | ---: | ---: | ---: |
| `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` | 32.33% | 76.03% | +43.70 pp |
| `Qwen/Qwen3-Embedding-0.6B` | 33.79% | **82.30%** | +48.51 pp |

The [full test report](reports/JEVEMBED_DATA_TEST.md) gives Choice, Score, and Noul accuracy and error metrics.

## Synthetic training data

The [synthesis guide](docs/synthesis.md) shows how to generate hard-labeled Choice, Score, or Noul JSONL for a fixed question using an OpenAI-compatible teacher. It includes four example configs, schema checks, configurable label quotas, a separate teacher label check, optional trusted reference rules, and resumable output. Review the generated data before using it with the trainer.

## Development and validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q

# Build a wheel and source distribution
python -m build
```

Tests use fixtures and mock models without downloading weights. Save generated traces under the ignored `artifacts/` directory.

## JevEmbed-Data test results

The six base models and two JevEmbed releases were evaluated on the same **66,482-question** [JevEmbed-Data](https://huggingface.co/datasets/HIT-TMG/JevEmbed-Data) `test` split. Runs used BF16, each model's configured prompts and scoring, and truncation at 1,024 tokens for KaLM-embedding-multilingual-mini-instruct-v2.5, Qwen3-Embedding-0.6B, Qwen3-Embedding-4B, Qwen3-Embedding-8B, and both JevEmbed releases; 512 for multilingual-e5-large-instruct; and 2,048 for CLM-v0.1-8B. Accuracy covers **64,110 hard-labeled** questions; soft labels are excluded. The released models used the JevEmbed-Data training split; the base rows show original weights.

| Model | Choice (17,487) | Score (24,260) | Noul (22,363) | Overall (64,110) |
| --- | ---: | ---: | ---: | ---: |
| `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` | 28.61% | 27.45% | 40.52% | 32.33% |
| `Qwen/Qwen3-Embedding-0.6B` | 33.02% | 28.56% | 40.05% | 33.79% |
| `Qwen/Qwen3-Embedding-4B` | 38.11% | 30.55% | 41.09% | 36.29% |
| `Qwen/Qwen3-Embedding-8B` | 43.20% | 33.49% | 42.57% | 39.30% |
| `intfloat/multilingual-e5-large-instruct` | 32.07% | 24.27% | 40.54% | 32.07% |
| `Contrastive-LM/CLM-v0.1-8B` | 28.59% | 25.85% | 53.74% | 36.33% |
| [JevEmbed-KaLM-Embedding-V2.5](https://huggingface.co/HIT-TMG/JevEmbed-KaLM-Embedding-V2.5) | 71.05% | 66.17% | 90.61% | 76.03% |
| [JevEmbed-Qwen3-Embedding-0.6B](https://huggingface.co/HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B) | 84.53% | 69.55% | 94.38% | **82.30%** |

The [full test report](reports/JEVEMBED_DATA_TEST.md) gives the error metrics, denominators, and evaluation command.

See the [compatibility boundaries](docs/compatibility.md) and [architecture and design](docs/design.md) for the implementation contract.

## Acknowledgments

JevEmbed uses [Sentence Transformers](https://www.sbert.net/) for model loading and embedding inference. We thank its maintainers and contributors for the library and its support for the embedding community.

## Citation

If you find JevEmbed useful, please consider citing the following papers:

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
