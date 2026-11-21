# Kanana-2-v-decide-3b-2611

JevEmbed includes a dedicated backend for Kakao's [Kanana-2-v-decide-3b-2611](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611). The checkpoint has a 3.67B multimodal backbone and was trained for typed decisions. It encodes each state/question and candidate separately into 2,560-dimensional vectors, then scores candidates with the temperature stored in the checkpoint.

The backend uses Kanana's own query and document instructions, chat template, image-free processor, last-token pooling, and L2 normalization. JevEmbed handles Choice, Score, and Noul distributions and caches repeated embeddings. Choice/Score temperature and Noul slope in the supplied YAML both come from the checkpoint's native decision temperature; they are not independently recalibrated on JevEmbed data.

## Install in an isolated environment

Kanana's custom model code requires Transformers 5.5.3 and PyTorch 2.10.0. The rest of the tested JevEmbed local model stack uses Transformers 4.51.0, so install this backend in a separate environment rather than upgrading a shared JevEmbed environment in place.

The model's reference environment uses Python 3.12. Choose a PyTorch wheel matching the deployment host's CUDA setup.

```powershell
python -m venv .venv-kanana
.\.venv-kanana\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[kanana]"
```

The model is downloaded from Hugging Face on the first inference. The supplied configuration pins model and custom-code files to revision `12820d654ed9789a0f7f48daeee1703c3483c464`:

```powershell
python -m jevembed --config configs/kanana-2-v-decide-3b-2611.yaml `
  --input examples/kanana_decide.json
```

For local/offline weights, download that exact revision and set `model_name_or_path` to its directory and `local_files_only: true` in a copied local YAML. Do not edit the tracked public configuration for local paths.

## Request shape

Use the standard JevEmbed Choice, Score, and Noul request. The model ID is `kanana-2-v-decide-3b-2611`; the Hugging Face repository ID is also registered as an alias. JevEmbed constructs Kanana's native state/question payload and uses the original criteria for its query prompt. Each candidate is encoded with Kanana's document instruction.

JevEmbed currently accepts text and JSON states. Kanana itself supports images and message histories, but those inputs are not available through JevEmbed's current request schema, so this integration does not yet use the visual encoder. The checkpoint runtime processes each text independently; `batch_size` limits JevEmbed's cache-miss batches, not a batched model forward.

The model card reports up to 20,480 tokens. Overlong state text and candidate descriptions are shortened using Kanana's own truncation helpers; set `overflow_policy: error` to reject overlong inputs instead. Criteria are included in the query as required by the native decision renderer.

The reference model's weights use Kakao's [KananaOpenLicense](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611/blob/main/LICENSE).

## Implementation and validation

The standard compiler constructs candidates, task plans, and Score legends. The Kanana adapter preserves the original state/question schema for the upstream renderer and delegates model loading, message rendering, truncation, and embedding to the pinned checkpoint's implementation. JevEmbed's existing scoring, embedding cache, batching, CLI, benchmark runner, and HTTP server remain the execution path.

Validation on 2026-11-10 passed 306 unit/integration tests, including 10 reference checks against the actual upstream renderer and tokenizer. These checks cover Choice, Score, Noul with and without criteria, ordered structured JSON, exact query/candidate token inputs, scoring equations, truncation, and option lists above 255 candidates. CLI `--explain`, wheel creation, and source-package creation also passed.

This validation did not load the full 3.67B checkpoint or establish model accuracy. The development GPU has 6 GiB VRAM, below the roughly 7.3 GB required for BF16 weights alone. The existing test-set evaluator's two tests were excluded from that run because the lightweight test environment did not contain PyTorch; training tests were outside this inference change.

To repeat the reference checks, install the development dependencies and tokenizer stack, download the pinned snapshot's `decision_kanana_v_decide.py`, tokenizer files, and `chat_template.jinja`, then set its path:

```powershell
$env:JEVEMBED_KANANA_REFERENCE = "C:\path\to\kanana-snapshot"
python -m pytest tests/integration/test_kanana_reference.py -q
```
