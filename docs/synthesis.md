# Generate supervised data for a fixed Jev question

The synthesis CLI asks an OpenAI-compatible chat teacher to write states and hard labels for **one question you define**. It writes the same supervised JSONL shape that `jevembed.training` reads. This is a data preparation tool: it does not load an embedding model, train a model, or split data into train and test sets.

Install the small synthesis extra, then point one of the sample configurations at your teacher's `/v1` API root and advertised model ID:

```bash
python -m pip install -e '.[synthesis]'
python -m jevembed.synthesis --config configs/synthesis/choice-dinosaur.yaml --validate-only
python -m jevembed.synthesis \
  --config configs/synthesis/choice-dinosaur.yaml \
  --output artifacts/synthesis/dinosaur
```

The included configs use `http://127.0.0.1:18081/v1` and `Qwen3-8B` as **local example settings**. Change them to match your service. For a remote service, set `teacher.api_key_env` to the *name* of an environment variable holding the key; do not put the key in YAML. The client calls `<base_url>/chat/completions`, sends `model`, `messages`, `temperature`, and `max_tokens`, and accepts safe extra request fields in `teacher.request_options`. It does not follow redirects. `--validate-only` checks the config and prints upper bounds on teacher calls, HTTP attempts, and requested completion tokens without contacting the service or creating output files.

## Define a task

A V1 config fixes `task.question` for the entire job. Define its `type`, `instructions`, and Choice/Score/Noul `criteria` exactly as an ordinary Jev request would. Choice accepts exact candidate keys with descriptions or `null`; Score requires an ordered list of 2–10 levels; Noul either omits criteria or supplies non-null `"true"` and `"false"` descriptions. Put every rule needed to infer the answer in **the question instructions or criteria**: these fields are exported to training data. `task.state_description` explains units and plausible inputs; `task.state_schema` is a JSON Schema Draft 2020-12 constraint whose root must be a string, object, or array. Object and array states remain structured JSON in training records, never stringified. Local `#` schema references are allowed; remote references are rejected. JSON Schema `format` is an annotation here, so use structural keywords such as `pattern` to enforce string constraints. `task.labeling_guidance` helps the teacher handle ambiguity, but is **not** exported, so it must not introduce a hidden decision rule. The generator never changes the question.

The four examples cover [Choice in a simple dinosaur-runner game](../configs/synthesis/choice-dinosaur.yaml), [text Score with explicit point ranges](../configs/synthesis/score-risk.yaml), [Noul without criteria for human escalation](../configs/synthesis/noul-human-escalation.yaml), and [Noul with true/false criteria for repeat contact](../configs/synthesis/noul-repeat-contact.yaml). The Noul configs reuse the **exact fixed questions** from the [official Jev example fixture](../tests/fixtures/official_noul_escalation.request.json), but their generated customer messages are synthetic and are not official Jev examples or labels. The dinosaur rule is illustrative and fully stated in its question instructions.

## Request a label mix

Each example uses `generation.label_quotas: balanced` to schedule counts that differ by at most one. `generation.count` must cover every label. Any remainder follows the fixed candidate or level order (for Noul, `true` then `false`). To request exact Noul counts instead:

```yaml
generation:
  count: 10
  label_quotas:
    'true': 5
    'false': 5
```

For Choice, use the exact candidate keys, such as `{Jump: 5, Keep running: 5}`. For Score, use quoted zero-based level indices, such as `{'0': 4, '1': 3, '2': 3}`. Explicit maps must include every label and sum to `generation.count`; zero counts are allowed. Omitting quotas preserves unconstrained generation. `--validate-only` shows the resolved counts before any teacher request.

The runner assigns target answers to seeded slots and spreads case types within each label. The generation prompt asks for a state that truly warrants its assigned answer. The separate verifier receives the state and fixed question **without** that target or the generated answer. Target and quota metadata are never exported into the training request or state. A mismatched generated label consumes an attempt without a verifier call.

Quotas schedule attempts, not guaranteed accepted counts: invalid labels, reference-rule failures, exact duplicates, and conflicting duplicates can leave a shortfall. `report.json` distinguishes completed slots from `quota_satisfied` and lists each label's requested, accepted, and missing count. The runner does not silently refill a failed quota.

The teacher must return a JSON object containing exactly `state` and `answer`. Answers are hard labels: `{"choice":"<exact candidate key>"}`, `{"level":<zero-based integer>}`, or `{"noul":true/false}`. It may instead return `{"reject":"ambiguous"}` to decline an ambiguous sample. A response may be raw JSON or one enclosing JSON code fence. Extra prose, duplicate keys, non-finite numbers, wrong schemas, soft labels, and unknown Choice keys are rejected. A configured `generation.case_types` list steers variety (`standard`, `negation`, `boundary`, `hard_negative`); these are prompts, not semantic guarantees.

By default a second teacher call checks each state against the **same fixed question**. The verification prompt withholds the generated answer; disagreement or rejection discards that attempt. `generation.max_attempts_per_sample` bounds attempts for each slot, and `teacher.max_retries` bounds transient HTTP retry attempts for each call (timeouts, 408, 429, and 5xx). Concurrency is limited by `generation.concurrency`. Review generated states and labels yourself: agreement by the same teacher is not ground truth, and schema checks only structural validity.

For a task with a trusted deterministic rule, add a reference check. Write a Python function `check(state, answer, question) -> bool` that returns `True` only for a correct hard label; `False` rejects that attempt. The runner calls it **after** schema and blind-teacher checks, on deep copies. It never executes teacher-generated code. For the dinosaur example, a function in your trusted `my_rules.py` could be:

```python
def check_dinosaur(state, answer, question):
    expected = "Jump" if state["distance_to_obstacle"] <= 5 else "Keep running"
    return answer["choice"] == expected
```

Run `python -m jevembed.synthesis --config configs/synthesis/choice-dinosaur.yaml --output artifacts/synthesis/dinosaur --validator my_rules:check_dinosaur` with the module on your Python path. Programmatic callers can pass `sample_validator=check_dinosaur, validator_id="dinosaur-rule-v1"` to `jevembed.synthesis.run.run`. A non-boolean return or exception aborts with a sanitized error. Only select Python functions you trust; make them thread-safe or set concurrency to 1. The CLI hashes the function's source file and name, while the Python API requires a stable explicit ID. The manifest stores only an identity hash, not the source path. Resume checks the prompt and validator identities and rechecks accepted journal rows; changing either requires a new output directory. The decision rule must still appear in the fixed question so the training input is self-contained. A reference function checks only what it implements, so review remains necessary.

## Review, resume, and train

The output directory contains `data.jsonl` (accepted training records), `report.json` (counts, label distribution, rejection reasons, token usage where reported), `manifest.json` (format and config fingerprint), and `journal.jsonl` (durable per-slot progress). IDs, label targets, and case allocation are deterministic for a given config and seed. Exact duplicate state/question pairs are emitted once. If the same pair receives conflicting labels, **both labels are quarantined**. The report shows any shortfall; the runner does not pad it with lower-quality data. Its usage counts cover journaled slots only: a request in flight at interruption or a fatal error before journaling may still be billed by the teacher. The journal stores accepted state and answer content but never API key values or raw teacher responses.

If interrupted, rerun the same command with `--resume`. Resume requires the same normalized config and a valid journal; it will not silently overwrite a different job. A completed job can be resumed to regenerate its report without contacting the teacher. Use a separate output directory for a changed config.

Each JSONL row has `id`, `group`, `request: {state, questions: {<question_id>: <fixed question>}}`, and `answers: {<question_id>: <hard answer>}`. Every row from one job has `group` equal to `task.id`, so that job stays together in group-held-out validation. For example:

```json
{"id":"risk-points-demo-000000","group":"risk-points-demo","request":{"state":"risk_points: 8","questions":{"risk":{"type":"score","instructions":"Choose level 0 for 0–3, level 1 for 4–7, and level 2 for 8–10 risk points.","criteria":["Low","Medium","High"]}}},"answers":{"risk":{"level":2}}}
```

The row is an illustration of the format, not output from a specific run. Before training, review `report.json` and representative records, then run the [training validator](training.md#train) with `--train-data artifacts/synthesis/dinosaur/data.jsonl --validate-only` and your model/training configurations. Keep any validation or test data independent: variants of the same fixed question and closely related states can leak across splits. Generated labels should not be treated as human annotations or benchmark results.
