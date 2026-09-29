# Benchmarking

JevEmbed benchmarks use a small, versioned JSON manifest plus JSONL cases. Requests use the normal JevEmbed schema but omit `model`, so the same cases can be evaluated with any configuration.

## Files

`benchmark.json` identifies the benchmark and its data file or files:

```json
{
  "schema_version": "jevembed-benchmark-v1",
  "name": "example-decisions",
  "version": "1.0",
  "split": "test",
  "description": "Small Choice, Score, and Noul example.",
  "data": "cases.jsonl"
}
```

For sharded data, `data` can be a nonempty array. Files are read in listed order:

```json
"data": ["cases-00000.jsonl", "cases-00001.jsonl", "cases-00002.jsonl"]
```

All paths are relative to the manifest, must stay in its directory, and must be unique JSONL files.

Each line of `cases.jsonl` contains:

- `id`: unique case ID;
- `group`: optional group ID;
- `request`: a JevEmbed request without `model`;
- `answers`: one target for every question in the request;
- `metadata`: optional JSON object, ignored during inference.

Targets follow the task type:

| Type | Hard target | Soft or continuous target |
| --- | --- | --- |
| Choice | `{"choice":"label"}` | `{"probabilities":{"label":0.8,"other":0.2}}` |
| Score | `{"level":2}` | `{"score":1.7}` or a probability distribution over string indices |
| Noul | `{"noul":true}` | `{"noul":0.8}` |

Score probability keys are zero-based string indices such as `"0"`, `"1"`, and `"2"`. Probabilities must include every candidate or level and sum to one.

See the runnable files in [`examples/benchmark`](../examples/benchmark).

## Run

```bash
jevembed-benchmark \
  --benchmark examples/benchmark/benchmark.json \
  --config configs/jevembed-qwen3-embedding-0.6b.yaml \
  --output results \
  --save-predictions
```

Repeat `--config` to compare models. Use `--limit N` for a smoke test and `--overwrite` to replace an existing result for the same benchmark and model.

The same runner is available in Python:

```python
from jevembed.benchmark import run_benchmark
from jevembed.config import ModelConfig

summary = run_benchmark(
    "examples/benchmark/benchmark.json",
    [ModelConfig.load("configs/jevembed-qwen3-embedding-0.6b.yaml")],
    "results",
    save_predictions=True,
)
```

## Results

The runner writes a stable directory for each model:

```text
results/example-decisions/
├── summary.json
└── jevembed-qwen3-embedding-0.6b/
    ├── results.json
    └── predictions.jsonl
```

`results.json` records benchmark and configuration hashes, task counts, token usage, timing, and the following metrics with their denominators:

- hard-label accuracy overall and for Choice, Score, and Noul;
- Score and Noul mean absolute error;
- Brier score and total variation distance for target distributions.

`predictions.jsonl` is written only with `--save-predictions`. Result files contain model and benchmark identities but no local filesystem paths.
