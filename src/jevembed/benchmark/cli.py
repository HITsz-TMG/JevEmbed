"""Command-line interface for JevEmbed benchmarks."""

import argparse
import json
from pathlib import Path
import sys

from ..config import ModelConfig
from ..errors import JevEmbedError
from .run import run_benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a versioned JevEmbed benchmark")
    parser.add_argument("--benchmark", type=Path, required=True, help="Benchmark manifest JSON")
    parser.add_argument("--config", action="append", required=True,
                        help="Model YAML; repeat to evaluate multiple models")
    parser.add_argument("--output", type=Path, default=Path("results"),
                        help="Result root directory (default: results)")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Also write per-case predictions.jsonl")
    parser.add_argument("--limit", type=int, default=0,
                        help="Smoke-test the first N cases; zero evaluates all cases")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing results for the selected benchmark and models")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    try:
        configs = [ModelConfig.load(path) for path in args.config]
        summary = run_benchmark(args.benchmark, configs, args.output,
                                save_predictions=args.save_predictions, limit=args.limit,
                                overwrite=args.overwrite)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (JevEmbedError, ValueError, OSError, ImportError) as exc:
        print(f"jevembed-benchmark: {exc}", file=sys.stderr)
        return 2
