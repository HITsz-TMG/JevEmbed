"""Run the five deployment examples in isolated processes and check their outputs.

Each environment is selected explicitly because the model groups require
different Transformers versions. --explain checks compilation only, without
loading weights; the report records this distinction.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    ("clm-qwen3.5-2b", "clm_qwen35_decide.json", "decision"),
    ("embeddinggemma-2-ft-typed-decision", "embeddinggemma_decide.json", "gemma"),
    ("jeva-4b", "jeva_decide.json", "decision"),
    ("qnaug-embeddinggemma2-decision", "qnaug_embeddinggemma_decide.json", "gemma"),
    ("nemotron-embed-1b-v2", "nemotron_decide.json", "standard"),
)


def validate_response(response, request):
    if response.get("model") != request["model"]:
        raise ValueError("Response model does not match the requested model")
    if set(response.get("answers", {})) != set(request["questions"]):
        raise ValueError("Response must answer every submitted question")
    for qid, question in request["questions"].items():
        answer = response["answers"][qid]
        kind = question["type"]
        if answer.get("type") != kind:
            raise ValueError(f"{qid}: incorrect answer type")
        if kind == "noul":
            values = [answer["noul"]]
        else:
            distribution = answer["probabilities"]
            labels = set(question["criteria"]) if kind == "choice" else {
                str(i) for i in range(len(question["criteria"]))}
            if set(distribution) != labels:
                raise ValueError(f"{qid}: incorrect probability labels")
            values = list(distribution.values())
            if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
                raise ValueError(f"{qid}: probabilities do not sum to one")
            if kind == "choice" and answer["choice"] not in labels:
                raise ValueError(f"{qid}: invalid chosen label")
            if kind == "score":
                score = answer["score"]
                if not math.isfinite(score) or not 0 <= score <= len(labels) - 1:
                    raise ValueError(f"{qid}: invalid score")
        if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError(f"{qid}: invalid probability")
    usage = response["usage"]
    if type(usage["input_tokens"]) is not int or usage["input_tokens"] <= 0 or usage["output_tokens"] != 0:
        raise ValueError("Expected positive input token usage and zero generated tokens")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for group, suffix in (("decision", "decision"), ("gemma", "gemma"), ("standard", "standard")):
        parser.add_argument(f"--{group}-python", type=Path,
                            help=f"Python executable for the {suffix} model environment")
    parser.add_argument("--model", choices=[entry[0] for entry in MODELS], action="append")
    parser.add_argument("--configs-dir", type=Path, default=ROOT / "configs/local")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / (
        "a100-smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"mode": "compile_only" if args.explain else "real_inference",
              "started_at": datetime.now(timezone.utc).isoformat(), "models": []}
    for model_id, example, group in MODELS:
        if args.model and model_id not in args.model:
            continue
        row = {"model": model_id, "status": "failed"}
        try:
            config = args.configs_dir.resolve() / f"{model_id}.local.yaml"
            # A caller can explicitly select a directory of public configs.
            if not config.is_file():
                config = args.configs_dir.resolve() / f"{model_id}.yaml"
            if (not config.is_file() and args.explain and
                    args.configs_dir.resolve() == ROOT / "configs/local"):
                config = ROOT / "configs" / f"{model_id}.yaml"
            if not config.is_file():
                raise FileNotFoundError(config)
            python_arg = getattr(args, f"{group}_python")
            if python_arg is None:
                raise ValueError(f"Pass --{group}-python for the {group} model environment")
            python = python_arg.resolve()
            output = args.output_dir.resolve() / f"{model_id}.json"
            command = [str(python), "-m", "jevembed", "--config", str(config),
                       "--input", str(ROOT / "examples" / example), "--output", str(output)]
            if args.explain:
                command.append("--explain")
            row.update(python=str(python), config=str(config), output=str(output))
            # Prevent stale output from being counted as a successful run.
            output.unlink(missing_ok=True)
            log = args.output_dir / f"{model_id}.log"
            try:
                process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                         encoding="utf-8", errors="replace", timeout=args.timeout)
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                log.write_text(stdout + "\n" + stderr, encoding="utf-8")
                raise RuntimeError(f"Inference timed out after {args.timeout}s; see {log}") from exc
            log.write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
            if process.returncode:
                raise RuntimeError(f"CLI exit {process.returncode}; see {log}")
            result = json.loads(output.read_text(encoding="utf-8"))
            request = json.loads((ROOT / "examples" / example).read_text(encoding="utf-8"))
            if args.explain:
                if result.get("model") != model_id or len(result.get("tasks", [])) != len(request["questions"]):
                    raise ValueError("Explain result does not match request")
            else:
                validate_response(result, request)
            row["status"] = "compiled" if args.explain else "passed"
        except Exception as exc:
            row["error"] = str(exc)
        report["models"].append(row)
        print(f"{model_id}: {row['status']}", flush=True)
        if "error" in row:
            print(row["error"], flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["ok"] = all(row["status"] != "failed" for row in report["models"])
    summary = args.output_dir / "summary.json"
    summary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.resolve())
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
