"""Evaluate JevEmbed on the unchanged JevBench public suite using upstream scoring."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JevEmbedAdapter:
    name = "jevembed_local"
    price_input_per_m = None
    price_output_per_m = None
    cost_basis = "local_gpu_no_provider_tariff"

    def __init__(self, client, model_id, synchronize):
        self.client, self.model_id, self.synchronize = client, model_id, synchronize

    def reserve_estimate(self, task):
        return 0.0

    def build_request(self, task):
        from jevbench.adapters.base import build_question
        return {"model": self.model_id, "state": task.state,
                "questions": {"decision": build_question(task)}}

    def run(self, task):
        from jevbench.adapters.base import DecisionResult
        from jevembed import ValidationError
        request = self.build_request(task)
        result = DecisionResult(adapter=self.name, ok=False, model=self.model_id,
                                request_body=request, probs_source="embedding_cosine_softmax_or_sigmoid")
        self.synchronize()
        start = time.perf_counter()
        try:
            evaluated = self.client.evaluate_with_trace(request)
            response, trace = evaluated["response"], evaluated["trace"]
            answer = response["answers"]["decision"]
            if task.question["type"] == "noul":
                result.probs = {"yes": answer["noul"], "no": 1 - answer["noul"]}
            else:
                result.probs = answer["probabilities"]
            result.usage = response["usage"]
            result.raw = {"response": response, "compiled_tasks": trace["tasks"],
                          "similarities": trace["questions"],
                          "runtime": {"cache_hits": trace["cache_hits"],
                                      "encoded_inputs": trace["encoded_inputs"],
                                      "truncated_inputs": trace["truncated_inputs"],
                                      "usage_sources": trace["usage_sources"]}}
            result.ok, result.status = True, 200
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.status = 422 if isinstance(exc, ValidationError) else 502
            result.raw = {"error": result.error}
        self.synchronize()
        result.latency_s = time.perf_counter() - start
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, help="Optional local weights; enables offline loading")
    parser.add_argument("--projection-path", type=Path, help="Optional local paired-projection checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory outside the benchmark repository")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    benchmark = args.benchmark_root.resolve()
    if args.output.resolve().is_relative_to(benchmark):
        parser.error("Run outputs must be outside the JevBench repository")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new or empty; existing evidence is never overwritten")
    sys.path.insert(0, str(benchmark))
    sys.path.insert(0, str(root / "src"))
    from jevembed import JevEmbed, ModelConfig
    from jevbench.budget import Ledger
    from jevbench.runner import Runner
    from jevbench.summarize import summarize
    from jevbench.tasks import dataset_hash, load_jsonl
    from jevbench import composite_v12 as composite
    import torch

    torch.set_num_threads(args.threads)
    cfg = replace(ModelConfig.load(args.config), device=args.device, dtype=args.dtype, cache_capacity=0)
    if args.model_path:
        cfg = replace(cfg, model_name_or_path=str(args.model_path.resolve()), local_files_only=True)
    if args.projection_path:
        cfg = replace(cfg, projection_name_or_path=str(args.projection_path.resolve()), local_files_only=True)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing a silent CPU fallback")
    synchronize = (lambda: torch.cuda.synchronize(cfg.device)) if cfg.device.startswith("cuda") else (lambda: None)
    manifest = json.loads((benchmark / "datasets/manifest.json").read_text())
    tiers, hashes, canonical_hashes = {}, {}, {}
    for tier, name in [("easy", "easy"), ("standard", "original"), ("hard", "hard")]:
        path = benchmark / "datasets/public" / f"{name}.jsonl"
        tasks = load_jsonl(str(path))
        frozen = next(item for item in manifest["splits"] if item["name"] == name)
        digest, canonical = sha256(path), dataset_hash(tasks)
        if digest != frozen["sha256"] or canonical != frozen["canonical_sha256"] or len(tasks) != frozen["n"]:
            raise ValueError(f"Dataset differs from the frozen manifest: {name}")
        tiers[tier] = tasks
        hashes[str(path.relative_to(benchmark))] = digest
        canonical_hashes[tier] = canonical
    tasks = [task for group in tiers.values() for task in group]
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs in public data")
    client = JevEmbed(config=cfg)
    backend = client._models[cfg.model_id].backend
    adapter = JevEmbedAdapter(client, cfg.model_id, synchronize)
    if not cfg.device.startswith("cuda"):
        adapter.cost_basis = "local_cpu_no_provider_tariff"
    # Compilation has no access to expected labels, groups, or task provenance.
    for task in tasks:
        client.explain(adapter.build_request(task))
    args.output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    print(f"Loading {cfg.model_id}; {len(tasks)} public tasks; temperature={cfg.scoring.choice_temperature}; cache=off", flush=True)
    t0 = time.perf_counter()
    identity = backend.prepare()
    synchronize()
    load_seconds = time.perf_counter() - t0
    print(f"Model loaded in {load_seconds:.2f}s; warming up", flush=True)
    warmup = {"model": cfg.model_id, "state": "A parcel is late.", "questions": {
        "decision": {"type": "choice", "instructions": "Choose the category.",
                     "criteria": {"shipping": "Delivery delay", "billing": "Invoice question"}}}}
    for _ in range(3):
        client.evaluate(warmup)
    synchronize()
    client.clear_cache()
    if cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(cfg.device)
    print("Recording weight hashes", flush=True)
    weights = {path.name: sha256(path) for path in sorted(Path(cfg.model_name_or_path).glob("*.safetensors"))}
    if cfg.projection_name_or_path:
        projection_root = Path(cfg.projection_name_or_path)
        projection_file = projection_root if projection_root.is_file() else projection_root / cfg.projection_filename
        if projection_file.is_file():
            weights[f"projection/{projection_file.name}"] = sha256(projection_file)
    metadata = {
        "started_utc": started, "model": cfg.model_id, "config": asdict(cfg),
        "identity": identity, "model_weight_sha256": weights,
        "dataset_sha256": hashes, "dataset_canonical_sha256": canonical_hashes,
        "benchmark_manifest_sha256": sha256(benchmark / "datasets/manifest.json"),
        "benchmark_code_sha256": {str(p.relative_to(benchmark)): sha256(p) for p in (benchmark / "jevbench").rglob("*.py")},
        "jevembed_code_sha256": {str(p.relative_to(root)): sha256(p) for p in (root / "src/jevembed").rglob("*.py")},
        "runner_sha256": sha256(Path(__file__)), "model_load_seconds": load_seconds,
        "python": sys.version.split()[0], "threads": args.threads, "concurrency": 1,
        "hardware": torch.cuda.get_device_name(cfg.device) if cfg.device.startswith("cuda") else "CPU",
        "device_memory_bytes": torch.cuda.get_device_properties(cfg.device).total_memory if cfg.device.startswith("cuda") else None,
        "scope": "231 public tasks only; separate Judge tier and held-out tasks unavailable; not an official leaderboard score",
        "mapping": "Original state/instructions/criteria; upstream build_question; no prompt rewriting or label leakage",
        "scoring": "Upstream Runner and summarize; Score accuracy uses argmax, ordinal MAE uses expected level; Noul maps to yes=p/no=1-p",
        "timing": "Serial in-process evaluate_with_trace; CUDA synchronized; model loading, hashing, three synthetic warmups, and disk writes excluded; cache disabled",
        "cost": "No provider tariff; costs remain null. Zero budget reservation is bookkeeping, not a claim of free compute.",
    }
    write_json(args.output / "metadata.json", metadata)
    ledger = Ledger(str(args.output / "ledger.jsonl"), cap_usd=0)
    runner = Runner(adapter, ledger, raw_dir=args.output / "raw", default_reserve_usd=0)
    records = runner.run_all(tasks, progress_every=10, results_path=args.output / "predictions.jsonl")
    def subset(items):
        ids = {t.id for t in items}
        return [r for r in records if r["task_id"] in ids]
    overall = summarize(tasks, records)
    tier_summaries = {tier: summarize(items, subset(items)) for tier, items in tiers.items()}
    type_summaries = {kind: summarize([t for t in tasks if t.question["type"] == kind],
                                      subset([t for t in tasks if t.question["type"] == kind]))
                      for kind in ["choice", "score", "noul"]}
    by_id = {r["task_id"]: r for r in records}
    tvds = [composite.tvd(by_id[t.id]["probs"], t.provenance["gold_probs"], t.labels)
            for t in tiers["hard"] if t.provenance.get("gold_probs") and by_id.get(t.id, {}).get("probs")]
    summary = {"model": cfg.model_id, "scope": metadata["scope"], "overall": overall,
               "tiers": tier_summaries, "primitives": type_summaries,
               "hard_gold_probability_items": len(tvds), "hard_mean_tvd": sum(tvds) / len(tvds) if tvds else None,
               "official_full_score": None,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(cfg.device) if cfg.device.startswith("cuda") else None,
               "finished_utc": datetime.now(timezone.utc).isoformat(),
               "limitations": ["Public subset only, not all 534 official decisions.",
                               "No separate Judge tier or hidden data; no full composite score or price estimate.",
                               "In-process synchronized latency with no HTTP/network overhead; not comparable to hosted API latency.",
                               "Parameters fixed before evaluation; no calibration fitting on benchmark tasks."]}
    write_json(args.output / "summary.json", summary)
    print("DONE", cfg.model_id, json.dumps({"attempted": len(records), "correct": overall["n_correct"],
        "accuracy": overall["accuracy"], "valid": overall["n_valid"], "latency": overall["latency"]}), flush=True)
    return 0 if len(records) == len(tasks) and all(r["ok"] for r in records) else 3


if __name__ == "__main__":
    raise SystemExit(main())
