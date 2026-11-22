"""Download pinned inference snapshots or prepare local A100 configurations.

No GPU libraries are required. Run from the repository with PyYAML installed;
the download command additionally needs huggingface-hub. It downloads model
artifacts only and does not import or execute checkpoint code.
"""

import argparse
import json
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = (
    "clm-qwen3.5-2b",
    "jeva-4b",
    "embeddinggemma-2-ft-typed-decision",
    "qnaug-embeddinggemma2-decision",
)
# Keep native PyTorch runtime files, tokenizer data and licensing material.
# Skip export directories to avoid downloading multiple copies of the weights.
INFERENCE_FILES = (
    "*.safetensors", "**/*.safetensors", "*.json", "**/*.json", "*.yaml", "**/*.yaml",
    "*.yml", "**/*.yml", "*.pt", "**/*.pt", "*.py", "**/*.py", "*.jinja", "**/*.jinja",
    "*.model", "**/*.model", "*.txt", "**/*.txt", "*.tiktoken", "*.vocab", "*.merges",
    "README*", "LICENSE*", "NOTICE*",
)


def load_config(model_id):
    source = ROOT / "configs" / f"{model_id}.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    revision = config.get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{source} must pin a full commit revision")
    return config


def download_snapshot(repo_id, revision, destination):
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(destination),
        allow_patterns=list(INFERENCE_FILES),
        ignore_patterns=["onnx/*", "openvino/*", "gguf/*", "coreml/*"],
        max_workers=4,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "local-configs"))
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_IDS, action="append",
                        help="Repeat to select models; defaults to the four deployment targets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs/local")
    args = parser.parse_args()
    weights_dir = args.weights_dir.expanduser().resolve()
    model_ids = args.model or MODEL_IDS
    if args.action == "download":
        weights_dir.mkdir(parents=True, exist_ok=True)
        for model_id in model_ids:
            config = load_config(model_id)
            root = weights_dir / model_id
            root.mkdir(parents=True, exist_ok=True)
            if model_id == "clm-qwen3.5-2b":
                download_snapshot(config["projection_name_or_path"], config["projection_revision"], root / "head")
                download_snapshot(config["model_name_or_path"], config["revision"], root / "backbone")
            else:
                download_snapshot(config["model_name_or_path"], config["revision"], root)
            print(f"Downloaded {model_id} at {config['revision']}")
        return

    # Validate every selected directory before creating any configuration.
    configs = []
    for model_id in model_ids:
        config = load_config(model_id)
        directory = weights_dir / model_id
        is_clm = model_id == "clm-qwen3.5-2b"
        is_jeva = model_id == "jeva-4b"
        is_gemma = model_id == "embeddinggemma-2-ft-typed-decision"
        is_clef = model_id == "qnaug-embeddinggemma2-decision"
        if is_clm:
            required = (directory / "head" / "clm_heads.pt", directory / "backbone" / "config.json")
            if not all(path.is_file() and path.stat().st_size for path in required):
                raise FileNotFoundError(f"Incomplete CLM assets under {directory}")
            weight_root = directory / "backbone"
        elif is_jeva:
            required = (directory / "config.yaml", directory / "head.pt",
                        directory / "backbone_adapter" / "adapter_config.json",
                        directory / "backbone_adapter" / "adapter_model.safetensors")
            if not all(path.is_file() and path.stat().st_size for path in required):
                raise FileNotFoundError(f"Incomplete Jeva assets under {directory}")
            native = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
            backbone = native["model"].get("backbone", "Qwen/Qwen3.5-4B")
            local_backbone = directory / backbone
            if (directory / "backbone_full").is_dir():
                local_backbone = directory / "backbone_full"
            weight_root = local_backbone if local_backbone.is_dir() else None
            if weight_root is None:
                print(f"{model_id}: external backbone {backbone} needs Hub access or an existing cache")
            elif not (weight_root / "config.json").is_file():
                raise FileNotFoundError(f"Missing Jeva backbone configuration under {weight_root}")
        elif is_gemma:
            required = (directory / "config.json", directory / "model.safetensors",
                        directory / "modeling_gemmadecision.py", directory / "configuration_gemmadecision.py",
                        directory / "tokenizer.json", directory / "tokenizer_config.json")
            if not all(path.is_file() and path.stat().st_size for path in required):
                raise FileNotFoundError(f"Incomplete Gemma Typed Decision assets under {directory}")
            weight_root = directory
        elif is_clef:
            required = (directory / "encoder" / "config.json", directory / "encoder" / "model.safetensors",
                        directory / "head.pt", directory / "encoder" / "tokenizer.json",
                        directory / "encoder" / "tokenizer_config.json")
            if not all(path.is_file() and path.stat().st_size for path in required):
                raise FileNotFoundError(f"Incomplete Clef Gemma assets under {directory}")
            weight_root = directory / "encoder"
        if weight_root is not None and not any(path.stat().st_size for path in weight_root.rglob("*.safetensors")):
            raise FileNotFoundError(f"No nonempty safetensors weights in {weight_root}")
        for index_file in weight_root.rglob("*.safetensors.index.json") if weight_root is not None else ():
            index = json.loads(index_file.read_text(encoding="utf-8"))
            for shard in set(index["weight_map"].values()):
                shard_path = index_file.parent / shard
                if not shard_path.is_file() or not shard_path.stat().st_size:
                    raise FileNotFoundError(f"Missing weight shard {shard_path}")
        if is_clm:
            config["model_name_or_path"] = (directory / "backbone").as_posix()
            config["projection_name_or_path"] = (directory / "head").as_posix()
        else:
            config["model_name_or_path"] = directory.as_posix()
        config["local_files_only"] = not is_jeva or weight_root is not None
        configs.append((model_id, config))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_id, config in configs:
        output = args.output_dir / f"{model_id}.local.yaml"
        output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                          encoding="utf-8")
        print(output.resolve())
    print("Local configuration files created. Run real inference to validate weights.")


if __name__ == "__main__":
    main()
