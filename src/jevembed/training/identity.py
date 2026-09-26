"""Portable content identity for the base artifacts used by LoRA training."""

import hashlib
import json
from pathlib import Path
import re

from ..errors import ValidationError


_TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
    "tokenizer.model", "spiece.model", "sentencepiece.bpe.model",
}
_CONFIG_FILES = {
    "config.json", "modules.json", "config_sentence_transformers.json",
    "sentence_bert_config.json",
}


def _is_weight(path):
    name = path.name
    if name.startswith(("adapter", "optimizer", "scheduler")):
        return False
    return path.suffix in {".safetensors", ".bin"} and name.startswith(
        ("model", "pytorch_model"))


def _artifact_files(root, *, include_python):
    files = []
    weight_candidates = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in {".cache", ".git", "__pycache__"} or part.startswith("checkpoint-")
               for part in relative.parts[:-1]):
            continue
        name = path.name
        if _is_weight(path):
            weight_candidates.setdefault(path.parent, []).append(path)
        elif (name in _TOKENIZER_FILES or name in _CONFIG_FILES
              or path.suffix in {".model", ".spm", ".tiktoken"}
              or (include_python and path.suffix == ".py")):
            files.append(path)
    for directory, candidates in weight_candidates.items():
        by_name = {path.name: path for path in candidates}
        for basename, suffix in (("model.safetensors", ".safetensors"),
                                 ("pytorch_model.bin", ".bin")):
            if basename in by_name:
                files.append(by_name[basename])
                break
            index = directory / f"{basename}.index.json"
            if index.is_file():
                try:
                    names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                    files.extend(by_name[name] for name in names)
                    files.append(index)
                except (KeyError, ValueError) as exc:
                    raise ValidationError(f"Invalid checkpoint index: {index.name}") from exc
                break
            shards = [path for path in candidates if path.suffix == suffix]
            if shards:
                files.extend(shards)
                break
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _snapshot_root(model_name_or_path, resolved_revision):
    root = Path(model_name_or_path)
    if root.is_dir():
        return root
    if not resolved_revision:
        raise ValidationError("Cannot verify the loaded base model: its remote revision is unknown")
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=model_name_or_path, revision=resolved_revision,
                                      local_files_only=True))
    except Exception as exc:
        raise ValidationError("Cannot locate the loaded base model snapshot in the local cache") from exc


def base_artifact_identity(model_name_or_path, resolved_revision, *, trust_remote_code=False):
    """Hash load-relevant files without recording machine-specific paths or mtimes."""
    root = _snapshot_root(model_name_or_path, resolved_revision)
    files = _artifact_files(root, include_python=trust_remote_code)
    if not any(_is_weight(path) for path in files):
        raise ValidationError("Cannot verify the loaded base model: no checkpoint weights found")
    if not any(path.name in _TOKENIZER_FILES for path in files):
        raise ValidationError("Cannot verify the loaded base model: no tokenizer files found")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                file_digest.update(block)
        digest.update(file_digest.digest())
    return {"algorithm": "sha256-artifacts-v1", "sha256": digest.hexdigest(), "file_count": len(files)}


def portable_base_reference(public_model_name_or_path, loaded_model_name_or_path, resolved_revision):
    """Keep local machine paths and optional cache commit metadata out of the manifest."""
    public = ("local_snapshot" if Path(public_model_name_or_path).is_dir()
              else public_model_name_or_path)
    revision = None if Path(loaded_model_name_or_path).is_dir() else resolved_revision
    return public, revision


def validate_resume_manifest(manifest_path, checkpoint, output, expected, *, allow_unverified_base=False,
                             legacy_base_revision=None, legacy_base_model=None):
    """Return whether the base was verified; always reject foreign checkpoints."""
    checkpoint = Path(checkpoint)
    output = Path(output)
    if not checkpoint.is_dir() or not Path(manifest_path).is_file():
        raise ValidationError("Resume requires an existing checkpoint and the original output manifest")
    if checkpoint.resolve().parent != output.resolve() or not re.fullmatch(r"checkpoint-\d+", checkpoint.name):
        raise ValidationError("Resume checkpoint must be a checkpoint-* directory directly inside the original output")
    try:
        saved = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidationError("Cannot read the original training manifest") from exc
    if not isinstance(saved, dict) or "base_artifacts" not in saved:
        if not allow_unverified_base:
            raise ValidationError("Cannot safely resume this legacy run: its manifest has no base artifact fingerprint; "
                                  "use --allow-unverified-base-resume only if you accept that risk")
        legacy_expected = {key: value for key, value in expected.items()
                           if key not in {"base_artifacts", "model_loading"}}
        if legacy_base_revision is not None:
            legacy_expected["base_revision"] = legacy_base_revision
        if legacy_base_model is not None:
            legacy_expected["base_model"] = legacy_base_model
        if saved != legacy_expected:
            raise ValidationError("Resume configuration/data differs from the saved legacy manifest")
        return False
    if saved != expected:
        raise ValidationError("Resume configuration/data or base artifacts differ from the saved manifest")
    return True
