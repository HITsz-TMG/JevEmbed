"""LoRA fine-tuning with SentenceTransformerTrainer and Jev task supervision."""
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path

from ..config import ModelConfig
from ..errors import ValidationError
from .data import check_disjoint, check_lengths, load_examples


@dataclass(frozen=True)
class TrainingConfig:
    epochs: float = 1.0
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_steps: int = -1
    max_input_tokens: int = 1024
    overflow_policy: str = "error"
    gradient_checkpointing: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    threads: int = 4

    def __post_init__(self):
        for name in ("batch_size", "gradient_accumulation_steps", "max_input_tokens", "lora_rank",
                     "lora_alpha", "logging_steps", "save_steps", "save_total_limit", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValidationError(f"{name} must be a positive integer")
        for name in ("epochs", "learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValidationError(f"{name} must be positive and finite")
        for name in ("warmup_ratio", "lora_dropout", "weight_decay"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValidationError(f"{name} must be in [0, 1)")
        if type(self.max_steps) is not int or self.max_steps == 0 or self.max_steps < -1:
            raise ValidationError("max_steps must be -1 or a positive integer")
        if type(self.seed) is not int or self.seed < 0 or type(self.gradient_checkpointing) is not bool:
            raise ValidationError("Invalid seed or gradient_checkpointing")
        if self.overflow_policy not in ("error", "truncate"):
            raise ValidationError("overflow_policy must be error or truncate")
        if self.target_modules is not None and (not isinstance(self.target_modules, list) or not self.target_modules
                or any(not isinstance(x, str) or not x for x in self.target_modules)):
            raise ValidationError("target_modules must be a nonempty list of module suffixes")


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def select_targets(model, requested):
    from torch.nn import Linear
    names = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, Linear)}
    targets = requested
    if targets is None:
        targets = next((group for group in (["q_proj", "v_proj"], ["query", "value"]) if set(group) <= names), None)
    if not targets or not set(targets) <= names:
        raise ValidationError("Cannot resolve LoRA targets; specify existing Linear module suffixes in target_modules")
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JevEmbed model configuration")
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, help="Use local base weights without modifying a public config")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--validate-only", action="store_true", help="Check supervision and split overlap without model loading")
    args = parser.parse_args(argv)
    import yaml
    raw = yaml.safe_load(args.training_config.read_text())
    if not isinstance(raw, dict):
        parser.error("Training configuration must be a mapping")
    try:
        settings = TrainingConfig(**raw)
    except TypeError as exc:
        parser.error(str(exc))
    config = ModelConfig.load(args.config)
    if config.backend != "sentence_transformers" or config.adapter_name_or_path:
        parser.error("Training requires a base SentenceTransformer configuration; use --resume-from-checkpoint to resume")
    train = load_examples(args.train_data, config)
    validation = load_examples(args.eval_data, config) if args.eval_data else []
    check_disjoint(train, validation)
    if args.validate_only:
        print(json.dumps({"train_questions": len(train), "validation_questions": len(validation),
                          "status": "valid; token lengths not checked without model loading"}))
        return 0
    if args.output.exists() and any(args.output.iterdir()) and not args.resume_from_checkpoint:
        parser.error("Output directory is not empty; choose a new run or explicitly resume a checkpoint")
    if args.device == "cpu" and args.dtype != "float32":
        parser.error("CPU training requires --dtype float32")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from sentence_transformers import SentenceTransformerTrainingArguments
    from transformers import set_seed
    from ..backends import SentenceTransformersBackend
    from .objective import JevCollator, JevLoss
    from .evaluation import evaluate
    from .trainer import JevTrainer

    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1 or (args.device == "cuda" and torch.cuda.device_count() > 1):
        parser.error("This trainer supports one device per process; select one GPU with CUDA_VISIBLE_DEVICES")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.device == "cuda" and args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("BF16 is unavailable; use --dtype float32")
    torch.set_num_threads(settings.threads)
    set_seed(settings.seed)
    local_config = replace(config, device=args.device, dtype=args.dtype, cache_capacity=0,
                           overflow_policy=settings.overflow_policy)
    if args.model_path:
        local_config = replace(local_config, model_name_or_path=str(args.model_path.resolve()), local_files_only=True)
    backend = SentenceTransformersBackend(config=local_config)
    print(f"Loading {config.model_id} for LoRA training", flush=True)
    identity = backend.prepare()
    model = backend._model
    model.max_seq_length = min(model.max_seq_length, settings.max_input_tokens)
    print(f"Checking all {len(train)} train and {len(validation)} validation questions; "
          f"token limit={model.max_seq_length}", flush=True)
    lengths = {"train": check_lengths(model, train, settings.overflow_policy)}
    if validation:
        lengths["validation"] = check_lengths(model, validation, settings.overflow_policy)
    targets = select_targets(model, settings.target_modules)
    model.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=settings.lora_rank,
                                lora_alpha=settings.lora_alpha, lora_dropout=settings.lora_dropout,
                                target_modules=targets, bias="none"))
    if hasattr(model[0].auto_model.config, "use_cache"):
        model[0].auto_model.config.use_cache = False
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Expected only LoRA parameters to be trainable")
    for _, parameter in trainable:
        parameter.data = parameter.data.float()
    # Keep PEFT metadata portable even when the base was loaded from a local snapshot.
    model[0].auto_model.peft_config["default"].base_model_name_or_path = config.model_name_or_path
    manifest = {"training": asdict(settings), "model_id": config.model_id, "scoring": asdict(config.scoring),
                "prompts": asdict(config.prompts), "target_modules": targets, "lengths": lengths,
                "base_revision": identity["resolved_revision"], "base_model": config.model_name_or_path,
                "dataset_sha256": {"train": hashlib.sha256(args.train_data.read_bytes()).hexdigest(),
                                   "validation": hashlib.sha256(args.eval_data.read_bytes()).hexdigest() if args.eval_data else None},
                "dtype": args.dtype, "trainable_parameters": sum(p.numel() for _, p in trainable),
                "total_parameters": sum(p.numel() for p in model.parameters())}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "training_manifest.json"
    if args.resume_from_checkpoint:
        if not args.resume_from_checkpoint.is_dir() or not manifest_path.exists():
            parser.error("Resume requires an existing checkpoint and the original output manifest")
        if json.loads(manifest_path.read_text()) != manifest:
            parser.error("Resume configuration/data differs from the saved manifest")
    else:
        write_json(manifest_path, manifest)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output), num_train_epochs=settings.epochs, max_steps=settings.max_steps,
        per_device_train_batch_size=settings.batch_size, per_device_eval_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate, warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay, max_grad_norm=settings.max_grad_norm,
        bf16=args.dtype == "bfloat16", fp16=False, use_cpu=args.device == "cpu",
        gradient_checkpointing=settings.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch" if validation else "no", save_strategy="steps", save_steps=settings.save_steps,
        save_total_limit=settings.save_total_limit, logging_steps=settings.logging_steps,
        report_to=[], push_to_hub=False, seed=settings.seed, data_seed=settings.seed,
        dataloader_num_workers=0, remove_unused_columns=False)
    trainer = JevTrainer(model=model, args=training_args,
                      train_dataset=Dataset.from_list([e.training_row() for e in train]),
                      eval_dataset=Dataset.from_list([e.training_row() for e in validation]) if validation else None,
                      loss=JevLoss(model, config.scoring), data_collator=JevCollator(model))
    print("Evaluating the base model on validation data" if validation else "No validation data supplied", flush=True)
    baseline = evaluate(model, validation, local_config) if validation else None
    if baseline is not None:
        write_json(args.output / "baseline_metrics.json", baseline)
    print(f"Starting optimization: batch_size={settings.batch_size}, "
          f"gradient_accumulation_steps={settings.gradient_accumulation_steps}, epochs={settings.epochs}", flush=True)
    train_result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    adapter_dir = args.output / "adapter"
    trainer.save_model(str(adapter_dir))
    trainer.save_state()
    after = evaluate(model, validation, local_config) if validation else None
    write_json(args.output / "metrics.json", {"train": train_result.metrics,
               "validation_base": baseline, "validation_after": after,
               "note": "Validation uses the configured training token limit; no held-out claims without eval-data."})
    inference = asdict(replace(config, adapter_name_or_path=str(adapter_dir), adapter_revision=None,
                              cache_capacity=0, max_input_tokens=model.max_seq_length,
                              overflow_policy=settings.overflow_policy))
    (args.output / "inference.yaml").write_text(yaml.safe_dump(inference, sort_keys=False), encoding="utf-8")
    print(json.dumps({"adapter": str(adapter_dir), "trainable_parameters": manifest["trainable_parameters"],
                      "validation_after": after}))
    return 0
