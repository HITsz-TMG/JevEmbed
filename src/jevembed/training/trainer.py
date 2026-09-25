"""Sentence Transformers integration with adapter-only checkpoints."""
import math
from pathlib import Path

from sentence_transformers import SentenceTransformerTrainer


class JevTrainer(SentenceTransformerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # JevLoss returns a microbatch mean and does not consume num_items_in_batch.
        # Transformers must divide by the gradient-accumulation count itself.
        self.model_accepts_loss_kwargs = False

    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        if self.args.world_size > 1:
            # SentenceTransformerTrainer disables Accelerate's even-batch
            # padding. That can leave the last rank with fewer microbatches,
            # making DDP hang on the final backward pass.
            sampler = getattr(dataloader, "batch_sampler", None)
            if not hasattr(sampler, "even_batches"):
                raise RuntimeError("Distributed Jev training requires an even-batch sampler")
            sampler.even_batches = True
        return dataloader

    def set_initial_training_values(self, args, dataloader, total_train_batch_size):
        values = list(super().set_initial_training_values(args, dataloader, total_train_batch_size))
        # Transformers 4.51 floors this count, ending epoch-based runs before
        # their final partial accumulation group. Count that update as well.
        if values[5]:
            updates = math.ceil(values[5] / args.gradient_accumulation_steps)
            values[1] = updates
            if values[4]:
                values[6] = math.ceil(args.num_train_epochs * updates)
            else:
                values[0] = math.ceil(values[6] / updates)
        return tuple(values)

    def add_model_card_callback(self, default_args_dict):
        pass  # Do not publish raw examples or local paths in automatic model cards.

    def _save(self, output_dir=None, state_dict=None):
        folder = Path(output_dir or self.args.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        self.model[0].auto_model.save_pretrained(folder, safe_serialization=True)

    def _load_from_checkpoint(self, checkpoint_path, *unused, **kwargs):
        from peft import load_peft_weights, set_peft_model_state_dict
        weights = load_peft_weights(str(checkpoint_path), device="cpu")
        # Transformers saves adapters with a PEFT wrapper prefix even when using
        # its native adapter mixin; mirror PeftAdapterMixin.load_adapter.
        weights = {key.removeprefix("base_model.model."): value for key, value in weights.items()}
        result = set_peft_model_state_dict(self.model[0].auto_model, weights, adapter_name="default")
        missing = [key for key in result.missing_keys if "lora_" in key]
        if missing or result.unexpected_keys:
            raise ValueError("Checkpoint adapter tensors do not match this model")
