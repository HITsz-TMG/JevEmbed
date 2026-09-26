"""Numerical checks for question-weighted gradient accumulation."""

from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

from accelerate import Accelerator
from jevembed.training.trainer import JevTrainer


class _ToyTrainer(JevTrainer):
    def __init__(self, accelerator, accumulation, world_size=1):
        # Only exercise the inherited training primitives, without constructing
        # a tokenizer or a SentenceTransformer model for this numerical test.
        self.accelerator = accelerator
        self.args = SimpleNamespace(gradient_accumulation_steps=accumulation,
                                    world_size=world_size, n_gpu=1)
        self.model_accepts_loss_kwargs = False
        self.compute_loss_func = None
        self.optimizer = None

    def _prepare_inputs(self, inputs):
        return inputs

    def compute_loss_context_manager(self):
        return nullcontext()

    def compute_loss(self, model, inputs, **kwargs):
        return (model(inputs["x"]).flatten() - inputs["label"]).square().mean()


def _batch(values):
    x = torch.tensor(values, dtype=torch.float64).reshape(-1, 1)
    return {"x": x, "label": 2 * x.flatten() + 1}


def _reference_update(values):
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = _batch(values)
    loss = (model(batch["x"]).flatten() - batch["label"]).square().mean()
    loss.backward()
    optimizer.step()
    return model.weight.detach().clone(), loss.item()


@pytest.mark.parametrize("groups,accumulation", [
    ([[1, 2], [3]], 2),          # short second microbatch
    ([[1, 2]], 4),               # final group shorter than configured accumulation
    ([[1, 2], [3], [4]], 3),     # unequal microbatches in a full group
])
def test_accumulation_matches_full_batch_update(groups, accumulation):
    expected_weight, expected_loss = _reference_update(sum(groups, []))
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(0.5)
    accelerator = Accelerator(gradient_accumulation_steps=accumulation, cpu=True)
    trainer = _ToyTrainer(accelerator, accumulation)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches, count = trainer.get_batch_samples(iter(map(_batch, groups)), accumulation, torch.device("cpu"))
    assert count == sum(map(len, groups))
    reported_loss = sum(trainer.training_step(model, batch, count).item() for batch in batches)
    trainer.optimizer.step()
    torch.testing.assert_close(model.weight, expected_weight)
    assert reported_loss == pytest.approx(expected_loss)


class _DDPAccelerator:
    def __init__(self, accumulation):
        self.gradient_accumulation_steps = accumulation

    def reduce(self, tensor, reduction):
        assert reduction == "sum"
        torch.distributed.all_reduce(tensor)
        return tensor

    def backward(self, loss):
        (loss / self.gradient_accumulation_steps).backward()


def _ddp_worker(rank, rendezvous, result):
    torch.distributed.init_process_group("gloo", init_method=f"file://{rendezvous}",
                                         rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            model.weight.fill_(0.5)
        model = torch.nn.parallel.DistributedDataParallel(model)
        trainer = _ToyTrainer(_DDPAccelerator(2), 2, world_size=2)
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        groups = [[1, 2], [3]] if rank == 0 else [[4], [5]]
        batches, count = trainer.get_batch_samples(iter(map(_batch, groups)), 2, torch.device("cpu"))
        assert count == 5
        for batch in batches:
            trainer.training_step(model, batch, count)
        trainer.optimizer.step()
        if rank == 0:
            torch.save(model.module.weight.detach(), result)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_ddp_accumulation_matches_global_full_batch_update(tmp_path):
    expected_weight, _ = _reference_update([1, 2, 3, 4, 5])
    result = tmp_path / "weight.pt"
    torch.multiprocessing.spawn(_ddp_worker, args=(tmp_path / "rendezvous", result),
                                nprocs=2, join=True)
    torch.testing.assert_close(torch.load(result, weights_only=True), expected_weight)


def test_real_trainer_matches_full_question_batch_update(tmp_path):
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, models
    from transformers import BertConfig, BertModel, BertTokenizerFast

    from jevembed.config import ScoringConfig
    from jevembed.training.objective import JevCollator, JevLoss

    base = tmp_path / "base"
    base.mkdir()
    (base / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\nd\n")
    BertTokenizerFast(vocab_file=str(base / "vocab.txt")).save_pretrained(base)
    BertModel(BertConfig(vocab_size=9, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2,
                         hidden_dropout_prob=0, attention_probs_dropout_prob=0)).save_pretrained(base)
    model = SentenceTransformer(modules=[models.Transformer(str(base)), models.Pooling(8)], device="cpu")
    reference = deepcopy(model)
    rows = [
        {"texts": ["a", "b", "c"], "path": 0, "target_mode": 0, "target": [1., 0.]},
        {"texts": ["b", "c", "d"], "path": 0, "target_mode": 0, "target": [0., 1.]},
        {"texts": ["d", "a", "b"], "path": 0, "target_mode": 0, "target": [1., 0.]},
    ]
    scoring = ScoringConfig()
    batch = JevCollator(reference)(rows)
    features = [{key.removeprefix("sentence_"): value for key, value in batch.items()
                 if key.startswith("sentence_")}]
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    JevLoss(reference, scoring)(features, batch["label"]).backward()
    reference_optimizer.step()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    trainer = JevTrainer(
        model=model, loss=JevLoss(model, scoring), train_dataset=Dataset.from_list(rows),
        data_collator=JevCollator(model), optimizers=(optimizer, scheduler),
        args=SentenceTransformerTrainingArguments(
            output_dir=str(tmp_path / "run"), use_cpu=True, report_to=[],
            per_device_train_batch_size=2, gradient_accumulation_steps=2,
            max_steps=1, save_strategy="no", remove_unused_columns=False,
            dataloader_drop_last=False, max_grad_norm=0),
    )
    assert trainer.accelerator.gradient_accumulation_steps == 1
    assert trainer.train().global_step == 1
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
