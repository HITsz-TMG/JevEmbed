"""Offline adapter checkpoint round-trip with a tiny randomly initialized encoder."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("datasets")
pytest.importorskip("sentence_transformers")

from peft import LoraConfig, TaskType
from transformers import BertConfig, BertModel, BertTokenizerFast
from sentence_transformers import SentenceTransformer, models, SentenceTransformerTrainingArguments
from datasets import Dataset
from jevembed.config import ScoringConfig
from jevembed.training.trainer import JevTrainer
from jevembed.training.objective import JevCollator, JevLoss


def test_adapter_checkpoint_restores_parameters_and_accumulation(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    (base/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\n')
    tokenizer = BertTokenizerFast(vocab_file=str(base/'vocab.txt'))
    tokenizer.save_pretrained(base)
    BertModel(BertConfig(vocab_size=8, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2)).save_pretrained(base)
    model = SentenceTransformer(modules=[models.Transformer(str(base)),models.Pooling(8)], device='cpu')
    model.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,r=2,lora_alpha=4,
                                target_modules=['query','value']))
    rows=[{'texts':['a','b','c'],'path':0,'target_mode':0,'target':[1.,0.]}]
    trainer = JevTrainer(model=model, loss=JevLoss(model,ScoringConfig()),
                        train_dataset=Dataset.from_list(rows),data_collator=JevCollator(model),
                        args=SentenceTransformerTrainingArguments(output_dir=str(tmp_path/'run'),use_cpu=True,
                             report_to=[],gradient_accumulation_steps=2,remove_unused_columns=False))
    assert trainer.model_accepts_loss_kwargs is False
    parameters = {name:p for name,p in model.named_parameters() if p.requires_grad}
    assert parameters and all('lora_' in name for name in parameters)
    with torch.no_grad():
        for parameter in parameters.values():parameter.add_(0.1)
    expected={name:p.detach().clone() for name,p in parameters.items()}
    checkpoint=tmp_path/'checkpoint'
    trainer.save_model(str(checkpoint))
    assert (checkpoint/'adapter_model.safetensors').is_file()
    assert not (checkpoint/'model.safetensors').exists()
    with torch.no_grad():
        for parameter in parameters.values():parameter.zero_()
    trainer._load_from_checkpoint(checkpoint)
    assert all(torch.equal(p,expected[name]) for name,p in parameters.items())

    # Ten questions at batch size two produce five microbatches. Transformers
    # 4.51 computes the final fetch length from questions rather than batches;
    # with accumulation three it otherwise drops the fifth microbatch.
    trainer.train_dataset = Dataset.from_list(rows * 10)
    trainer.args.per_device_train_batch_size = 2
    trainer.args.gradient_accumulation_steps = 3
    trainer.args.num_train_epochs = 1
    trainer.args.save_strategy = 'no'
    observed_batch_sizes = []
    original_collator = trainer.data_collator
    def record_batch(batch):
        observed_batch_sizes.append(len(batch))
        return original_collator(batch)
    record_batch.valid_label_columns = original_collator.valid_label_columns
    trainer.data_collator = record_batch
    result = trainer.train()
    assert result.global_step == 2
    assert trainer.state.epoch == pytest.approx(1.0)
    assert sum(observed_batch_sizes) == 10
