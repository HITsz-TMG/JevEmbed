"""Differentiable versions of JevEmbed's production scoring functions."""
import torch
from torch import nn
from torch.nn import functional as F

from .data import PATHS


class JevCollator:
    valid_label_columns = ["target"]

    def __init__(self, model):
        self.model = model

    def __call__(self, rows):
        # Flatten candidate groups without padding or cross-request negatives.
        texts, packed = [], []
        width = max(len(row["target"]) for row in rows)
        for row in rows:
            packed.append([len(texts), len(row["texts"]), row["path"], row["target_mode"], len(row["target"])]
                          + row["target"] + [0.0] * (width - len(row["target"])))
            texts.extend(row["texts"])
        # Labels first: Trainer's batch-size accounting must count questions, not embeddings.
        batch = {"label": torch.tensor(packed, dtype=torch.float32)}
        batch.update({f"sentence_{key}": value for key, value in self.model.tokenize(texts).items()})
        return batch


def task_loss(vectors, path, mode, target, config):
    target = target.to(device=vectors.device, dtype=torch.float32)
    vectors = F.normalize(vectors.float(), p=2, dim=-1)
    similarities = (vectors[1:] @ vectors[0]).clamp(-1, 1)
    if path.startswith("noul"):
        parameters = getattr(config, path)
        value = similarities[0] if path == "noul_similarity" else similarities[0] - similarities[1]
        logit = parameters.slope * value + parameters.intercept
        return F.binary_cross_entropy_with_logits(logit, target[0])
    logits = similarities / getattr(config, f"{path}_temperature")
    if mode == 1:
        expected = (logits.softmax(-1) * torch.arange(len(logits), device=logits.device)).sum()
        # Normalize for different rubric sizes; evaluate MAE in original level units.
        return ((expected - target[0]) / (len(logits) - 1)).square()
    return -(target * F.log_softmax(logits, dim=-1)).sum()


class JevLoss(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model, self.config = model, config

    def forward(self, sentence_features, labels):
        # forward(), not encode(): gradients flow through pooling into LoRA modules.
        vectors = self.model(sentence_features[0])["sentence_embedding"]
        losses = []
        for row in labels:
            offset, count, path, mode, width = [int(x) for x in row[:5].tolist()]
            losses.append(task_loss(vectors[offset:offset + count], PATHS[path], mode,
                                    row[5:5 + width], self.config))
        return torch.stack(losses).mean()
