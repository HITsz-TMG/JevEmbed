"""Held-out task metrics using the unchanged inference scoring path."""
import torch
import torch.distributed as dist
from ..scoring import normalize_vectors, normalized_entropy, score_plan
from .objective import task_loss


def evaluate(model, examples, config):
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    was_training = model.training
    model.eval()
    losses, choice, levels, score_errors, noul_errors, noul_hard = [], [], [], [], [], []
    brier, tvd = [], []
    try:
        with torch.no_grad():
            for index in range(rank, len(examples), world_size):
                example = examples[index]
                vectors = model.encode(example.texts, prompt="", batch_size=config.batch_size,
                                       convert_to_tensor=True, normalize_embeddings=True,
                                       show_progress_bar=False)
                target = torch.tensor(example.target, device=vectors.device, dtype=torch.float32)
                losses.append(task_loss(vectors, example.plan.path, example.target_mode,
                                        target, config.scoring).item())
                normalized = normalize_vectors(vectors.float().cpu().tolist(), len(example.texts))
                answer, _ = score_plan(example.plan, normalized, config.scoring, normalized_entropy)
                if example.plan.kind == "noul":
                    gold = example.target[0]
                    brier.append(2 * (answer["noul"] - gold) ** 2)
                    tvd.append(abs(answer["noul"] - gold))
                    noul_errors.append(abs(answer["noul"] - gold))
                    if gold in (0.0, 1.0):
                        noul_hard.append((answer["noul"] >= 0.5) == bool(gold))
                elif example.plan.kind == "score":
                    gold = example.target[0] if example.target_mode == 1 else sum(i*p for i,p in enumerate(example.target))
                    score_errors.append(abs(answer["score"] - gold))
                    if example.target_mode == 0 and max(example.target) == 1.0:
                        prediction = max(answer["probabilities"], key=answer["probabilities"].get)
                        levels.append(int(prediction) == example.target.index(1.0))
                elif max(example.target) == 1.0:
                    choice.append(answer["choice"] == example.plan.labels[example.target.index(1.0)])
                if example.plan.kind != "noul" and example.target_mode == 0:
                    probabilities = [answer["probabilities"][label] for label in example.plan.labels]
                    brier.append(sum((p-q)**2 for p,q in zip(probabilities, example.target)))
                    tvd.append(sum(abs(p-q) for p,q in zip(probabilities, example.target)) / 2)
    finally:
        model.train(was_training)
    metrics = [("loss", losses), ("choice_accuracy", choice), ("score_level_accuracy", levels),
               ("score_mae", score_errors), ("noul_mae", noul_errors),
               ("noul_binary_accuracy", noul_hard), ("target_distribution_brier", brier),
               ("target_distribution_tvd", tvd)]
    totals = torch.tensor([[sum(values), len(values)] for _, values in metrics],
                          dtype=torch.float64, device=next(model.parameters()).device)
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    result = {"questions": len(examples)}
    for (name, _), (total, count) in zip(metrics, totals.tolist()):
        n = int(count)
        result[name] = total / n if n else None
        if name != "loss":
            result[name + "_n"] = n
    return result
