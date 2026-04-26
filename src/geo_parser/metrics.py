from __future__ import annotations

import torch

from .logic import LogicConstraintLoss, build_pair_mask


def binarize_logits(logits: torch.Tensor, threshold: float | torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    if isinstance(threshold, torch.Tensor):
        threshold_tensor = threshold.to(device=logits.device, dtype=probs.dtype)
        while threshold_tensor.dim() < probs.dim():
            threshold_tensor = threshold_tensor.unsqueeze(0)
        return (probs >= threshold_tensor).to(logits.dtype)
    return (probs >= threshold).to(logits.dtype)


def full_relation_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float | torch.Tensor = 0.5,
    node_mask: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    preds = binarize_logits(logits, threshold).to(targets.dtype)
    matches = (preds == targets).all(dim=-1)
    pair_mask = build_pair_mask(node_mask, logits.shape[1], logits.device)
    if valid_mask is not None:
        eval_pair_mask = valid_mask.any(dim=-1)
        pair_mask = pair_mask & eval_pair_mask
    matches = matches | ~pair_mask
    return matches.all(dim=(1, 2)).float().mean()


def logic_violation_rate(
    logits: torch.Tensor,
    node_mask: torch.Tensor | None = None,
    threshold: float | torch.Tensor = 0.5,
    relation_names: list[str] | tuple[str, ...] | None = None,
) -> torch.Tensor:
    binary = binarize_logits(logits, threshold).float()
    breakdown = LogicConstraintLoss(relation_names=relation_names)(binary, node_mask=node_mask)
    return breakdown.total
