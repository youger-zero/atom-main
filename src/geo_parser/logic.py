from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .constants import (
    DEFAULT_EXCLUSIVE_PAIRS,
    RELATION_TO_INDEX,
    TRANSITIVE_RELATIONS,
    UNDIRECTED_RELATIONS,
    filter_exclusive_pairs,
    normalize_relations,
)


def lukasiewicz_t_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.clamp(a + b - 1.0, min=0.0)


def build_pair_mask(node_mask: torch.Tensor | None, num_nodes: int, device: torch.device) -> torch.Tensor:
    diagonal = ~torch.eye(num_nodes, dtype=torch.bool, device=device).unsqueeze(0)
    if node_mask is None:
        return diagonal
    return (node_mask[:, :, None] & node_mask[:, None, :]) & diagonal


def build_relation_valid_mask(
    node_features: torch.Tensor,
    relation_names: Sequence[str],
    node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    num_nodes = node_features.shape[1]
    device = node_features.device
    pair_mask = build_pair_mask(node_mask, num_nodes, device)

    point_mask = node_features[..., 0] > 0.5
    line_mask = node_features[..., 1] > 0.5
    circle_mask = node_features[..., 2] > 0.5

    line_line = pair_mask & line_mask[:, :, None] & line_mask[:, None, :]
    line_circle = pair_mask & (
        (line_mask[:, :, None] & circle_mask[:, None, :])
        | (circle_mask[:, :, None] & line_mask[:, None, :])
    )
    point_any = pair_mask & (
        point_mask[:, :, None]
        | point_mask[:, None, :]
    )

    relation_masks = []
    for relation_name in relation_names:
        if relation_name == "intersect":
            relation_masks.append(line_line | line_circle)
        elif relation_name in {"parallel", "perpendicular", "bisect"}:
            relation_masks.append(line_line)
        elif relation_name == "tangent":
            relation_masks.append(line_circle)
        else:
            relation_masks.append(pair_mask & ~point_any)
    return torch.stack(relation_masks, dim=-1)


@dataclass
class LogicLossBreakdown:
    symmetry: torch.Tensor
    transitivity: torch.Tensor
    exclusivity: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.symmetry + self.transitivity + self.exclusivity


class LogicConstraintLoss(nn.Module):
    def __init__(
        self,
        relation_names: Sequence[str] | None = None,
        exclusive_pairs: Sequence[tuple[str, str]] = DEFAULT_EXCLUSIVE_PAIRS,
        transitive_relations: Iterable[str] = TRANSITIVE_RELATIONS,
        undirected_relations: Iterable[str] = UNDIRECTED_RELATIONS,
    ) -> None:
        super().__init__()
        self.relation_names = normalize_relations(relation_names)
        self.relation_to_local_index = {name: idx for idx, name in enumerate(self.relation_names)}
        self.exclusive_pairs = tuple(
            (self.relation_to_local_index[a], self.relation_to_local_index[b])
            for a, b in filter_exclusive_pairs(self.relation_names, tuple(exclusive_pairs))
        )
        self.transitive_relations = tuple(
            self.relation_to_local_index[name]
            for name in transitive_relations
            if name in self.relation_to_local_index
        )
        self.undirected_relations = tuple(
            self.relation_to_local_index[name]
            for name in undirected_relations
            if name in self.relation_to_local_index
        )

    def forward(
        self,
        relation_probs: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        knn_indices: torch.Tensor | None = None,
    ) -> LogicLossBreakdown:
        symmetry = self.symmetry_loss(relation_probs, node_mask)
        transitivity = self.transitivity_loss(relation_probs, node_mask, knn_indices)
        exclusivity = self.exclusivity_loss(relation_probs, node_mask)
        return LogicLossBreakdown(symmetry=symmetry, transitivity=transitivity, exclusivity=exclusivity)

    def symmetry_loss(self, relation_probs: torch.Tensor, node_mask: torch.Tensor | None) -> torch.Tensor:
        if not self.undirected_relations:
            return relation_probs.new_zeros(())
        probs = relation_probs[..., list(self.undirected_relations)]
        diff = (probs - probs.transpose(1, 2)).abs()
        pair_mask = build_pair_mask(node_mask, diff.shape[1], diff.device)
        diff = diff * pair_mask.unsqueeze(-1)
        denom = pair_mask.sum().clamp_min(1)
        return diff.sum() / denom

    def transitivity_loss(
        self,
        relation_probs: torch.Tensor,
        node_mask: torch.Tensor | None,
        knn_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.transitive_relations:
            return relation_probs.new_zeros(())
        total = relation_probs.new_zeros(())
        count = relation_probs.new_zeros(())
        num_nodes = relation_probs.shape[1]
        if knn_indices is None:
            base = torch.arange(num_nodes, device=relation_probs.device)
            knn_indices = base.unsqueeze(0).unsqueeze(1).expand(relation_probs.shape[0], num_nodes, num_nodes)
        pair_mask = build_pair_mask(node_mask, num_nodes, relation_probs.device)
        for relation_idx in self.transitive_relations:
            rel = relation_probs[..., relation_idx]
            left = rel.unsqueeze(3)
            right = rel.unsqueeze(1)
            premise = lukasiewicz_t_norm(left, right)
            conclusion = rel.unsqueeze(2)
            violation = torch.relu(premise - conclusion)

            triplet_mask = pair_mask.unsqueeze(3) & pair_mask.transpose(1, 2).unsqueeze(1)

            if knn_indices is not None:
                sampled = torch.zeros_like(violation, dtype=torch.bool)
                sampled.scatter_(3, knn_indices.unsqueeze(2), True)
                triplet_mask = triplet_mask & sampled

            identity = torch.eye(num_nodes, dtype=torch.bool, device=relation_probs.device)
            triplet_mask = triplet_mask & ~identity.view(1, num_nodes, 1, num_nodes)
            triplet_mask = triplet_mask & ~identity.view(1, 1, num_nodes, num_nodes)

            violation = violation * triplet_mask
            total = total + violation.sum()
            count = count + triplet_mask.sum().clamp_min(1)
        return total / count

    def exclusivity_loss(self, relation_probs: torch.Tensor, node_mask: torch.Tensor | None) -> torch.Tensor:
        if not self.exclusive_pairs:
            return relation_probs.new_zeros(())
        total = relation_probs.new_zeros(())
        pair_mask = build_pair_mask(node_mask, relation_probs.shape[1], relation_probs.device)
        for left_idx, right_idx in self.exclusive_pairs:
            pair_violation = relation_probs[..., left_idx] * relation_probs[..., right_idx]
            pair_violation = pair_violation * pair_mask
            total = total + pair_violation.sum() / pair_mask.sum().clamp_min(1)
        return total / len(self.exclusive_pairs)
