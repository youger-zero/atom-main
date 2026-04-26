from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import ATOM_FAMILIES, RELATIONS, normalize_relations
from .logic import LogicConstraintLoss, LogicLossBreakdown, build_pair_mask, build_relation_valid_mask


@dataclass
class ParserBatch:
    node_features: torch.Tensor
    texts: list[str]
    attention_mask: torch.Tensor | None = None
    node_mask: torch.Tensor | None = None
    relation_targets: torch.Tensor | None = None
    atomic_targets: torch.Tensor | None = None
    knn_indices: torch.Tensor | None = None


@dataclass
class ParserOutputs:
    logits: torch.Tensor
    probabilities: torch.Tensor
    atomic_logits: torch.Tensor
    atomic_probabilities: torch.Tensor
    text_embedding: torch.Tensor
    query: torch.Tensor
    logic: LogicLossBreakdown | None = None
    loss: torch.Tensor | None = None


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 4096, max_length: int = 128) -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length

    def __call__(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = torch.zeros(len(texts), self.max_length, dtype=torch.long, device=device)
        attention = torch.zeros_like(token_ids, dtype=torch.bool)
        for row, text in enumerate(texts):
            tokens = text.lower().split()[: self.max_length]
            for col, token in enumerate(tokens):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                token_ids[row, col] = int.from_bytes(digest, "little") % self.vocab_size
                attention[row, col] = True
        return token_ids, attention


class AtomicSemanticProbe(nn.Module):
    def __init__(self, hidden_dim: int, num_atoms: int = len(ATOM_FAMILIES), vocab_size: int = 4096) -> None:
        super().__init__()
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(hidden_dim, num_atoms)
        self.query_proj = nn.Linear(num_atoms, hidden_dim)

    def forward(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_ids, attention = self.tokenizer(texts, device=device)
        embedded = self.embedding(token_ids)
        encoded = self.encoder(embedded, src_key_padding_mask=~attention)
        pooled = masked_mean(encoded, attention)
        atomic_logits = self.classifier(pooled)
        atomic_probs = torch.sigmoid(atomic_logits)
        query = self.query_proj(atomic_probs)
        return atomic_logits, atomic_probs, pooled, query


class SemanticGuidedCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        semantic_query: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.query_proj(semantic_query).unsqueeze(1)
        attended, _ = self.attention(
            query=query,
            key=node_embeddings,
            value=node_embeddings,
            key_padding_mask=None if node_mask is None else ~node_mask,
        )
        attended = attended.expand(-1, node_embeddings.shape[1], -1)
        return self.out(torch.cat([node_embeddings, attended], dim=-1))


class NeighborhoodReasoner(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node_embeddings: torch.Tensor, knn_indices: torch.Tensor | None) -> torch.Tensor:
        if knn_indices is None or knn_indices.numel() == 0:
            return node_embeddings
        hidden_dim = node_embeddings.shape[-1]
        expanded_nodes = node_embeddings.unsqueeze(1).expand(-1, node_embeddings.shape[1], -1, -1)
        expanded_index = knn_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
        neighbors = torch.gather(expanded_nodes, 2, expanded_index)
        neighbor_mean = neighbors.mean(dim=2)
        update = self.mlp(torch.cat([node_embeddings, neighbor_mean, node_embeddings - neighbor_mean], dim=-1))
        return node_embeddings + update


class PairwiseGraphHead(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_relations),
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        left = node_embeddings.unsqueeze(2).expand(-1, -1, node_embeddings.shape[1], -1)
        right = node_embeddings.unsqueeze(1).expand(-1, node_embeddings.shape[1], -1, -1)
        pairwise = torch.cat([left, right, left * right], dim=-1)
        return self.edge_mlp(pairwise)


class FeedbackUpdater(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_relations, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, logits: torch.Tensor, node_mask: torch.Tensor | None = None) -> torch.Tensor:
        pooled = logits.mean(dim=(1, 2))
        if node_mask is not None:
            pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
            denom = pair_mask.sum(dim=(1, 2), keepdim=False).clamp_min(1).unsqueeze(-1)
            pooled = (logits * pair_mask.unsqueeze(-1)).sum(dim=(1, 2)) / denom
        return self.mlp(pooled)


class GeometryRelationParser(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        feedback_rounds: int = 2,
        logic_weight: float = 0.1,
        atomic_weight: float = 0.2,
        relation_names: tuple[str, ...] | list[str] | None = None,
        relation_pos_weight: torch.Tensor | None = None,
        invalid_logit_bias: float | None = None,
        use_neighborhood_reasoner: bool = False,
        use_atomic_relation_bias: bool = False,
        disable_text_guidance: bool = False,
        disable_atomic_loss: bool = False,
        use_global_text_fusion: bool = False,
        feature_ablation: str = "full",
    ) -> None:
        super().__init__()
        self.feedback_rounds = feedback_rounds
        self.logic_weight = logic_weight
        self.atomic_weight = atomic_weight
        self.relation_names = normalize_relations(relation_names)
        self.num_relations = len(self.relation_names)
        self.invalid_logit_bias = invalid_logit_bias
        self.use_neighborhood_reasoner = use_neighborhood_reasoner
        self.use_atomic_relation_bias = use_atomic_relation_bias
        self.disable_text_guidance = disable_text_guidance
        self.disable_atomic_loss = disable_atomic_loss
        self.use_global_text_fusion = use_global_text_fusion
        self.feature_ablation = feature_ablation

        self.visual_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.atomic_probe = AtomicSemanticProbe(hidden_dim=hidden_dim)
        self.cross_attention = SemanticGuidedCrossAttention(hidden_dim=hidden_dim)
        self.neighborhood_reasoner = NeighborhoodReasoner(hidden_dim=hidden_dim)
        self.graph_head = PairwiseGraphHead(hidden_dim=hidden_dim, num_relations=self.num_relations)
        self.feedback_updater = FeedbackUpdater(hidden_dim=hidden_dim, num_relations=self.num_relations)
        self.logic_loss = LogicConstraintLoss(relation_names=self.relation_names)
        self.atomic_to_relation = nn.Linear(len(ATOM_FAMILIES), self.num_relations)
        self.global_text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.atomic_residual_proj = nn.Linear(hidden_dim, hidden_dim)
        if relation_pos_weight is not None:
            self.register_buffer("relation_pos_weight", relation_pos_weight.float())
        else:
            self.register_buffer("relation_pos_weight", torch.ones(self.num_relations))

    def forward(self, batch: ParserBatch) -> ParserOutputs:
        node_features = batch.node_features
        if self.feature_ablation == "type_only":
            node_features = torch.cat(
                [node_features[..., :3], torch.zeros_like(node_features[..., 3:])],
                dim=-1,
            )
        elif self.feature_ablation == "zero":
            node_features = torch.zeros_like(node_features)

        node_embeddings = self.visual_encoder(node_features)
        if self.use_neighborhood_reasoner:
            node_embeddings = self.neighborhood_reasoner(node_embeddings, batch.knn_indices)
        atomic_logits, atomic_probs, text_embedding, query = self.atomic_probe(batch.texts, device=node_embeddings.device)
        if self.disable_text_guidance:
            atomic_logits = torch.zeros_like(atomic_logits)
            atomic_probs = torch.zeros_like(atomic_probs)
            text_embedding = torch.zeros_like(text_embedding)
            current_query = torch.zeros_like(query)
        elif self.use_global_text_fusion:
            current_query = self.global_text_proj(text_embedding)
        else:
            current_query = query
            node_embeddings = node_embeddings + self.atomic_residual_proj(text_embedding).unsqueeze(1)
        valid_mask = build_relation_valid_mask(
            batch.node_features,
            self.relation_names,
            node_mask=batch.node_mask,
        )
        relation_bias = self.atomic_to_relation(atomic_probs).unsqueeze(1).unsqueeze(2) if self.use_atomic_relation_bias else 0.0
        logits = None
        for round_idx in range(self.feedback_rounds):
            fused = self.cross_attention(node_embeddings, current_query, node_mask=batch.node_mask)
            logits = self.graph_head(fused) + relation_bias
            if self.invalid_logit_bias is not None:
                logits = logits.masked_fill(~valid_mask, self.invalid_logit_bias)
            if round_idx < self.feedback_rounds - 1:
                current_query = current_query + self.feedback_updater(logits, node_mask=batch.node_mask)

        assert logits is not None
        probs = torch.sigmoid(logits)

        logic_breakdown = None
        total_loss = None
        if batch.relation_targets is not None:
            relation_loss = masked_relation_bce(
                logits,
                batch.relation_targets.float(),
                batch.node_mask,
                valid_mask=valid_mask,
                pos_weight=self.relation_pos_weight,
            )
            total_loss = relation_loss
            if batch.atomic_targets is not None and not self.disable_atomic_loss:
                atomic_loss = F.binary_cross_entropy_with_logits(atomic_logits, batch.atomic_targets.float())
                total_loss = total_loss + self.atomic_weight * atomic_loss
            logic_breakdown = self.logic_loss(
                probs,
                node_mask=batch.node_mask,
                knn_indices=batch.knn_indices,
            )
            total_loss = total_loss + self.logic_weight * logic_breakdown.total

        return ParserOutputs(
            logits=logits,
            probabilities=probs,
            atomic_logits=atomic_logits,
            atomic_probabilities=atomic_probs,
            text_embedding=text_embedding,
            query=current_query,
            logic=logic_breakdown,
            loss=total_loss,
        )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    total = (values * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return total / denom


def masked_relation_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    node_mask: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    pair_mask = build_pair_mask(node_mask, logits.shape[1], logits.device).unsqueeze(-1)
    if valid_mask is not None:
        pair_mask = pair_mask & valid_mask
    loss = loss * pair_mask
    return loss.sum() / pair_mask.sum().clamp_min(1)
