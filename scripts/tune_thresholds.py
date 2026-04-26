from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import to_parser_batch  # noqa: E402
from src.geo_parser import (  # noqa: E402
    GeometryRelationParser,
    PGDP5KDataset,
    collate_samples,
    full_relation_accuracy,
    logic_violation_rate,
)
from src.geo_parser.logic import build_relation_valid_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune scalar or per-relation thresholds on a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ext-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-threshold", type=float, default=0.35)
    parser.add_argument("--max-threshold", type=float, default=0.70)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def evaluate_thresholds(
    logits_batches: list[torch.Tensor],
    targets_batches: list[torch.Tensor],
    mask_batches: list[torch.Tensor],
    valid_batches: list[torch.Tensor],
    relation_names: tuple[str, ...],
    thresholds: torch.Tensor,
) -> tuple[float, float]:
    fra_values = []
    lvr_values = []
    for logits, targets, node_mask, valid_mask in zip(logits_batches, targets_batches, mask_batches, valid_batches):
        fra_values.append(
            full_relation_accuracy(
                logits,
                targets,
                threshold=thresholds,
                node_mask=node_mask,
                valid_mask=valid_mask,
            ).item()
        )
        lvr_values.append(
            logic_violation_rate(
                logits,
                node_mask=node_mask,
                threshold=thresholds,
                relation_names=relation_names,
            ).item()
        )
    return sum(fra_values) / max(len(fra_values), 1), sum(lvr_values) / max(len(lvr_values), 1)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    relation_names = tuple(cfg["active_relations"])

    dataset = PGDP5KDataset(
        args.data_root,
        args.split,
        max_nodes=cfg.get("max_nodes", 64),
        knn_k=cfg.get("knn_k", 8),
        active_relations=relation_names,
        ext_root=args.ext_root,
        shuffle_text=cfg.get("shuffle_text", False),
        shuffle_seed=cfg.get("seed", 42),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)

    model = GeometryRelationParser(
        input_dim=cfg.get("input_dim", 12),
        hidden_dim=cfg.get("hidden_dim", 256),
        feedback_rounds=cfg.get("feedback_rounds", 2),
        logic_weight=cfg.get("logic_weight", 0.1),
        relation_names=relation_names,
        use_neighborhood_reasoner=cfg.get("use_neighborhood_reasoner", False),
        use_atomic_relation_bias=cfg.get("use_atomic_relation_bias", False),
        disable_text_guidance=cfg.get("disable_text_guidance", False),
        disable_atomic_loss=cfg.get("disable_atomic_loss", False),
        use_global_text_fusion=cfg.get("use_global_text_fusion", False),
        feature_ablation=cfg.get("feature_ablation", "full"),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    logits_batches = []
    targets_batches = []
    mask_batches = []
    valid_batches = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = to_parser_batch(raw_batch)
            outputs = model(batch)
            logits_batches.append(outputs.logits.detach().cpu())
            targets_batches.append(batch.relation_targets.detach().cpu())
            mask_batches.append(batch.node_mask.detach().cpu())
            valid_batches.append(
                build_relation_valid_mask(
                    batch.node_features.detach().cpu(),
                    relation_names,
                    node_mask=batch.node_mask.detach().cpu(),
                )
            )

    values = []
    current = torch.full((len(relation_names),), 0.5, dtype=torch.float32)
    grid = []
    value = args.min_threshold
    while value <= args.max_threshold + 1e-8:
        grid.append(round(value, 4))
        value += args.step

    best_fra, best_lvr = evaluate_thresholds(
        logits_batches,
        targets_batches,
        mask_batches,
        valid_batches,
        relation_names,
        current,
    )

    for round_idx in range(args.rounds):
        improved = False
        for rel_idx in range(len(relation_names)):
            local_best = (best_fra, -best_lvr, current[rel_idx].item())
            local_threshold = current[rel_idx].item()
            for candidate in grid:
                trial = current.clone()
                trial[rel_idx] = candidate
                fra, lvr = evaluate_thresholds(
                    logits_batches,
                    targets_batches,
                    mask_batches,
                    valid_batches,
                    relation_names,
                    trial,
                )
                if (fra, -lvr, -abs(candidate - 0.5)) > (local_best[0], local_best[1], -abs(local_threshold - 0.5)):
                    local_best = (fra, -lvr, candidate)
                    local_threshold = candidate
            if local_threshold != current[rel_idx].item():
                current[rel_idx] = local_threshold
                best_fra = local_best[0]
                best_lvr = -local_best[1]
                improved = True
        values.append(
            {
                "round": round_idx + 1,
                "thresholds": current.tolist(),
                "val_fra": best_fra,
                "val_lvr": best_lvr,
            }
        )
        if not improved:
            break

    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "relation_names": relation_names,
        "thresholds": current.tolist(),
        "val_fra": best_fra,
        "val_lvr": best_lvr,
        "history": values,
    }
    print(json.dumps(result, indent=2))

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
