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
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a dataset split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ext-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-json", default=None)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    relation_names = tuple(cfg["active_relations"])

    if args.threshold_json:
        threshold_payload = json.loads(Path(args.threshold_json).read_text(encoding="utf-8"))
        thresholds = torch.tensor(threshold_payload["thresholds"], dtype=torch.float32)
    elif args.threshold is not None:
        thresholds = args.threshold
    else:
        thresholds = float(ckpt.get("metrics", {}).get("threshold", 0.5))

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

    fra_values = []
    lvr_values = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = to_parser_batch(raw_batch)
            outputs = model(batch)
            valid_mask = build_relation_valid_mask(
                batch.node_features.detach().cpu(),
                relation_names,
                node_mask=batch.node_mask.detach().cpu(),
            )
            fra_values.append(
                full_relation_accuracy(
                    outputs.logits.detach().cpu(),
                    batch.relation_targets.detach().cpu(),
                    threshold=thresholds,
                    node_mask=batch.node_mask.detach().cpu(),
                    valid_mask=valid_mask,
                ).item()
            )
            lvr_values.append(
                logic_violation_rate(
                    outputs.logits.detach().cpu(),
                    node_mask=batch.node_mask.detach().cpu(),
                    threshold=thresholds,
                    relation_names=relation_names,
                ).item()
            )

    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "relation_names": relation_names,
        "thresholds": thresholds.tolist() if isinstance(thresholds, torch.Tensor) else thresholds,
        "fra": sum(fra_values) / max(len(fra_values), 1),
        "lvr": sum(lvr_values) / max(len(lvr_values), 1),
    }
    print(json.dumps(result, indent=2))

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
