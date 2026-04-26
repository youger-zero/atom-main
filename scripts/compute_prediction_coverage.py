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
from src.geo_parser import GeometryRelationParser, PGDP5KDataset, collate_samples  # noqa: E402
from src.geo_parser.logic import build_relation_valid_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute predicted positive relation coverage statistics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ext-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-json", default=None)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def load_thresholds(args: argparse.Namespace, ckpt: dict) -> torch.Tensor | float:
    if args.threshold_json:
        payload = json.loads(Path(args.threshold_json).read_text(encoding="utf-8"))
        return torch.tensor(payload["thresholds"], dtype=torch.float32)
    if args.threshold is not None:
        return args.threshold
    return float(ckpt.get("metrics", {}).get("threshold", 0.5))


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    relation_names = tuple(cfg["active_relations"])
    thresholds = load_thresholds(args, ckpt)

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

    predicted_positive_total = 0.0
    gold_positive_total = 0.0
    sample_count = 0
    per_relation_pred = torch.zeros(len(relation_names), dtype=torch.float64)
    per_relation_gold = torch.zeros(len(relation_names), dtype=torch.float64)

    with torch.no_grad():
        for raw_batch in loader:
            batch = to_parser_batch(raw_batch)
            outputs = model(batch)
            probs = torch.sigmoid(outputs.logits.detach().cpu())
            threshold_tensor = thresholds
            if isinstance(threshold_tensor, torch.Tensor):
                while threshold_tensor.dim() < probs.dim():
                    threshold_tensor = threshold_tensor.unsqueeze(0)
            preds = (probs >= threshold_tensor).float()
            targets = batch.relation_targets.detach().cpu().float()

            valid_mask = build_relation_valid_mask(
                batch.node_features.detach().cpu(),
                relation_names,
                node_mask=batch.node_mask.detach().cpu(),
            )
            preds = preds * valid_mask
            targets = targets * valid_mask

            predicted_positive_total += preds.sum().item() / 2.0
            gold_positive_total += targets.sum().item() / 2.0
            sample_count += preds.shape[0]
            per_relation_pred += preds.sum(dim=(0, 1, 2)).double() / 2.0
            per_relation_gold += targets.sum(dim=(0, 1, 2)).double() / 2.0

    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "relation_names": list(relation_names),
        "thresholds": thresholds.tolist() if isinstance(thresholds, torch.Tensor) else float(thresholds),
        "avg_predicted_positive_relations_per_sample": predicted_positive_total / max(sample_count, 1),
        "avg_gold_positive_relations_per_sample": gold_positive_total / max(sample_count, 1),
        "per_relation_avg_predicted": {
            relation_names[idx]: float(per_relation_pred[idx].item() / max(sample_count, 1))
            for idx in range(len(relation_names))
        },
        "per_relation_avg_gold": {
            relation_names[idx]: float(per_relation_gold[idx].item() / max(sample_count, 1))
            for idx in range(len(relation_names))
        },
    }
    print(json.dumps(result, indent=2))
    if args.output_path:
        Path(args.output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
