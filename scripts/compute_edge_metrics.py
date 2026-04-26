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
    parser = argparse.ArgumentParser(description="Compute edge-level and per-class metrics for a checkpoint.")
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

    tp = torch.zeros(len(relation_names), dtype=torch.float64)
    fp = torch.zeros(len(relation_names), dtype=torch.float64)
    fn = torch.zeros(len(relation_names), dtype=torch.float64)

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
            pair_mask = (batch.node_mask[:, :, None] & batch.node_mask[:, None, :]).detach().cpu()
            eye = torch.eye(pair_mask.shape[1], dtype=torch.bool).unsqueeze(0)
            pair_mask = pair_mask & ~eye
            valid_mask = build_relation_valid_mask(
                batch.node_features.detach().cpu(),
                relation_names,
                node_mask=batch.node_mask.detach().cpu(),
            )
            eval_mask = pair_mask.unsqueeze(-1).expand_as(preds) & valid_mask

            for rel_idx in range(len(relation_names)):
                rel_mask = eval_mask[..., rel_idx]
                rel_preds = preds[..., rel_idx][rel_mask]
                rel_targets = targets[..., rel_idx][rel_mask]
                tp[rel_idx] += ((rel_preds == 1) & (rel_targets == 1)).sum()
                fp[rel_idx] += ((rel_preds == 1) & (rel_targets == 0)).sum()
                fn[rel_idx] += ((rel_preds == 0) & (rel_targets == 1)).sum()

    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / torch.clamp(tp + fn, min=1.0)
    f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-12)

    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_precision = micro_tp / max(float(micro_tp + micro_fp), 1.0)
    micro_recall = micro_tp / max(float(micro_tp + micro_fn), 1.0)
    micro_f1 = 2 * micro_precision * micro_recall / max(float(micro_precision + micro_recall), 1e-12)
    macro_f1 = float(f1.mean().item())

    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "relation_names": list(relation_names),
        "thresholds": thresholds.tolist() if isinstance(thresholds, torch.Tensor) else float(thresholds),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "per_class": {
            relation_names[idx]: {
                "precision": float(precision[idx].item()),
                "recall": float(recall[idx].item()),
                "f1": float(f1[idx].item()),
                "tp": float(tp[idx].item()),
                "fp": float(fp[idx].item()),
                "fn": float(fn[idx].item()),
            }
            for idx in range(len(relation_names))
        },
    }
    print(json.dumps(result, indent=2))
    if args.output_path:
        Path(args.output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
