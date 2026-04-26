from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import to_parser_batch  # noqa: E402
from src.geo_parser import GeometryRelationParser, PGDP5KDataset, collate_samples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure parameter count and inference latency.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ext-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means use the whole split.")
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


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

    device = torch.device(args.device)
    model.to(device)
    total_params, trainable_params = count_parameters(model)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    warmup_done = 0
    timed_batches = 0
    total_samples = 0
    elapsed = 0.0

    with torch.no_grad():
        for raw_batch in loader:
            batch = to_parser_batch(raw_batch)
            batch = batch.__class__(
                node_features=batch.node_features.to(device),
                texts=batch.texts,
                relation_targets=None if batch.relation_targets is None else batch.relation_targets.to(device),
                atomic_targets=None if batch.atomic_targets is None else batch.atomic_targets.to(device),
                node_mask=None if batch.node_mask is None else batch.node_mask.to(device),
                knn_indices=None if batch.knn_indices is None else batch.knn_indices.to(device),
            )

            if warmup_done < args.warmup_batches:
                _ = model(batch)
                sync()
                warmup_done += 1
                continue

            if args.max_batches and timed_batches >= args.max_batches:
                break

            sync()
            start = time.perf_counter()
            outputs = model(batch)
            sync()
            elapsed += time.perf_counter() - start
            timed_batches += 1
            total_samples += outputs.logits.shape[0]

    ms_per_sample = (elapsed / max(total_samples, 1)) * 1000.0
    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "device": args.device,
        "batch_size": args.batch_size,
        "feedback_rounds": cfg.get("feedback_rounds", 2),
        "logic_weight": cfg.get("logic_weight", 0.1),
        "use_atomic_relation_bias": cfg.get("use_atomic_relation_bias", False),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_m": total_params / 1_000_000.0,
        "trainable_params_m": trainable_params / 1_000_000.0,
        "timed_batches": timed_batches,
        "timed_samples": total_samples,
        "inference_ms_per_sample": ms_per_sample,
    }
    print(json.dumps(result, indent=2))
    if args.output_path:
        Path(args.output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
