from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.geo_parser import (
    GeometryRelationParser,
    PGDP5KDataset,
    ParserBatch,
    RELATIONS,
    collate_samples,
    full_relation_accuracy,
    logic_violation_rate,
)
from src.geo_parser.constants import normalize_relations
from src.geo_parser.logic import build_relation_valid_mask


@dataclass
class TrainConfig:
    data_root: str = "data/PGDP5K"
    ext_root: str | None = None
    input_dim: int = 12
    hidden_dim: int = 256
    batch_size: int = 8
    epochs: int = 2
    lr: float = 1e-3
    logic_weight: float = 0.1
    feedback_rounds: int = 2
    max_nodes: int = 64
    knn_k: int = 8
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs/default"
    save_every_epoch: bool = True
    grad_clip: float = 1.0
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    early_stop_patience: int = 6
    seed: int = 42
    active_relations: tuple[str, ...] = RELATIONS
    threshold_search: bool = False
    use_neighborhood_reasoner: bool = False
    use_atomic_relation_bias: bool = False
    disable_text_guidance: bool = False
    disable_atomic_loss: bool = False
    use_global_text_fusion: bool = False
    feature_ablation: str = "full"
    shuffle_text: bool = False


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the geometry relation parser on PGDP5K.")
    parser.add_argument("--data-root", default="data/PGDP5K")
    parser.add_argument("--input-dim", type=int, default=12)
    parser.add_argument("--ext-root", default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--logic-weight", type=float, default=0.1)
    parser.add_argument("--feedback-rounds", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/default")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--active-relations", default="auto")
    parser.add_argument("--threshold-search", action="store_true")
    parser.add_argument("--use-neighborhood-reasoner", action="store_true")
    parser.add_argument("--use-atomic-relation-bias", action="store_true")
    parser.add_argument("--disable-text-guidance", action="store_true")
    parser.add_argument("--disable-atomic-loss", action="store_true")
    parser.add_argument("--use-global-text-fusion", action="store_true")
    parser.add_argument("--feature-ablation", choices=["full", "type_only", "zero"], default="full")
    parser.add_argument("--shuffle-text", action="store_true")
    parser.add_argument("--no-save-every-epoch", action="store_true")
    args = parser.parse_args()
    active_relations = tuple()
    if args.active_relations != "auto":
        active_relations = normalize_relations([part.strip() for part in args.active_relations.split(",")])
    return TrainConfig(
        data_root=args.data_root,
        ext_root=args.ext_root,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        logic_weight=args.logic_weight,
        feedback_rounds=args.feedback_rounds,
        max_nodes=args.max_nodes,
        knn_k=args.knn_k,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
        save_every_epoch=not args.no_save_every_epoch,
        grad_clip=args.grad_clip,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_lr=args.min_lr,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        active_relations=active_relations or RELATIONS,
        threshold_search=args.threshold_search,
        use_neighborhood_reasoner=args.use_neighborhood_reasoner,
        use_atomic_relation_bias=args.use_atomic_relation_bias,
        disable_text_guidance=args.disable_text_guidance,
        disable_atomic_loss=args.disable_atomic_loss,
        use_global_text_fusion=args.use_global_text_fusion,
        feature_ablation=args.feature_ablation,
        shuffle_text=args.shuffle_text,
    )


def to_parser_batch(raw_batch: dict[str, torch.Tensor | list[str]]) -> ParserBatch:
    return ParserBatch(
        node_features=raw_batch["node_features"],
        texts=raw_batch["texts"],
        relation_targets=raw_batch["relation_targets"],
        atomic_targets=raw_batch["atomic_targets"],
        node_mask=raw_batch["node_mask"],
        knn_indices=raw_batch["knn_indices"],
    )


def move_batch_to_device(batch: ParserBatch, device: torch.device) -> ParserBatch:
    return ParserBatch(
        node_features=batch.node_features.to(device),
        texts=batch.texts,
        relation_targets=None if batch.relation_targets is None else batch.relation_targets.to(device),
        atomic_targets=None if batch.atomic_targets is None else batch.atomic_targets.to(device),
        node_mask=None if batch.node_mask is None else batch.node_mask.to(device),
        knn_indices=None if batch.knn_indices is None else batch.knn_indices.to(device),
    )


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: GeometryRelationParser,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
    config: TrainConfig,
    filename: str,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "config": asdict(config),
    }
    torch.save(payload, output_dir / filename)


def summarize_epoch(
    epoch: int,
    train_loss: float,
    val_fra: float,
    val_lvr: float,
    lr: float,
    best_fra: float,
    patience_used: int,
    patience_limit: int,
    threshold: float,
) -> str:
    return (
        f"epoch={epoch} "
        f"train_loss={train_loss:.4f} "
        f"val_fra={val_fra:.4f} "
        f"val_lvr={val_lvr:.4f} "
        f"threshold={threshold:.2f} "
        f"lr={lr:.6g} "
        f"best_fra={best_fra:.4f} "
        f"patience={patience_used}/{patience_limit}"
    )


def resolve_active_relations(
    data_root: Path,
    configured_relations: tuple[str, ...],
    ext_root: str | None,
) -> tuple[str, ...]:
    stats_base = Path(ext_root) if ext_root is not None else data_root / "Ext-PGDP5K"
    stats_path = stats_base / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        active_relations = stats.get("global", {}).get("active_relations", [])
        if active_relations:
            return normalize_relations(active_relations)
    return normalize_relations(configured_relations)


def search_best_threshold(
    logits_batches: list[torch.Tensor],
    targets_batches: list[torch.Tensor],
    node_mask_batches: list[torch.Tensor],
    valid_mask_batches: list[torch.Tensor],
    relation_names: tuple[str, ...],
) -> tuple[float, float, float]:
    best_threshold = 0.5
    best_fra = float("-inf")
    best_lvr = float("inf")
    for step in range(35, 71, 1):
        threshold = step / 100.0
        fra_values = []
        lvr_values = []
        for logits, targets, node_mask, valid_mask in zip(
            logits_batches,
            targets_batches,
            node_mask_batches,
            valid_mask_batches,
        ):
            fra_values.append(
                full_relation_accuracy(
                    logits,
                    targets,
                    threshold=threshold,
                    node_mask=node_mask,
                    valid_mask=valid_mask,
                ).item()
            )
            lvr_values.append(
                logic_violation_rate(
                    logits,
                    node_mask=node_mask,
                    threshold=threshold,
                    relation_names=relation_names,
                ).item()
            )
        fra = sum(fra_values) / max(len(fra_values), 1)
        lvr = sum(lvr_values) / max(len(lvr_values), 1)
        if fra > best_fra or (fra == best_fra and lvr < best_lvr):
            best_threshold = threshold
            best_fra = fra
            best_lvr = lvr
    return best_threshold, best_fra, best_lvr


def main() -> None:
    config = parse_args()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in the current environment.")

    data_root = Path(config.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Missing dataset directory: {data_root}")

    active_relations = resolve_active_relations(data_root, config.active_relations, config.ext_root)
    config.active_relations = active_relations

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    device = torch.device(config.device)

    train_dataset = PGDP5KDataset(
        data_root,
        split="train",
        max_nodes=config.max_nodes,
        knn_k=config.knn_k,
        active_relations=active_relations,
        ext_root=config.ext_root,
        shuffle_text=config.shuffle_text,
        shuffle_seed=config.seed,
    )
    val_dataset = PGDP5KDataset(
        data_root,
        split="val",
        max_nodes=config.max_nodes,
        knn_k=config.knn_k,
        active_relations=active_relations,
        ext_root=config.ext_root,
        shuffle_text=config.shuffle_text,
        shuffle_seed=config.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_samples,
    )

    model = GeometryRelationParser(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        feedback_rounds=config.feedback_rounds,
        logic_weight=config.logic_weight,
        relation_names=active_relations,
        use_neighborhood_reasoner=config.use_neighborhood_reasoner,
        use_atomic_relation_bias=config.use_atomic_relation_bias,
        disable_text_guidance=config.disable_text_guidance,
        disable_atomic_loss=config.disable_atomic_loss,
        use_global_text_fusion=config.use_global_text_fusion,
        feature_ablation=config.feature_ablation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_lr,
    )

    best_fra = float("-inf")
    best_threshold = 0.5
    history: list[dict[str, float]] = []
    best_epoch = 0
    epochs_without_improvement = 0

    model.train()
    for epoch in range(config.epochs):
        train_loss = 0.0
        steps = 0
        for raw_batch in train_loader:
            batch = move_batch_to_device(to_parser_batch(raw_batch), device)
            outputs = model(batch)
            assert outputs.loss is not None
            optimizer.zero_grad()
            outputs.loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_loss += outputs.loss.item()
            steps += 1

        model.eval()
        val_logits = []
        val_targets = []
        val_masks = []
        val_valid_masks = []
        with torch.no_grad():
            for raw_batch in val_loader:
                batch = move_batch_to_device(to_parser_batch(raw_batch), device)
                outputs = model(batch)
                val_logits.append(outputs.logits.detach().cpu())
                val_targets.append(batch.relation_targets.detach().cpu())
                val_masks.append(batch.node_mask.detach().cpu())
                val_valid_masks.append(
                    build_relation_valid_mask(
                        batch.node_features.detach().cpu(),
                        active_relations,
                        node_mask=batch.node_mask.detach().cpu(),
                    )
                )
        model.train()

        if config.threshold_search:
            epoch_threshold, epoch_val_fra, epoch_val_lvr = search_best_threshold(
                val_logits,
                val_targets,
                val_masks,
                val_valid_masks,
                active_relations,
            )
        else:
            fra_values = []
            lvr_values = []
            for logits, targets, node_mask, valid_mask in zip(
                val_logits,
                val_targets,
                val_masks,
                val_valid_masks,
            ):
                fra_values.append(
                    full_relation_accuracy(
                        logits,
                        targets,
                        threshold=0.5,
                        node_mask=node_mask,
                        valid_mask=None,
                    ).item()
                )
                lvr_values.append(
                    logic_violation_rate(
                        logits,
                        node_mask=node_mask,
                        threshold=0.5,
                        relation_names=active_relations,
                    ).item()
                )
            epoch_threshold = 0.5
            epoch_val_fra = sum(fra_values) / max(len(fra_values), 1)
            epoch_val_lvr = sum(lvr_values) / max(len(lvr_values), 1)

        epoch_metrics = {
            "train_loss": train_loss / max(steps, 1),
            "val_fra": epoch_val_fra,
            "val_lvr": epoch_val_lvr,
            "threshold": epoch_threshold,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append({"epoch": epoch + 1, **epoch_metrics})
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if config.save_every_epoch:
            save_checkpoint(
                output_dir=output_dir,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                metrics=epoch_metrics,
                config=config,
                filename=f"epoch_{epoch + 1}.pt",
            )
        if epoch_metrics["val_fra"] > best_fra:
            best_fra = epoch_metrics["val_fra"]
            best_threshold = epoch_metrics["threshold"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir=output_dir,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                metrics=epoch_metrics,
                config=config,
                filename="best.pt",
            )
        else:
            epochs_without_improvement += 1

        scheduler.step(epoch_metrics["val_fra"])

        summary_line = summarize_epoch(
            epoch=epoch + 1,
            train_loss=epoch_metrics["train_loss"],
            val_fra=epoch_metrics["val_fra"],
            val_lvr=epoch_metrics["val_lvr"],
            lr=epoch_metrics["lr"],
            best_fra=best_fra,
            patience_used=epochs_without_improvement,
            patience_limit=config.early_stop_patience,
            threshold=epoch_metrics["threshold"],
        )
        with (output_dir / "train.log").open("a", encoding="utf-8") as log_file:
            log_file.write(summary_line + "\n")

        print(summary_line)

        if epochs_without_improvement >= config.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1}: "
                f"best val_fra={best_fra:.4f} from epoch {best_epoch}."
            )
            break

    summary = {
        "active_relations": list(active_relations),
        "best_epoch": best_epoch,
        "best_val_fra": best_fra,
        "best_threshold": best_threshold,
        "epochs_ran": len(history),
        "final_lr": optimizer.param_groups[0]["lr"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
