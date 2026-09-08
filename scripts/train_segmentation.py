#!/usr/bin/env python
"""Train leaf semantic segmentation from a persisted group-aware split."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splitting import load_split, validate_persisted_split
from src.segmentation.dataset import BasilSegmentationDataset
from src.segmentation.engine import fit_segmentation, resolve_device
from src.segmentation.losses import build_segmentation_loss
from src.segmentation.model import build_segmentation_model
from src.utils.config import load_config, save_config
from src.utils.io_utils import save_json
from src.utils.manifest import build_run_manifest
from src.utils.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Train basil leaf segmentation.")
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.training.optimizer != "adamw" or config.training.lr_scheduler != "cosine":
        raise ValueError("This trainer supports optimizer=adamw and lr_scheduler=cosine.")
    set_seed(int(config.experiment.seed))
    split_path = Path(config.split.split_file)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split not found: {split_path}. Run prepare, validate, and split.")
    validate_persisted_split(load_split(split_path), config.split.group_column)
    common = {
        "records": split_path,
        "images_dir": config.dataset.images_dir,
        "masks_dir": config.dataset.masks_dir,
        "image_size": config.dataset.image_size,
    }
    train = BasilSegmentationDataset(
        **common,
        split="train",
        augment=bool(config.augmentation.enabled),
        augmentation=config.augmentation.to_dict(),
    )
    val = BasilSegmentationDataset(**common, split="val")
    options = {
        "batch_size": int(config.training.batch_size),
        "num_workers": int(config.dataset.num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train, shuffle=True, **options)
    val_loader = DataLoader(val, shuffle=False, **options)
    model_config = config.model.to_dict()
    model = build_segmentation_model(**model_config)
    criterion = build_segmentation_loss(
        config.loss.type,
        dice_weight=float(config.loss.dice_weight),
        bce_weight=float(config.loss.bce_weight),
        focal_gamma=float(config.loss.focal_gamma),
        focal_alpha=float(config.loss.focal_alpha),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config.training.epochs)
    )
    device = resolve_device(args.device)
    output = Path(config.experiment.output_dir)
    save_config(config, output / "resolved_config.yaml")
    shutil.copy2(split_path, output / "split.csv")
    save_json(
        build_run_manifest(seed=int(config.experiment.seed), device=str(device)),
        output / "run_manifest.json",
    )
    history = fit_segmentation(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device=device,
        epochs=int(config.training.epochs),
        checkpoint_dir=config.training.checkpoint_dir,
        model_config=model_config,
        threshold=float(config.metrics.threshold),
        patience=int(config.training.early_stopping_patience),
        scheduler=scheduler,
        mixed_precision=bool(config.training.mixed_precision),
    )
    print(f"Completed {len(history)} epoch(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
