#!/usr/bin/env python
"""Evaluate a trusted segmentation checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segmentation.dataset import BasilSegmentationDataset
from src.segmentation.engine import (
    load_segmentation_checkpoint,
    resolve_device,
    run_segmentation_epoch,
)
from src.segmentation.losses import build_segmentation_loss
from src.segmentation.model import build_segmentation_model
from src.utils.config import load_config
from src.utils.io_utils import save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate basil segmentation.")
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_path = Path(
        args.checkpoint or Path(config.training.checkpoint_dir) / "best_model.pt"
    )
    device = resolve_device(args.device)
    model = build_segmentation_model(**config.model.to_dict())
    load_segmentation_checkpoint(checkpoint_path, model, device=device)
    dataset = BasilSegmentationDataset(
        config.split.split_file,
        images_dir=config.dataset.images_dir,
        masks_dir=config.dataset.masks_dir,
        split="test",
        image_size=config.dataset.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.training.batch_size),
        num_workers=int(config.dataset.num_workers),
    )
    criterion = build_segmentation_loss(
        config.loss.type,
        dice_weight=float(config.loss.dice_weight),
        bce_weight=float(config.loss.bce_weight),
        focal_gamma=float(config.loss.focal_gamma),
        focal_alpha=float(config.loss.focal_alpha),
    )
    with torch.inference_mode():
        metrics = run_segmentation_epoch(
            model, loader, criterion, device=device, threshold=float(config.metrics.threshold)
        )
    save_json(metrics, Path(config.experiment.output_dir) / "test_metrics.json")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
