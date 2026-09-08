#!/usr/bin/env python
"""Evaluate a trusted classification checkpoint on held-out test data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.dataset import BasilClassificationDataset
from src.classification.engine import evaluate_classifier, load_classification_checkpoint
from src.classification.losses import build_classification_loss
from src.classification.model import build_classifier
from src.segmentation.engine import resolve_device
from src.utils.config import load_config
from src.utils.io_utils import save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate basil classification.")
    parser.add_argument("--config", default="configs/classification.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    path = Path(args.checkpoint or Path(config.training.checkpoint_dir) / "best_model.pt")
    device = resolve_device(args.device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_classifier(**checkpoint["model_config"])
    load_classification_checkpoint(path, model, device=device)
    classes = list(checkpoint["class_names"])
    dataset = BasilClassificationDataset(
        config.split.split_file,
        class_names=classes,
        root_dir=config.dataset.get("root_dir", "."),
        split="test",
        image_size=config.dataset.image_size,
        input_mode=checkpoint["input_mode"],
        background_mode=config.dataset.get("background_mode", "black"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.training.batch_size),
        num_workers=int(config.dataset.num_workers),
    )
    metrics = evaluate_classifier(
        model,
        loader,
        build_classification_loss("cross_entropy"),
        device=device,
        class_names=classes,
    )
    save_json(metrics, Path(config.experiment.output_dir) / "test_metrics.json")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
