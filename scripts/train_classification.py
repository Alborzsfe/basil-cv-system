#!/usr/bin/env python
"""Train disease classification from explicit group-split metadata."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.dataset import BasilClassificationDataset
from src.classification.engine import fit_classifier
from src.classification.losses import build_classification_loss
from src.classification.model import build_classifier, freeze_backbone
from src.classification.sampling import inverse_frequency_class_weights, make_weighted_sampler
from src.data.splitting import load_split, validate_persisted_split
from src.segmentation.engine import resolve_device
from src.utils.config import load_config, save_config
from src.utils.io_utils import save_json
from src.utils.manifest import build_run_manifest
from src.utils.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Train basil disease classification.")
    parser.add_argument("--config", default="configs/classification.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.training.optimizer != "adamw" or config.training.lr_scheduler != "cosine":
        raise ValueError("This trainer supports optimizer=adamw and lr_scheduler=cosine.")
    classes = list(config.classes)
    if len(classes) < 2 or any(
        not isinstance(x, str) or not x.strip() or x == "unknown" for x in classes
    ):
        raise ValueError("Configure at least two verified classes; unknown is inference-only.")
    set_seed(int(config.experiment.seed))
    device = resolve_device(args.device)
    split_path = Path(config.split.split_file)
    if not split_path.is_file():
        raise FileNotFoundError(f"Classification split not found: {split_path}.")
    validate_persisted_split(load_split(split_path), config.split.group_column)
    common = {
        "records": split_path,
        "class_names": classes,
        "root_dir": config.dataset.get("root_dir", "."),
        "image_size": config.dataset.image_size,
        "input_mode": config.dataset.input_mode,
        "background_mode": config.dataset.get("background_mode", "black"),
    }
    train = BasilClassificationDataset(
        **common,
        split="train",
        augment=bool(config.augmentation.enabled),
        augmentation=config.augmentation.to_dict(),
    )
    val = BasilClassificationDataset(**common, split="val")
    options = {
        "batch_size": int(config.training.batch_size),
        "num_workers": int(config.dataset.num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    loss_type, strategy = str(config.loss.type), str(config.sampling.strategy)
    if loss_type == "weighted_cross_entropy" and strategy == "weighted_sampler":
        raise ValueError("Use weighted loss or weighted sampling, not both.")
    if strategy not in {"none", "weighted_sampler"}:
        raise ValueError(f"Unsupported sampling strategy '{strategy}'.")
    labels = train.records["label"].astype(str).tolist()
    sampler = (
        make_weighted_sampler(labels, classes, seed=int(config.experiment.seed))
        if strategy == "weighted_sampler"
        else None
    )
    train_loader = DataLoader(train, shuffle=sampler is None, sampler=sampler, **options)
    val_loader = DataLoader(val, shuffle=False, **options)
    in_channels = 4 if config.dataset.input_mode == "leaf_only_mask_aware" else 3
    model = build_classifier(
        config.model.architecture,
        num_classes=len(classes),
        pretrained=bool(config.model.pretrained),
        in_channels=in_channels,
    )
    if bool(config.model.freeze_backbone_initially):
        freeze_backbone(model, config.model.architecture)
    weights = (
        inverse_frequency_class_weights(labels, classes).to(device)
        if loss_type == "weighted_cross_entropy"
        else None
    )
    criterion = build_classification_loss(
        loss_type, focal_gamma=float(config.loss.focal_gamma), class_weights=weights
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.head_learning_rate),
        weight_decay=float(config.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(config.training.epochs))
    output = Path(config.experiment.output_dir)
    save_config(config, output / "resolved_config.yaml")
    shutil.copy2(split_path, output / "split.csv")
    save_json(
        build_run_manifest(seed=int(config.experiment.seed), device=str(device)),
        output / "run_manifest.json",
    )
    model_config = {
        "architecture": config.model.architecture,
        "num_classes": len(classes),
        "pretrained": False,
        "in_channels": in_channels,
    }
    history = fit_classifier(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device=device,
        class_names=classes,
        epochs=int(config.training.epochs),
        checkpoint_dir=config.training.checkpoint_dir,
        model_config=model_config,
        input_mode=config.dataset.input_mode,
        patience=int(config.training.early_stopping_patience),
        scheduler=scheduler,
        mixed_precision=bool(config.training.mixed_precision),
        unfreeze_after_epoch=int(config.model.unfreeze_after_epoch)
        if bool(config.model.freeze_backbone_initially)
        else None,
        fine_tune_learning_rate=float(config.model.fine_tune_learning_rate),
    )
    print(f"Completed {len(history)} epoch(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
