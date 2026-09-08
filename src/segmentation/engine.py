"""Training, validation, and checkpointing for binary segmentation."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.evaluation.segmentation import SegmentationMeter
from src.utils.io_utils import ensure_dir, save_json


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def run_segmentation_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    criterion: nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    meter = SegmentationMeter()
    total_loss, sample_count = 0.0, 0
    autocast_enabled = device.type == "cuda" and scaler is not None
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        size = int(images.shape[0])
        total_loss += float(loss.detach()) * size
        sample_count += size
        meter.update(torch.sigmoid(logits), masks, threshold=threshold)
    if not sample_count:
        raise ValueError("DataLoader produced no samples.")
    return {"loss": total_loss / sample_count, **meter.compute()}


def fit_segmentation(
    model: nn.Module,
    train_loader: Iterable[dict[str, Any]],
    val_loader: Iterable[dict[str, Any]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epochs: int,
    checkpoint_dir: str | Path,
    model_config: dict[str, Any],
    threshold: float = 0.5,
    patience: int = 10,
    scheduler: Any = None,
    mixed_precision: bool = True,
) -> list[dict[str, Any]]:
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive.")
    destination = ensure_dir(checkpoint_dir)
    model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")
    best_dice, stale = -1.0, 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_metrics = run_segmentation_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            threshold=threshold,
        )
        with torch.inference_mode():
            val_metrics = run_segmentation_epoch(
                model, val_loader, criterion, device=device, threshold=threshold
            )
        if scheduler is not None:
            scheduler.step()
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        checkpoint = {
            "schema_version": 1,
            "task": "binary_leaf_segmentation",
            "epoch": epoch,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": record,
        }
        torch.save(checkpoint, destination / "last_model.pt")
        if val_metrics["dice"] > best_dice:
            best_dice, stale = val_metrics["dice"], 0
            torch.save(checkpoint, destination / "best_model.pt")
        else:
            stale += 1
        if stale >= patience:
            break
    save_json({"history": history, "best_val_dice": best_dice}, destination / "history.json")
    return history


def load_segmentation_checkpoint(
    path: str | Path, model: nn.Module, *, device: torch.device
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("task") != "binary_leaf_segmentation":
        raise ValueError("Checkpoint is not a basil binary-segmentation checkpoint.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return checkpoint
