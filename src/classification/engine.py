"""Training, evaluation, and checkpointing for disease classification."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.evaluation.classification import classification_metrics
from src.utils.io_utils import ensure_dir, save_json


def run_classification_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    criterion: nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, list[int], np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss, sample_count = 0.0, 0
    all_targets: list[int] = []
    all_probabilities: list[np.ndarray] = []
    autocast_enabled = device.type == "cuda" and scaler is not None
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
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
        all_targets.extend(targets.detach().cpu().tolist())
        all_probabilities.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
    if not sample_count:
        raise ValueError("DataLoader produced no samples.")
    return total_loss / sample_count, all_targets, np.concatenate(all_probabilities)


def evaluate_classifier(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    criterion: nn.Module,
    *,
    device: torch.device,
    class_names: Sequence[str],
) -> dict[str, object]:
    model.to(device)
    with torch.inference_mode():
        loss, targets, probabilities = run_classification_epoch(
            model, loader, criterion, device=device
        )
    return {"loss": loss, **classification_metrics(targets, probabilities, class_names)}


def fit_classifier(
    model: nn.Module,
    train_loader: Iterable[dict[str, Any]],
    val_loader: Iterable[dict[str, Any]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    class_names: Sequence[str],
    epochs: int,
    checkpoint_dir: str | Path,
    model_config: dict[str, Any],
    input_mode: str,
    patience: int = 8,
    scheduler: Any = None,
    mixed_precision: bool = True,
    unfreeze_after_epoch: int | None = None,
    fine_tune_learning_rate: float | None = None,
) -> list[dict[str, Any]]:
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive.")
    destination = ensure_dir(checkpoint_dir)
    model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")
    best_f1, stale = -1.0, 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        if unfreeze_after_epoch is not None and epoch == unfreeze_after_epoch:
            for parameter in model.parameters():
                parameter.requires_grad = True
            if fine_tune_learning_rate is not None:
                for group in optimizer.param_groups:
                    group["lr"] = fine_tune_learning_rate
        started = time.perf_counter()
        train_loss, targets, probabilities = run_classification_epoch(
            model, train_loader, criterion, device=device, optimizer=optimizer, scaler=scaler
        )
        train_metrics = classification_metrics(targets, probabilities, class_names)
        val_metrics = evaluate_classifier(
            model, val_loader, criterion, device=device, class_names=class_names
        )
        if scheduler is not None:
            scheduler.step()
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "train": {"loss": train_loss, **train_metrics},
            "val": val_metrics,
        }
        history.append(record)
        checkpoint = {
            "schema_version": 1,
            "task": "basil_disease_classification",
            "epoch": epoch,
            "model_config": model_config,
            "input_mode": input_mode,
            "class_names": list(class_names),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": record,
        }
        torch.save(checkpoint, destination / "last_model.pt")
        current = float(val_metrics["f1_macro"])
        if current > best_f1:
            best_f1, stale = current, 0
            torch.save(checkpoint, destination / "best_model.pt")
        else:
            stale += 1
        if stale >= patience:
            break
    save_json({"history": history, "best_val_macro_f1": best_f1}, destination / "history.json")
    return history


def load_classification_checkpoint(
    path: str | Path, model: nn.Module, *, device: torch.device
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("task") != "basil_disease_classification":
        raise ValueError("Checkpoint is not a basil disease-classification checkpoint.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return checkpoint
