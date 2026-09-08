from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.classification.engine import (
    evaluate_classifier,
    fit_classifier,
    load_classification_checkpoint,
    run_classification_epoch,
)
from src.classification.model import build_classifier
from src.segmentation.engine import (
    fit_segmentation,
    load_segmentation_checkpoint,
    run_segmentation_epoch,
)
from src.segmentation.model import build_segmentation_model


def _segmentation_batches() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "image": torch.randn(2, 3, 16, 16),
            "mask": torch.randint(0, 2, (2, 1, 16, 16)).float(),
        }
    ]


def _classification_batches() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "image": torch.randn(4, 3, 16, 16),
            "target": torch.tensor([0, 1, 0, 1]),
        }
    ]


def test_segmentation_fit_evaluate_and_reload(tmp_path: Path) -> None:
    config = {
        "architecture": "tiny_unet",
        "encoder_name": "tiny",
        "encoder_weights": None,
        "base_channels": 2,
    }
    model = build_segmentation_model(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = fit_segmentation(
        model,
        _segmentation_batches(),
        _segmentation_batches(),
        torch.nn.BCEWithLogitsLoss(),
        optimizer,
        device=torch.device("cpu"),
        epochs=1,
        checkpoint_dir=tmp_path,
        model_config=config,
        patience=1,
        mixed_precision=False,
    )
    assert len(history) == 1
    assert (tmp_path / "best_model.pt").is_file()
    assert (tmp_path / "history.json").is_file()
    restored = build_segmentation_model(**config)
    checkpoint = load_segmentation_checkpoint(
        tmp_path / "best_model.pt", restored, device=torch.device("cpu")
    )
    assert checkpoint["task"] == "binary_leaf_segmentation"
    metrics = run_segmentation_epoch(
        restored,
        _segmentation_batches(),
        torch.nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"),
    )
    assert 0 <= metrics["dice"] <= 1


def test_classification_fit_evaluate_and_reload(tmp_path: Path) -> None:
    config = {
        "architecture": "tiny_cnn",
        "num_classes": 2,
        "pretrained": False,
        "in_channels": 3,
    }
    model = build_classifier(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    history = fit_classifier(
        model,
        _classification_batches(),
        _classification_batches(),
        criterion,
        optimizer,
        device=torch.device("cpu"),
        class_names=["healthy", "spotted"],
        epochs=1,
        checkpoint_dir=tmp_path,
        model_config=config,
        input_mode="raw_image",
        patience=1,
        mixed_precision=False,
    )
    assert len(history) == 1
    restored = build_classifier(**config)
    checkpoint = load_classification_checkpoint(
        tmp_path / "best_model.pt", restored, device=torch.device("cpu")
    )
    assert checkpoint["class_names"] == ["healthy", "spotted"]
    metrics = evaluate_classifier(
        restored,
        _classification_batches(),
        criterion,
        device=torch.device("cpu"),
        class_names=["healthy", "spotted"],
    )
    assert metrics["num_samples"] == 4


def test_empty_epoch_loaders_are_rejected() -> None:
    segmenter = build_segmentation_model(
        "tiny_unet", encoder_name="tiny", encoder_weights=None, base_channels=2
    )
    with pytest.raises(ValueError, match="no samples"):
        run_segmentation_epoch(
            segmenter,
            [],
            torch.nn.BCEWithLogitsLoss(),
            device=torch.device("cpu"),
        )
    classifier = build_classifier("tiny_cnn", num_classes=2, pretrained=False)
    with pytest.raises(ValueError, match="no samples"):
        run_classification_epoch(
            classifier,
            [],
            torch.nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )
