from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.prepare_dataset import update_metadata
from scripts.split_classification import validate_metadata
from src.classification.losses import build_classification_loss
from src.classification.sampling import (
    class_counts,
    inverse_frequency_class_weights,
    make_weighted_sampler,
)
from src.data.splitting import group_aware_split, validate_persisted_split
from src.data.validation import DatasetValidator
from src.segmentation.engine import resolve_device
from src.segmentation.losses import build_segmentation_loss
from src.utils.config import Config
from src.utils.io_utils import ImageReadError, safe_read_mask


@pytest.mark.parametrize("loss_type", ["dice_bce", "dice", "bce", "focal"])
def test_segmentation_losses_backpropagate(loss_type: str) -> None:
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 8, 8)).float()
    loss = build_segmentation_loss(loss_type)(logits, target)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("loss_type", ["cross_entropy", "weighted_cross_entropy", "focal"])
def test_classification_losses_backpropagate(loss_type: str) -> None:
    logits = torch.randn(4, 3, requires_grad=True)
    target = torch.tensor([0, 1, 2, 0])
    weights = torch.tensor([1.0, 2.0, 3.0]) if loss_type != "cross_entropy" else None
    loss = build_classification_loss(loss_type, class_weights=weights)(logits, target)
    assert torch.isfinite(loss)
    loss.backward()


def test_sampling_helpers_are_balanced_and_deterministic() -> None:
    labels = ["a", "a", "a", "b"]
    names = ["a", "b"]
    assert class_counts(labels, names).tolist() == [3, 1]
    weights = inverse_frequency_class_weights(labels, names)
    assert weights[1] == pytest.approx(3 * weights[0])
    first = list(make_weighted_sampler(labels, names, seed=7))
    second = list(make_weighted_sampler(labels, names, seed=7))
    assert first == second
    with pytest.raises(ValueError):
        class_counts(["a"], names)


def test_resolve_device_and_split_validation() -> None:
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(ValueError):
        resolve_device("tpu")
    split = pd.DataFrame(
        {
            "plant_id": ["a", "b", "c"],
            "split": ["train", "val", "test"],
        }
    )
    validate_persisted_split(split, "plant_id")
    leaking = pd.concat(
        [split, pd.DataFrame({"plant_id": ["a"], "split": ["test"]})], ignore_index=True
    )
    with pytest.raises(ValueError, match="leakage"):
        validate_persisted_split(leaking, "plant_id")
    with pytest.raises(ValueError):
        group_aware_split(split, "plant_id", 0.8, 0.3, 42)


def test_prepare_metadata_refreshes_missing_mask(tmp_path: Path) -> None:
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    image_path = images / "leaf.JPG"
    cv2.imwrite(str(image_path), np.full((8, 8, 3), 80, dtype=np.uint8))
    existing = pd.DataFrame(
        [
            {
                "image_filename": image_path.name,
                "mask_filename": "leaf.MISSING",
                "plant_id": "plant-7",
            }
        ]
    )
    initial, added, refreshed = update_metadata([image_path], masks, existing)
    assert (added, refreshed) == (0, 0)
    assert initial.loc[0, "plant_id"] == "plant-7"
    cv2.imwrite(str(masks / "LEAF.png"), np.zeros((8, 8), dtype=np.uint8))
    updated, added, refreshed = update_metadata([image_path], masks, initial)
    assert (added, refreshed) == (0, 1)
    assert updated.loc[0, "mask_filename"] == "LEAF.png"
    assert updated.loc[0, "plant_id"] == "plant-7"


def test_validator_suffix_mapping_and_inconsistent_color_mask(tmp_path: Path) -> None:
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    cv2.imwrite(str(images / "leaf_image.png"), np.full((10, 10, 3), 100, dtype=np.uint8))
    valid = np.zeros((10, 10), dtype=np.uint8)
    valid[2:8, 2:8] = 255
    cv2.imwrite(str(masks / "leaf_mask.png"), valid)
    result = DatasetValidator(images, masks, {"_image": "_mask"}).validate()
    assert result.is_valid

    inconsistent = np.stack((valid, np.zeros_like(valid), valid), axis=2)
    path = masks / "bad.png"
    cv2.imwrite(str(path), inconsistent)
    with pytest.raises(ImageReadError, match="inconsistent"):
        safe_read_mask(path)


def test_classification_metadata_contract(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    cv2.imwrite(str(image), np.full((8, 8, 3), 100, dtype=np.uint8))
    cv2.imwrite(str(mask), np.ones((8, 8), dtype=np.uint8))
    frame = pd.DataFrame(
        [
            {"image_path": image.name, "mask_path": mask.name, "label": "a", "plant_id": "1"},
            {"image_path": "image2.png", "mask_path": mask.name, "label": "b", "plant_id": "2"},
        ]
    )
    second = tmp_path / "image2.png"
    cv2.imwrite(str(second), np.full((8, 8, 3), 120, dtype=np.uint8))
    config = Config(
        {
            "classes": ["a", "b"],
            "split": {"group_column": "plant_id"},
            "dataset": {"root_dir": str(tmp_path), "input_mode": "leaf_only"},
        }
    )
    validate_metadata(frame, config)
    frame.loc[0, "plant_id"] = "UNKNOWN"
    with pytest.raises(ValueError, match="grouping"):
        validate_metadata(frame, config)
