"""Tests for Phase 1 (configuration) and Phase 2 (dataset validation,
reporting, and group-aware splitting) using small synthetic images/masks.

Run with:
    pytest tests/test_phase1_phase2.py -v
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.data.report import build_dataset_report
from src.data.splitting import group_aware_split, split_counts
from src.data.validation import DatasetValidator
from src.utils.config import load_config, save_config


def _write_image_and_mask(
    images_dir: Path,
    masks_dir: Path,
    name: str,
    size=(64, 64),
    leaf_fraction: float = 0.3,
    mask_value_convention: int = 1,
) -> None:
    height, width = size
    image = np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    leaf_pixels = int(height * width * leaf_fraction)
    flat_mask = mask.reshape(-1)
    flat_mask[:leaf_pixels] = mask_value_convention
    mask = flat_mask.reshape(height, width)

    cv2.imwrite(str(images_dir / f"{name}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(masks_dir / f"{name}.png"), mask)


@pytest.fixture()
def synthetic_dataset(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    for index in range(5):
        _write_image_and_mask(images_dir, masks_dir, f"plant_{index:03d}")

    return images_dir, masks_dir


def test_config_load_and_save(tmp_path: Path):
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("dataset:\n  image_size: [512, 512]\n  num_workers: 4\n")

    config = load_config(config_path)
    assert config.dataset.image_size == [512, 512]
    assert config.dataset.num_workers == 4
    assert config["dataset"]["num_workers"] == 4

    out_path = tmp_path / "saved_config.yaml"
    save_config(config, out_path)
    reloaded = load_config(out_path)
    assert reloaded.dataset.num_workers == 4


def test_config_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_validator_passes_on_clean_dataset(synthetic_dataset):
    images_dir, masks_dir = synthetic_dataset
    validator = DatasetValidator(images_dir=images_dir, masks_dir=masks_dir)
    result = validator.validate()

    assert result.is_valid
    assert len(result.per_image_stats) == 5
    for stats in result.per_image_stats:
        assert 0.0 <= stats.leaf_coverage <= 1.0


def test_validator_detects_missing_mask(synthetic_dataset):
    images_dir, masks_dir = synthetic_dataset
    extra_image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "plant_999.jpg"), extra_image)

    validator = DatasetValidator(images_dir=images_dir, masks_dir=masks_dir)
    result = validator.validate()

    assert not result.is_valid
    missing_mask_issues = [i for i in result.errors if i.category == "missing_mask"]
    assert len(missing_mask_issues) == 1
    assert missing_mask_issues[0].filename == "plant_999.jpg"


def test_validator_detects_dimension_mismatch(synthetic_dataset):
    images_dir, masks_dir = synthetic_dataset
    bad_mask = np.zeros((32, 32), dtype=np.uint8)
    cv2.imwrite(str(masks_dir / "plant_000.png"), bad_mask)

    validator = DatasetValidator(images_dir=images_dir, masks_dir=masks_dir)
    result = validator.validate()

    assert not result.is_valid
    mismatch_issues = [i for i in result.errors if i.category == "dimension_mismatch"]
    assert len(mismatch_issues) == 1


def test_validator_detects_unexpected_mask_values(synthetic_dataset):
    images_dir, masks_dir = synthetic_dataset
    weird_mask = np.full((64, 64), 42, dtype=np.uint8)
    cv2.imwrite(str(masks_dir / "plant_000.png"), weird_mask)

    validator = DatasetValidator(images_dir=images_dir, masks_dir=masks_dir)
    result = validator.validate()

    assert not result.is_valid
    unexpected_value_issues = [i for i in result.errors if i.category == "unexpected_mask_values"]
    assert len(unexpected_value_issues) == 1


def test_validator_detects_empty_mask_as_warning(synthetic_dataset):
    images_dir, masks_dir = synthetic_dataset
    empty_mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.imwrite(str(masks_dir / "plant_000.png"), empty_mask)

    validator = DatasetValidator(images_dir=images_dir, masks_dir=masks_dir)
    result = validator.validate()

    # Empty masks are warnings, not errors: the dataset can still be valid.
    assert result.is_valid
    empty_mask_issues = [i for i in result.warnings if i.category == "empty_mask"]
    assert len(empty_mask_issues) == 1


def test_validator_detects_duplicate_filenames(synthetic_dataset):
    images_dir, masks_dir = synthetic_dataset
    duplicate_image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "plant_000.PNG"), duplicate_image)

    validator = DatasetValidator(images_dir=images_dir, masks_dir=masks_dir)
    result = validator.validate()

    assert not result.is_valid
    duplicate_issues = [i for i in result.errors if i.category == "duplicate_filename"]
    assert len(duplicate_issues) >= 1


def test_dataset_report_statistics(synthetic_dataset):
    images_dir, masks_dir = synthetic_dataset
    validator = DatasetValidator(images_dir=images_dir, masks_dir=masks_dir)
    result = validator.validate()

    report = build_dataset_report(result, num_image_files=5, num_mask_files=5)
    assert report.num_valid_pairs == 5
    assert report.num_errors == 0
    assert 0.0 <= report.mean_leaf_coverage <= 1.0
    assert report.min_leaf_coverage <= report.mean_leaf_coverage <= report.max_leaf_coverage


def test_group_aware_split_prevents_leakage():
    rows = []
    for plant_index in range(20):
        for image_index in range(3):
            rows.append(
                {
                    "image_filename": f"plant_{plant_index}_{image_index}.jpg",
                    "mask_filename": f"plant_{plant_index}_{image_index}.png",
                    "plant_id": f"plant_{plant_index}",
                }
            )
    metadata = pd.DataFrame(rows)

    split_metadata = group_aware_split(
        metadata, group_column="plant_id", test_size=0.2, val_size=0.15, seed=42
    )

    train_groups = set(split_metadata[split_metadata["split"] == "train"]["plant_id"])
    val_groups = set(split_metadata[split_metadata["split"] == "val"]["plant_id"])
    test_groups = set(split_metadata[split_metadata["split"] == "test"]["plant_id"])

    assert not (train_groups & val_groups)
    assert not (train_groups & test_groups)
    assert not (val_groups & test_groups)

    counts = split_counts(split_metadata)
    assert sum(counts.values()) == len(metadata)
    assert counts["train"] > 0
    assert counts["test"] > 0


def test_group_aware_split_rejects_too_few_groups():
    metadata = pd.DataFrame(
        {
            "image_filename": ["a.jpg", "b.jpg"],
            "mask_filename": ["a.png", "b.png"],
            "plant_id": ["plant_0", "plant_1"],
        }
    )
    with pytest.raises(ValueError):
        group_aware_split(metadata, group_column="plant_id", test_size=0.2, val_size=0.15, seed=42)
