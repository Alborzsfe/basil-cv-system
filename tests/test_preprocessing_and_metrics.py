from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.classification import classification_metrics
from src.evaluation.segmentation import SegmentationMeter, segmentation_metrics
from src.preprocessing.background import (
    apply_background,
    clean_binary_mask,
    leaf_coverage,
    normalize_binary_mask,
)
from src.preprocessing.quality import assess_image_quality


def test_mask_normalization_cleanup_and_coverage() -> None:
    mask = np.zeros((15, 15), dtype=np.uint8)
    mask[3:12, 3:12] = 255
    mask[0, 0] = 255
    normalized = normalize_binary_mask(mask)
    cleaned = clean_binary_mask(mask, kernel_size=3, minimum_component_area=20)
    assert normalized.dtype == np.uint8
    assert normalized[0, 0] == 1
    assert cleaned[0, 0] == 0
    assert 0.25 < leaf_coverage(cleaned) < 0.5


@pytest.mark.parametrize("mode", ["black", "gray", "blur"])
def test_background_modes_preserve_foreground(mode: str) -> None:
    image = np.full((9, 9, 3), (20, 150, 40), dtype=np.uint8)
    image[:, :4] = (220, 30, 30)
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[2:7, 2:7] = 1
    output = apply_background(image, mask, mode=mode, gray_value=90, blur_kernel_size=3)
    assert np.array_equal(output[4, 4], image[4, 4])
    if mode == "black":
        assert output[0, 0].tolist() == [0, 0, 0]
    elif mode == "gray":
        assert output[0, 0].tolist() == [90, 90, 90]
    else:
        assert output.shape == image.shape


def test_background_and_mask_validation() -> None:
    with pytest.raises(ValueError):
        normalize_binary_mask(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        clean_binary_mask(np.zeros((3, 3)), kernel_size=2)
    with pytest.raises(ValueError):
        apply_background(np.zeros((4, 4, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        apply_background(np.zeros((4, 4, 3)), np.zeros((4, 4)), mode="white")


def test_quality_gate_reports_specific_reasons() -> None:
    dark = np.zeros((20, 20, 3), dtype=np.uint8)
    report = assess_image_quality(dark)
    assert not report.passed
    assert {"too_dark", "too_blurry"}.issubset(report.reasons)
    textured = np.indices((32, 32)).sum(axis=0) % 2 * 255
    rgb = np.repeat(textured[..., None], 3, axis=2).astype(np.uint8)
    accepted = assess_image_quality(rgb)
    assert accepted.passed
    assert accepted.to_dict()["reasons"] == []


def test_segmentation_metrics_and_meter() -> None:
    target = np.array([[1, 1], [0, 0]], dtype=np.float32)
    prediction = np.array([[0.9, 0.7], [0.6, 0.1]], dtype=np.float32)
    metrics = segmentation_metrics(prediction, target)
    assert metrics["dice"] == pytest.approx(0.8)
    assert metrics["iou"] == pytest.approx(2 / 3)
    meter = SegmentationMeter()
    meter.update(prediction, target)
    meter.update(np.zeros((2, 2)), np.zeros((2, 2)))
    assert meter.compute()["pixel_accuracy"] == pytest.approx(7 / 8)
    empty = segmentation_metrics(np.zeros((2, 2)), np.zeros((2, 2)))
    assert empty["dice"] == 1.0


def test_classification_metrics_cover_multiclass_and_absent_class() -> None:
    truth = [0, 1, 0, 1]
    scores = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
    metrics = classification_metrics(truth, scores, ["healthy", "spot", "other"])
    assert metrics["accuracy"] == 1.0
    assert metrics["num_samples"] == 4
    assert metrics["per_class"]["other"]["support"] == 0
    assert metrics["roc_auc_macro_ovr"] == 1.0


def test_classification_metrics_reject_invalid_probabilities() -> None:
    with pytest.raises(ValueError):
        classification_metrics([0], np.array([[0.7, 0.4]]), ["a", "b"])
