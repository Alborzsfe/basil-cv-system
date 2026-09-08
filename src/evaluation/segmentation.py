"""Binary semantic-segmentation metrics with explicit empty-mask behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_binary(array: Any, threshold: float) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    values = np.asarray(array)
    if values.size == 0:
        raise ValueError("Metric inputs cannot be empty.")
    if not np.isfinite(values).all():
        raise ValueError("Metric inputs must contain only finite values.")
    return values >= threshold


def segmentation_confusion(
    prediction: Any, target: Any, *, threshold: float = 0.5
) -> tuple[int, int, int, int]:
    predicted = _as_binary(prediction, threshold)
    expected = _as_binary(target, 0.5)
    if predicted.shape != expected.shape:
        raise ValueError(f"Prediction/target shapes differ: {predicted.shape} vs {expected.shape}.")
    tp = int(np.logical_and(predicted, expected).sum())
    fp = int(np.logical_and(predicted, ~expected).sum())
    fn = int(np.logical_and(~predicted, expected).sum())
    tn = int(np.logical_and(~predicted, ~expected).sum())
    return tp, fp, fn, tn


def _ratio(numerator: float, denominator: float, empty_value: float) -> float:
    return float(numerator / denominator) if denominator else float(empty_value)


def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    both_empty = tp + fp + fn == 0
    dice = _ratio(2 * tp, 2 * tp + fp + fn, 1.0)
    return {
        "dice": dice,
        "iou": _ratio(tp, tp + fp + fn, 1.0),
        "precision": _ratio(tp, tp + fp, 1.0 if both_empty else 0.0),
        "recall": _ratio(tp, tp + fn, 1.0 if both_empty else 0.0),
        "f1": dice,
        "pixel_accuracy": _ratio(tp + tn, tp + fp + fn + tn, 1.0),
    }


def segmentation_metrics(
    prediction: Any, target: Any, *, threshold: float = 0.5
) -> dict[str, float]:
    return metrics_from_confusion(*segmentation_confusion(prediction, target, threshold=threshold))


@dataclass
class SegmentationMeter:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(self, prediction: Any, target: Any, *, threshold: float = 0.5) -> None:
        tp, fp, fn, tn = segmentation_confusion(prediction, target, threshold=threshold)
        self.true_positive += tp
        self.false_positive += fp
        self.false_negative += fn
        self.true_negative += tn

    def compute(self) -> dict[str, float]:
        return metrics_from_confusion(
            self.true_positive, self.false_positive, self.false_negative, self.true_negative
        )
