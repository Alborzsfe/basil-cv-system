"""Classification metrics suitable for imbalanced multi-class disease data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(
    y_true: Sequence[int], probabilities: np.ndarray, class_names: Sequence[str]
) -> dict[str, object]:
    truth = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    names = list(class_names)
    if truth.ndim != 1 or truth.size == 0:
        raise ValueError("y_true must be a non-empty one-dimensional sequence.")
    if scores.ndim != 2 or scores.shape != (truth.size, len(names)):
        raise ValueError("probabilities must have shape (samples, classes).")
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("class_names must contain at least two unique names.")
    if (truth < 0).any() or (truth >= len(names)).any():
        raise ValueError("y_true contains an out-of-range class index.")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("probabilities must be finite and non-negative.")
    if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to one.")
    prediction = scores.argmax(axis=1)
    labels = np.arange(len(names))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=labels, zero_division=0
    )
    macro = precision_recall_fscore_support(
        truth, prediction, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        truth, prediction, labels=labels, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(truth, prediction, labels=labels)
    per_class = {}
    for index, name in enumerate(names):
        tn = int(
            matrix.sum() - matrix[index, :].sum() - matrix[:, index].sum() + matrix[index, index]
        )
        fp = int(matrix[:, index].sum() - matrix[index, index])
        per_class[name] = {
            "precision": float(precision[index]),
            "recall_sensitivity": float(recall[index]),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    roc_auc, pr_auc = _auc(truth, scores, len(names))
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "f1_weighted": float(weighted[2]),
        "roc_auc_macro_ovr": roc_auc,
        "pr_auc_macro": pr_auc,
        "confusion_matrix": matrix.tolist(),
        "class_names": names,
        "per_class": per_class,
        "num_samples": int(truth.size),
    }


def _auc(
    truth: np.ndarray, scores: np.ndarray, num_classes: int
) -> tuple[float | None, float | None]:
    binary = label_binarize(truth, classes=np.arange(num_classes))
    if num_classes == 2:
        binary = np.column_stack([1 - binary[:, 0], binary[:, 0]])
    valid = [index for index in range(num_classes) if np.unique(binary[:, index]).size == 2]
    if not valid:
        return None, None
    try:
        roc = float(np.mean([roc_auc_score(binary[:, i], scores[:, i]) for i in valid]))
        pr = float(np.mean([average_precision_score(binary[:, i], scores[:, i]) for i in valid]))
        return roc, pr
    except ValueError:
        return None, None
