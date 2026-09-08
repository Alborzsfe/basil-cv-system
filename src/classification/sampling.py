"""Auditable class-imbalance helpers for disease classification."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def class_counts(labels: Sequence[str], class_names: Sequence[str]) -> np.ndarray:
    names = list(class_names)
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("class_names must contain at least two unique labels.")
    unknown = sorted(set(labels) - set(names))
    if unknown:
        raise ValueError(f"Labels absent from class_names: {unknown}.")
    counts = np.asarray([sum(label == name for label in labels) for name in names], dtype=np.int64)
    missing = [name for name, count in zip(names, counts, strict=True) if count == 0]
    if missing:
        raise ValueError(f"Training split has no samples for configured classes: {missing}.")
    return counts


def inverse_frequency_class_weights(
    labels: Sequence[str], class_names: Sequence[str]
) -> torch.Tensor:
    counts = class_counts(labels, class_names).astype(np.float64)
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)


def make_weighted_sampler(
    labels: Sequence[str], class_names: Sequence[str], *, seed: int
) -> WeightedRandomSampler:
    names = list(class_names)
    counts = class_counts(labels, names).astype(np.float64)
    by_name = {name: 1 / count for name, count in zip(names, counts, strict=True)}
    weights = torch.tensor([by_name[label] for label in labels], dtype=torch.double)
    return WeightedRandomSampler(
        weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(int(seed))
    )
