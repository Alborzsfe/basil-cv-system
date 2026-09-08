"""Cross-entropy variants for class-imbalanced disease classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MulticlassFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("Focal gamma cannot be negative.")
        self.gamma = gamma
        self.register_buffer("class_weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, target, weight=self.class_weight, reduction="none")
        return ((1 - torch.exp(-cross_entropy)).pow(self.gamma) * cross_entropy).mean()


def build_classification_loss(
    loss_type: str = "cross_entropy",
    *,
    focal_gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss_type == "weighted_cross_entropy":
        if class_weights is None:
            raise ValueError("weighted_cross_entropy requires class_weights.")
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "focal":
        return MulticlassFocalLoss(focal_gamma, class_weights)
    raise ValueError(f"Unsupported classification loss '{loss_type}'.")
