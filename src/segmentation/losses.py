"""Loss functions for imbalanced binary leaf segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        dims = tuple(range(1, probability.ndim))
        intersection = (probability * target).sum(dim=dims)
        denominator = probability.sum(dim=dims) + target.sum(dim=dims)
        return 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5) -> None:
        super().__init__()
        if dice_weight < 0 or bce_weight < 0 or dice_weight + bce_weight <= 0:
            raise ValueError("Loss weights must be non-negative with a positive sum.")
        total = dice_weight + bce_weight
        self.dice_weight, self.bce_weight = dice_weight / total, bce_weight / total
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(
            logits, target
        ) + self.bce_weight * F.binary_cross_entropy_with_logits(logits, target)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        if not 0 <= alpha <= 1 or gamma < 0:
            raise ValueError("Focal alpha must be in [0, 1] and gamma cannot be negative.")
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probability = torch.sigmoid(logits)
        p_true = probability * target + (1 - probability) * (1 - target)
        alpha_true = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_true * (1 - p_true).pow(self.gamma) * bce).mean()


def build_segmentation_loss(
    loss_type: str = "dice_bce",
    *,
    dice_weight: float = 0.5,
    bce_weight: float = 0.5,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> nn.Module:
    if loss_type == "dice_bce":
        return DiceBCELoss(dice_weight, bce_weight)
    if loss_type == "dice":
        return DiceLoss()
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_type == "focal":
        return BinaryFocalLoss(focal_alpha, focal_gamma)
    raise ValueError(f"Unsupported segmentation loss '{loss_type}'.")
