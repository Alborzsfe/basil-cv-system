"""Minimal Grad-CAM implementation without a third-party dependency."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(
        self, _module: nn.Module, _inputs: object, output: torch.Tensor
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("The Grad-CAM target layer must return one tensor.")
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def generate(self, input_tensor: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("Grad-CAM expects shape (1, C, H, W).")
        self.activations = self.gradients = None
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        target = int(logits.argmax(1).item()) if class_index is None else int(class_index)
        if target < 0 or target >= logits.shape[1]:
            raise ValueError("class_index is outside the classifier output range.")
        logits[0, target].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Target layer did not produce Grad-CAM data.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = torch.nn.functional.interpolate(
            heatmap, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        heatmap -= heatmap.min()
        maximum = heatmap.max()
        if float(maximum.detach()) > 0:
            heatmap /= maximum
        return heatmap.detach().cpu().numpy().astype(np.float32)

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def overlay_heatmap(
    image_rgb: np.ndarray, heatmap: np.ndarray, *, alpha: float = 0.4
) -> np.ndarray:
    image = np.asarray(image_rgb)
    values = np.asarray(heatmap, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or values.ndim != 2:
        raise ValueError("Expected an RGB image and a two-dimensional heatmap.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    values = cv2.resize(values, (image.shape[1], image.shape[0]))
    colored = cv2.cvtColor(
        cv2.applyColorMap((np.clip(values, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    return cv2.addWeighted(image.astype(np.uint8), 1 - alpha, colored, alpha, 0)
