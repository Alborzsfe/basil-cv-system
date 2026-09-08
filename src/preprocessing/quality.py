"""Deterministic image-quality checks used before model inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageQualityReport:
    """Measured image quality and any rejection reasons."""

    passed: bool
    mean_brightness: float
    laplacian_variance: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def assess_image_quality(
    image_rgb: np.ndarray,
    *,
    min_mean_brightness: float = 20.0,
    max_mean_brightness: float = 235.0,
    min_laplacian_variance: float = 30.0,
) -> ImageQualityReport:
    """Reject images that are too dark, too bright, or severely blurred."""
    image = _validate_rgb_image(image_rgb)
    if min_mean_brightness < 0 or max_mean_brightness > 255:
        raise ValueError("Brightness thresholds must be within the 0-255 range.")
    if min_mean_brightness >= max_mean_brightness:
        raise ValueError("Minimum brightness must be lower than maximum brightness.")
    if min_laplacian_variance < 0:
        raise ValueError("Minimum Laplacian variance cannot be negative.")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(gray.mean())
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons: list[str] = []
    if mean_brightness < min_mean_brightness:
        reasons.append("too_dark")
    if mean_brightness > max_mean_brightness:
        reasons.append("too_bright")
    if laplacian_variance < min_laplacian_variance:
        reasons.append("too_blurry")
    return ImageQualityReport(
        passed=not reasons,
        mean_brightness=mean_brightness,
        laplacian_variance=laplacian_variance,
        reasons=tuple(reasons),
    )


def _validate_rgb_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image_rgb must be a NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError(f"Expected a non-empty RGB image, got shape {image.shape}.")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError("Image values must be finite numeric values.")
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image
