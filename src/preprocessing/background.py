"""Binary-mask cleanup and reproducible background removal."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

BackgroundMode = Literal["black", "gray", "blur"]


def normalize_binary_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Normalize 0/1, 0/255, or probability masks to uint8 values 0/1."""
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"Expected a non-empty 2D mask, got shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Mask values must be finite numeric values.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    maximum = float(array.max(initial=0))
    cutoff = threshold * 255.0 if maximum > 1.0 else threshold
    return (array >= cutoff).astype(np.uint8)


def clean_binary_mask(
    mask: np.ndarray,
    *,
    threshold: float = 0.5,
    kernel_size: int = 5,
    minimum_component_area: int = 0,
) -> np.ndarray:
    """Remove speckles, close small holes, and discard tiny components."""
    binary = normalize_binary_mask(mask, threshold)
    if kernel_size < 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be zero or a positive odd integer.")
    if minimum_component_area < 0:
        raise ValueError("minimum_component_area cannot be negative.")
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if minimum_component_area > 0 and binary.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        cleaned = np.zeros_like(binary)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_component_area:
                cleaned[labels == label] = 1
        binary = cleaned
    return binary


def leaf_coverage(mask: np.ndarray, threshold: float = 0.5) -> float:
    return float(normalize_binary_mask(mask, threshold).mean())


def apply_background(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    mode: BackgroundMode = "black",
    gray_value: int = 128,
    blur_kernel_size: int = 31,
) -> np.ndarray:
    """Keep foreground leaf pixels and replace only the background."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}.")
    binary = normalize_binary_mask(mask)
    if binary.shape != image.shape[:2]:
        raise ValueError(f"Image/mask dimensions differ: {image.shape[:2]} vs {binary.shape}.")
    if mode not in {"black", "gray", "blur"}:
        raise ValueError(f"Unsupported background mode: {mode}.")
    if not 0 <= gray_value <= 255:
        raise ValueError("gray_value must be between 0 and 255.")
    foreground = binary.astype(bool)[..., None]
    if mode == "black":
        background = np.zeros_like(image)
    elif mode == "gray":
        background = np.full_like(image, gray_value)
    else:
        if blur_kernel_size < 3 or blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be an odd integer of at least 3.")
        background = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
    return np.where(foreground, image, background).astype(image.dtype, copy=False)
