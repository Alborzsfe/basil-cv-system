"""Safe file and image I/O helpers.

Centralizing image reading here ensures corrupt or unreadable files are
handled consistently (never silently) across dataset validation,
preprocessing, and inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ImageReadError(Exception):
    """Raised when an image file cannot be read or decoded."""


def safe_read_image(path: str | Path, color_mode: str = "rgb") -> np.ndarray:
    """Read an image from disk, raising a clear error instead of failing silently.

    Args:
        path: Path to the image file.
        color_mode: Either "rgb" or "bgr". OpenCV loads images as BGR by
            default; "rgb" converts to the more commonly expected order.

    Returns:
        The image as a NumPy array of shape (H, W, 3), dtype uint8.

    Raises:
        ImageReadError: If the file does not exist, cannot be decoded, or
            is not a valid 3-channel image.
    """
    image_path = Path(path)
    if not image_path.exists():
        raise ImageReadError(f"Image file does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageReadError(f"Failed to decode image (possibly corrupt): {image_path}")

    if color_mode == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_mode != "bgr":
        raise ValueError(f"Unsupported color_mode '{color_mode}', expected 'rgb' or 'bgr'.")

    return image


def safe_read_mask(path: str | Path) -> np.ndarray:
    """Read a segmentation mask as a single-channel array.

    Args:
        path: Path to the mask file.

    Returns:
        A 2D NumPy array (H, W) with the raw pixel values found in the file.

    Raises:
        ImageReadError: If the file does not exist or cannot be decoded.
    """
    mask_path = Path(path)
    if not mask_path.exists():
        raise ImageReadError(f"Mask file does not exist: {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ImageReadError(f"Failed to decode mask (possibly corrupt): {mask_path}")

    if mask.ndim == 3:
        # Some annotation tools export masks as 3-channel images with the
        # class encoded identically in every channel. Reduce to a single
        # channel rather than silently using only one arbitrary channel.
        if not all(
            np.array_equal(mask[..., 0], mask[..., channel]) for channel in range(1, mask.shape[2])
        ):
            raise ImageReadError(f"Multi-channel mask has inconsistent channel values: {mask_path}")
        mask = mask[..., 0]

    return mask


def save_json(data: dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """Save a dictionary as a human-readable JSON file, creating parents as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=indent, ensure_ascii=True, default=_json_default)


def _json_default(value: Any) -> Any:
    """Fallback JSON encoder for NumPy scalar/array types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable.")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (including parents) if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def find_image_files(directory: str | Path, extensions: tuple | None = None) -> list:
    """List image files in a directory sorted by filename.

    Args:
        directory: Directory to scan (non-recursive).
        extensions: Allowed file extensions (lowercase, with leading dot).
            Defaults to common raster formats.

    Returns:
        A sorted list of :class:`Path` objects.
    """
    if extensions is None:
        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    directory_path = Path(directory)
    if not directory_path.exists():
        return []
    return sorted(
        p for p in directory_path.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )
