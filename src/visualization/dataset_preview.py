"""Dataset preview visualizations.

Used by the dataset-analysis phase to sanity-check images and masks before
any model training happens. This module only visualizes existing
annotations; it does not perform segmentation inference (see
``src/preprocessing`` for that, once implemented).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.io_utils import ensure_dir

# Semi-transparent green overlay color for visualizing leaf mask pixels.
DEFAULT_OVERLAY_COLOR = (0, 255, 0)


def create_mask_overlay(
    image: np.ndarray,
    binary_mask: np.ndarray,
    overlay_color: Sequence[int] = DEFAULT_OVERLAY_COLOR,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a binary leaf mask on top of an RGB image for visual inspection.

    Args:
        image: RGB image, shape (H, W, 3), dtype uint8.
        binary_mask: Binary mask, shape (H, W), values in {0, 1} (or any
            positive/zero convention; any nonzero value is treated as leaf).
        overlay_color: RGB color used to tint leaf pixels.
        alpha: Blend strength of the overlay color, in [0, 1].

    Returns:
        A new RGB image with the leaf region tinted by ``overlay_color``.
    """
    if image.shape[:2] != binary_mask.shape[:2]:
        raise ValueError(
            f"Image shape {image.shape[:2]} does not match mask shape {binary_mask.shape[:2]}."
        )

    overlay = image.copy().astype(np.float32)
    color_layer = np.zeros_like(overlay)
    color_layer[..., 0] = overlay_color[0]
    color_layer[..., 1] = overlay_color[1]
    color_layer[..., 2] = overlay_color[2]

    leaf_pixels = binary_mask > 0
    overlay[leaf_pixels] = (1 - alpha) * overlay[leaf_pixels] + alpha * color_layer[leaf_pixels]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_sample_grid(
    samples: list[dict],
    output_path: str | Path | None = None,
    max_samples: int = 8,
) -> plt.Figure:
    """Plot a grid of (image, mask, overlay) triplets for a set of samples.

    Args:
        samples: A list of dictionaries, each with keys ``"image"``,
            ``"mask"``, and ``"filename"``.
        output_path: If provided, the figure is also saved to this path.
        max_samples: Maximum number of samples to display.

    Returns:
        The created Matplotlib figure.
    """
    samples = samples[:max_samples]
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * max(num_samples, 1)))
    if num_samples == 1:
        axes = axes.reshape(1, 3)

    for row, sample in enumerate(samples):
        image = sample["image"]
        mask = sample["mask"]
        binary_mask = (mask > 0).astype(np.uint8)
        overlay = create_mask_overlay(image, binary_mask)

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Image: {sample.get('filename', row)}", fontsize=8)
        axes[row, 1].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Binary mask", fontsize=8)
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay", fontsize=8)
        for col in range(3):
            axes[row, col].axis("off")

    fig.tight_layout()
    if output_path is not None:
        output_file = Path(output_path)
        ensure_dir(output_file.parent)
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
    return fig


def plot_leaf_coverage_histogram(
    leaf_coverages: Sequence[float],
    output_path: str | Path | None = None,
    bins: int = 30,
) -> plt.Figure:
    """Plot a histogram of per-image leaf coverage values.

    Useful for spotting datasets dominated by near-empty or near-full
    masks, which can bias segmentation training.
    """
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(leaf_coverages, bins=bins, color="#2e7d32", edgecolor="black")
    axis.set_xlabel("Leaf coverage (fraction of image)")
    axis.set_ylabel("Number of images")
    axis.set_title("Leaf Coverage Distribution")
    fig.tight_layout()
    if output_path is not None:
        output_file = Path(output_path)
        ensure_dir(output_file.parent)
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
    return fig
