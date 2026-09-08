#!/usr/bin/env python
"""Run a deterministic, synthetic end-to-end smoke test.

This verifies software wiring only. The generated image and deterministic
models are not biological data and the output is not a scientific result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.pipeline import BasilCVPipeline
from src.utils.io_utils import ensure_dir, save_json


class SyntheticGreenSegmenter(nn.Module):
    """Deterministically identify green-dominant pixels in normalized RGB."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        red, green, blue = (tensor * self.std + self.mean).unbind(dim=1)
        green_excess = green - torch.maximum(red, blue)
        return (green_excess * 30).unsqueeze(1)


class SyntheticPatternClassifier(nn.Module):
    """Return deterministic logits solely to exercise the classifier stage."""

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch = tensor.shape[0]
        logits = tensor.new_tensor((1.4, 0.0))
        return logits.expand(batch, -1)


def make_synthetic_image() -> np.ndarray:
    """Create a textured scene with one stylized green leaf."""
    rng = np.random.default_rng(2026)
    image = np.full((256, 256, 3), (108, 126, 148), dtype=np.uint8)
    for y in range(0, 256, 16):
        image[y : y + 8] = (125, 137, 151)
    noise = rng.integers(-12, 13, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.ellipse(image, (128, 128), (74, 116), 0, 0, 360, (43, 174, 72), -1)
    cv2.line(image, (128, 28), (128, 227), (35, 108, 48), 3)
    for center in ((94, 91), (157, 112), (104, 161), (151, 184)):
        cv2.circle(image, center, 8, (126, 74, 42), -1)
    return image


def _relative_paths(paths: dict[str, str], root: Path) -> dict[str, str]:
    relative: dict[str, str] = {}
    for name, value in paths.items():
        path = Path(value).resolve()
        relative[name] = path.relative_to(root).as_posix()
    return relative


def _write_montage(original: np.ndarray, mask_path: Path, leaf_path: Path, output: Path) -> None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    leaf_bgr = cv2.imread(str(leaf_path), cv2.IMREAD_COLOR)
    if mask is None or leaf_bgr is None:
        raise OSError("Smoke-test artifacts could not be read back.")
    panels = [
        cv2.cvtColor(original, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
        leaf_bgr,
    ]
    labels = ("synthetic input", "binary mask", "leaf only")
    for panel, label in zip(panels, labels, strict=True):
        cv2.rectangle(panel, (0, 0), (256, 29), (20, 20, 20), -1)
        cv2.putText(panel, label, (9, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1)
    if not cv2.imwrite(str(output), np.concatenate(panels, axis=1)):
        raise OSError(f"Could not write montage: {output}.")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    destination = ensure_dir(root / "docs" / "results" / "smoke")
    image = make_synthetic_image()
    pipeline = BasilCVPipeline(
        SyntheticGreenSegmenter(),
        SyntheticPatternClassifier(),
        class_names=("synthetic_healthy_like", "synthetic_spotted_like"),
        device="cpu",
        segmentation_size=(128, 128),
        classification_size=(96, 96),
        confidence_threshold=0.6,
        minimum_leaf_coverage=0.05,
        mask_kernel_size=5,
        minimum_component_area=64,
    )
    result = pipeline.predict(image, artifacts_dir=destination)
    if result.status != "ok":
        raise RuntimeError(f"Synthetic smoke test failed: {result.to_dict()}.")

    original_paths = dict(result.artifact_paths)
    run_directory = Path(original_paths["result"]).parent
    montage = destination / "pipeline_montage.png"
    _write_montage(
        image,
        Path(original_paths["binary_mask"]),
        Path(original_paths["leaf_only"]),
        montage,
    )
    result.artifact_paths = _relative_paths(original_paths, root)
    result.artifact_paths["montage"] = montage.relative_to(root).as_posix()
    save_json(result.to_dict(), run_directory / "result.json")
    summary = {
        "scope": "synthetic functional smoke test only",
        "scientific_claim": False,
        "result": result.to_dict(),
    }
    save_json(summary, destination / "summary.json")
    print(
        f"Synthetic smoke test passed: status={result.status}, "
        f"prediction={result.prediction}, coverage={result.leaf_coverage:.4f}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
