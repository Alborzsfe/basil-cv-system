"""Raw RGB image to quality-gated, explainable basil disease prediction."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from src.classification.model import find_last_convolution
from src.preprocessing.background import apply_background, clean_binary_mask, leaf_coverage
from src.preprocessing.quality import ImageQualityReport, assess_image_quality
from src.segmentation.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.segmentation.engine import resolve_device
from src.utils.io_utils import ensure_dir, safe_read_image, save_json
from src.visualization.gradcam import GradCAM, overlay_heatmap


@dataclass
class InferenceResult:
    status: str
    prediction: str | None
    confidence: float | None
    probabilities: dict[str, float]
    leaf_coverage: float | None
    quality: ImageQualityReport
    artifact_paths: dict[str, str] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["quality"] = self.quality.to_dict()
        return output


class BasilCVPipeline:
    def __init__(
        self,
        segmentation_model: nn.Module,
        classification_model: nn.Module,
        *,
        class_names: Sequence[str],
        device: str | torch.device = "auto",
        segmentation_size: Sequence[int] = (512, 512),
        classification_size: Sequence[int] = (384, 384),
        classification_input_mode: str = "leaf_only",
        segmentation_threshold: float = 0.5,
        confidence_threshold: float = 0.6,
        minimum_leaf_coverage: float = 0.02,
        background_mode: str = "black",
        gray_value: int = 128,
        background_blur_kernel_size: int = 31,
        mask_kernel_size: int = 5,
        minimum_component_area: int = 0,
        quality_thresholds: Mapping[str, float] | None = None,
    ) -> None:
        names = list(class_names)
        if len(names) < 2 or len(names) != len(set(names)):
            raise ValueError("class_names must contain at least two unique labels.")
        if classification_input_mode not in {"raw_image", "leaf_only", "leaf_only_mask_aware"}:
            raise ValueError(f"Unsupported classification input mode: {classification_input_mode}.")
        if not 0 <= segmentation_threshold <= 1 or not 0 <= confidence_threshold <= 1:
            raise ValueError("Probability thresholds must be between zero and one.")
        if not 0 <= minimum_leaf_coverage <= 1:
            raise ValueError("minimum_leaf_coverage must be between zero and one.")
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.segmentation_model = segmentation_model.to(self.device).eval()
        self.classification_model = classification_model.to(self.device).eval()
        self.class_names = names
        self.segmentation_size = _size(segmentation_size)
        self.classification_size = _size(classification_size)
        self.classification_input_mode = classification_input_mode
        self.segmentation_threshold = segmentation_threshold
        self.confidence_threshold = confidence_threshold
        self.minimum_leaf_coverage = minimum_leaf_coverage
        self.background_mode = background_mode
        self.gray_value = gray_value
        self.background_blur_kernel_size = background_blur_kernel_size
        self.mask_kernel_size = mask_kernel_size
        self.minimum_component_area = minimum_component_area
        self.quality_thresholds = dict(quality_thresholds or {})

    def predict(
        self,
        image: str | Path | np.ndarray,
        *,
        artifacts_dir: str | Path | None = None,
        generate_grad_cam: bool = False,
    ) -> InferenceResult:
        image_rgb = (
            safe_read_image(image) if isinstance(image, (str, Path)) else _validate_image(image)
        )
        quality = assess_image_quality(image_rgb, **self.quality_thresholds)
        if not quality.passed:
            result = InferenceResult(
                "quality_rejected", None, None, {}, None, quality, reason=",".join(quality.reasons)
            )
            self._save_result(image_rgb, None, None, None, result, artifacts_dir)
            return result
        segment_input = _image_tensor(image_rgb, self.segmentation_size).to(self.device)
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.segmentation_model(segment_input))
        if probabilities.ndim != 4 or probabilities.shape[1] != 1:
            raise ValueError("Segmentation model must output shape (N, 1, H, W).")
        probability_mask = cv2.resize(
            probabilities[0, 0].cpu().numpy(), (image_rgb.shape[1], image_rgb.shape[0])
        )
        binary = clean_binary_mask(
            probability_mask,
            threshold=self.segmentation_threshold,
            kernel_size=self.mask_kernel_size,
            minimum_component_area=self.minimum_component_area,
        )
        coverage = leaf_coverage(binary)
        leaf_only = apply_background(
            image_rgb,
            binary,
            mode=self.background_mode,
            gray_value=self.gray_value,
            blur_kernel_size=self.background_blur_kernel_size,
        )
        if coverage < self.minimum_leaf_coverage:
            result = InferenceResult(
                "insufficient_leaf_area",
                None,
                None,
                {},
                coverage,
                quality,
                reason="insufficient_visible_leaf_area",
            )
            self._save_result(image_rgb, probability_mask, binary, leaf_only, result, artifacts_dir)
            return result
        classifier_image = image_rgb if self.classification_input_mode == "raw_image" else leaf_only
        classifier_input = _image_tensor(classifier_image, self.classification_size)
        if self.classification_input_mode == "leaf_only_mask_aware":
            resized_mask = cv2.resize(
                binary,
                (self.classification_size[1], self.classification_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            classifier_input = torch.cat(
                (classifier_input, torch.from_numpy(resized_mask[None, None]).float()), dim=1
            )
        classifier_input = classifier_input.to(self.device)
        with torch.inference_mode():
            logits = self.classification_model(classifier_input)
        if logits.ndim != 2 or logits.shape != (1, len(self.class_names)):
            raise ValueError("Classifier output shape does not match class_names.")
        with torch.inference_mode():
            scores = torch.softmax(logits, dim=1)[0]
        confidence, predicted_index = torch.max(scores, 0)
        confidence_value = float(confidence.cpu())
        prediction = (
            self.class_names[int(predicted_index)]
            if confidence_value >= self.confidence_threshold
            else "unknown"
        )
        probability_map = {name: float(scores[i].cpu()) for i, name in enumerate(self.class_names)}
        grad_overlay = None
        if generate_grad_cam:
            layer = find_last_convolution(self.classification_model)
            with GradCAM(self.classification_model, layer) as cam:
                heatmap = cam.generate(classifier_input, int(predicted_index))
            grad_overlay = overlay_heatmap(classifier_image, heatmap)
        result = InferenceResult(
            "ok", prediction, confidence_value, probability_map, coverage, quality
        )
        self._save_result(
            image_rgb, probability_mask, binary, leaf_only, result, artifacts_dir, grad_overlay
        )
        return result

    def _save_result(
        self,
        original: np.ndarray,
        probability_mask: np.ndarray | None,
        binary_mask: np.ndarray | None,
        leaf_only: np.ndarray | None,
        result: InferenceResult,
        artifacts_dir: str | Path | None,
        grad_cam_overlay: np.ndarray | None = None,
    ) -> None:
        if artifacts_dir is None:
            return
        destination = ensure_dir(
            Path(artifacts_dir) / hashlib.sha256(original.tobytes()).hexdigest()[:12]
        )
        paths = {"original": _write_rgb(destination / "original.png", original)}
        if probability_mask is not None:
            paths["probability_mask"] = _write_gray(
                destination / "probability_mask.png",
                np.clip(probability_mask * 255, 0, 255).astype(np.uint8),
            )
        if binary_mask is not None:
            paths["binary_mask"] = _write_gray(destination / "binary_mask.png", binary_mask * 255)
        if leaf_only is not None:
            paths["leaf_only"] = _write_rgb(destination / "leaf_only.png", leaf_only)
        if grad_cam_overlay is not None:
            paths["grad_cam"] = _write_rgb(destination / "grad_cam.png", grad_cam_overlay)
        result_path = destination / "result.json"
        paths["result"] = str(result_path)
        result.artifact_paths = paths
        save_json(result.to_dict(), result_path)


def _image_tensor(image: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    resized = cv2.resize(image, (width, height))
    normalized = (resized.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))[None].float()


def _validate_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or not array.size:
        raise ValueError(f"Expected a non-empty RGB image, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Image contains non-finite values.")
    return np.clip(array, 0, 255).astype(np.uint8)


def _size(values: Sequence[int]) -> tuple[int, int]:
    if len(values) != 2 or min(int(value) for value in values) <= 0:
        raise ValueError("Image sizes must contain two positive integers.")
    return int(values[0]), int(values[1])


def _write_rgb(path: Path, image: np.ndarray) -> str:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write image artifact: {path}.")
    return str(path)


def _write_gray(path: Path, image: np.ndarray) -> str:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write mask artifact: {path}.")
    return str(path)
