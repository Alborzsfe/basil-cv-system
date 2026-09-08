from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.classification.dataset import BasilClassificationDataset
from src.classification.model import (
    build_classifier,
    find_last_convolution,
    freeze_backbone,
)
from src.inference.pipeline import BasilCVPipeline
from src.segmentation.dataset import BasilSegmentationDataset
from src.segmentation.model import build_segmentation_model
from src.visualization.gradcam import GradCAM, overlay_heatmap


def _write_pair(root: Path) -> tuple[Path, Path]:
    image = np.zeros((24, 30, 3), dtype=np.uint8)
    image[:, :] = (40, 170, 65)
    image[::2, ::2] = (180, 60, 40)
    mask = np.zeros((24, 30), dtype=np.uint8)
    mask[3:21, 5:25] = 255
    image_path, mask_path = root / "leaf.png", root / "leaf_mask.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask)
    return image_path, mask_path


def test_segmentation_dataset_returns_aligned_tensors(tmp_path: Path) -> None:
    image_path, mask_path = _write_pair(tmp_path)
    records = pd.DataFrame(
        [{"image_filename": image_path.name, "mask_filename": mask_path.name, "split": "train"}]
    )
    dataset = BasilSegmentationDataset(
        records,
        images_dir=tmp_path,
        masks_dir=tmp_path,
        split="train",
        image_size=(32, 40),
        augment=True,
        augmentation={"horizontal_flip_prob": 1.0, "rotation_limit_degrees": 4},
    )
    item = dataset[0]
    assert item["image"].shape == (3, 32, 40)
    assert item["mask"].shape == (1, 32, 40)
    assert set(torch.unique(item["mask"]).tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize(
    ("mode", "channels"),
    [("raw_image", 3), ("leaf_only", 3), ("leaf_only_mask_aware", 4)],
)
def test_classification_dataset_input_modes(tmp_path: Path, mode: str, channels: int) -> None:
    image_path, mask_path = _write_pair(tmp_path)
    records = pd.DataFrame(
        [
            {
                "image_path": image_path.name,
                "mask_path": mask_path.name,
                "label": "healthy",
                "split": "train",
            }
        ]
    )
    dataset = BasilClassificationDataset(
        records,
        class_names=["healthy", "spotted"],
        root_dir=tmp_path,
        split="train",
        image_size=(28, 28),
        input_mode=mode,
        augment=True,
        augmentation={"horizontal_flip_prob": 1.0, "brightness_limit": 0.1},
    )
    item = dataset[0]
    assert item["image"].shape == (channels, 28, 28)
    assert item["target"].item() == 0


def test_tiny_models_forward_and_backpropagate() -> None:
    segmenter = build_segmentation_model(
        "tiny_unet", encoder_name="tiny", base_channels=4, encoder_weights=None
    )
    segment_logits = segmenter(torch.randn(2, 3, 32, 32))
    assert segment_logits.shape == (2, 1, 32, 32)
    segment_logits.mean().backward()

    classifier = build_classifier("tiny_cnn", num_classes=3, pretrained=False, in_channels=4)
    class_logits = classifier(torch.randn(2, 4, 32, 32))
    assert class_logits.shape == (2, 3)
    class_logits.mean().backward()


def test_freezing_and_model_factory_validation() -> None:
    model = build_classifier("tiny_cnn", num_classes=2, pretrained=False)
    freeze_backbone(model, "tiny_cnn")
    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert isinstance(find_last_convolution(model), nn.Conv2d)
    with pytest.raises(ValueError):
        build_classifier("missing", num_classes=2)
    with pytest.raises(ValueError):
        build_segmentation_model("missing")


def test_grad_cam_and_overlay() -> None:
    model = build_classifier("tiny_cnn", num_classes=2, pretrained=False)
    tensor = torch.randn(1, 3, 32, 32)
    with GradCAM(model, find_last_convolution(model)) as cam:
        heatmap = cam.generate(tensor)
    assert heatmap.shape == (32, 32)
    assert 0 <= heatmap.min() <= heatmap.max() <= 1
    image = np.full((24, 20, 3), 100, dtype=np.uint8)
    assert overlay_heatmap(image, heatmap).shape == image.shape


class _ConstantSegmenter(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (tensor.shape[0], 1, tensor.shape[2], tensor.shape[3]),
            self.value,
            device=tensor.device,
        )


class _ConstantClassifier(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.new_tensor((3.0, 1.0)).expand(tensor.shape[0], -1)


def _textured_image() -> np.ndarray:
    grid = (np.indices((40, 44)).sum(axis=0) % 2 * 100 + 75).astype(np.uint8)
    return np.stack((grid, np.roll(grid, 1, axis=0), np.roll(grid, 1, axis=1)), axis=2)


def _pipeline(segment_value: float = 10, **options: object) -> BasilCVPipeline:
    return BasilCVPipeline(
        _ConstantSegmenter(segment_value),
        _ConstantClassifier(),
        class_names=["healthy", "spotted"],
        device="cpu",
        segmentation_size=(32, 32),
        classification_size=(32, 32),
        quality_thresholds={"min_laplacian_variance": 0},
        **options,
    )


def test_pipeline_success_unknown_and_artifacts(tmp_path: Path) -> None:
    result = _pipeline().predict(_textured_image(), artifacts_dir=tmp_path)
    assert result.status == "ok"
    assert result.prediction == "healthy"
    assert result.leaf_coverage == 1.0
    assert set(result.artifact_paths) == {
        "original",
        "probability_mask",
        "binary_mask",
        "leaf_only",
        "result",
    }
    saved = json.loads(Path(result.artifact_paths["result"]).read_text())
    assert saved["prediction"] == "healthy"

    uncertain = _pipeline(confidence_threshold=0.99).predict(_textured_image())
    assert uncertain.status == "ok"
    assert uncertain.prediction == "unknown"


def test_pipeline_rejection_paths() -> None:
    rejected = _pipeline().predict(np.zeros((32, 32, 3), dtype=np.uint8))
    assert rejected.status == "quality_rejected"
    assert "too_dark" in rejected.reason
    insufficient = _pipeline(-10, minimum_leaf_coverage=0.1).predict(_textured_image())
    assert insufficient.status == "insufficient_leaf_area"
    assert insufficient.prediction is None


def test_pipeline_configuration_validation() -> None:
    with pytest.raises(ValueError):
        BasilCVPipeline(
            _ConstantSegmenter(1),
            _ConstantClassifier(),
            class_names=["only"],
            device="cpu",
        )
    with pytest.raises(ValueError):
        _pipeline(classification_input_mode="invalid")
