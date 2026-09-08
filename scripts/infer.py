#!/usr/bin/env python
"""Run quality-gated end-to-end inference with trusted local checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.engine import load_classification_checkpoint
from src.classification.model import build_classifier
from src.inference.pipeline import BasilCVPipeline
from src.segmentation.engine import load_segmentation_checkpoint, resolve_device
from src.segmentation.model import build_segmentation_model
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer basil disease from one RGB image.")
    parser.add_argument("image")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-grad-cam", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    segment_config = load_config(config.segmentation_config)
    class_config = load_config(config.classification_config)
    device = resolve_device(args.device or config.device)
    segment_path = Path(config.segmentation_checkpoint)
    segment_checkpoint = torch.load(segment_path, map_location=device, weights_only=True)
    segmenter = build_segmentation_model(**segment_checkpoint["model_config"])
    load_segmentation_checkpoint(segment_path, segmenter, device=device)
    class_path = Path(config.classification_checkpoint)
    class_checkpoint = torch.load(class_path, map_location=device, weights_only=True)
    classifier = build_classifier(**class_checkpoint["model_config"])
    load_classification_checkpoint(class_path, classifier, device=device)
    pipeline = BasilCVPipeline(
        segmenter,
        classifier,
        class_names=class_checkpoint["class_names"],
        device=device,
        segmentation_size=segment_config.dataset.image_size,
        classification_size=class_config.dataset.image_size,
        classification_input_mode=class_checkpoint["input_mode"],
        segmentation_threshold=float(segment_config.metrics.threshold),
        confidence_threshold=float(config.classification.confidence_threshold),
        minimum_leaf_coverage=float(config.leaf_coverage.minimum_required_coverage),
        background_mode=config.background_removal.mode,
        gray_value=int(config.background_removal.gray_value),
        background_blur_kernel_size=int(config.background_removal.blur_kernel_size),
        mask_kernel_size=int(config.mask_cleanup.kernel_size),
        minimum_component_area=int(config.mask_cleanup.minimum_component_area),
        quality_thresholds=config.image_quality.to_dict(),
    )
    artifacts = (
        config.output.artifacts_dir if bool(config.output.save_intermediate_artifacts) else None
    )
    result = pipeline.predict(
        args.image,
        artifacts_dir=artifacts,
        generate_grad_cam=bool(config.explainability.generate_grad_cam) and not args.no_grad_cam,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
