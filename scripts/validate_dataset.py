#!/usr/bin/env python
"""Validate the basil segmentation dataset and generate a dataset report.

This script must be run before any segmentation training. It checks for:

* Missing masks / orphan masks
* Corrupt images or masks
* Image/mask dimension mismatches
* Empty masks
* Masks with unexpected pixel values
* Duplicate/ambiguous filenames

and generates a Markdown + JSON dataset report with image/mask counts,
dimensions, and leaf coverage statistics.

Usage:
    python scripts/validate_dataset.py --config configs/segmentation.yaml
    python scripts/validate_dataset.py --config configs/segmentation.yaml --allow-warnings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.report import build_dataset_report, save_dataset_report
from src.data.validation import DatasetValidator
from src.utils.config import load_config
from src.utils.io_utils import find_image_files
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the basil segmentation dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/segmentation.yaml",
        help="Path to the segmentation configuration file.",
    )
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Exit successfully even if warning-level issues (e.g. empty masks) are found. "
        "Error-level issues (e.g. missing masks, corrupt files) always cause a nonzero exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    images_dir = config.dataset.images_dir
    masks_dir = config.dataset.masks_dir
    output_dir = Path(config.experiment.output_dir) / "dataset_report"

    logger.info("Validating dataset: images='%s', masks='%s'", images_dir, masks_dir)

    validator = DatasetValidator(images_dir=images_dir, masks_dir=masks_dir)
    validation_result = validator.validate()

    num_image_files = len(find_image_files(images_dir))
    num_mask_files = len(find_image_files(masks_dir))

    report = build_dataset_report(validation_result, num_image_files, num_mask_files)
    output_paths = save_dataset_report(report, validation_result, output_dir)

    logger.info("Report written to:")
    for kind, path in output_paths.items():
        logger.info("  - %s: %s", kind, path)

    logger.info(
        "Summary: %d valid pairs, %d errors, %d warnings, mean leaf coverage=%.4f",
        report.num_valid_pairs,
        report.num_errors,
        report.num_warnings,
        report.mean_leaf_coverage,
    )

    if validation_result.errors:
        logger.error(
            "Dataset validation FAILED with %d error(s). Training cannot proceed until "
            "these are fixed. See the report for details.",
            len(validation_result.errors),
        )
        return 1

    if validation_result.warnings and not args.allow_warnings:
        logger.warning(
            "Dataset validation found %d warning(s). Re-run with --allow-warnings to "
            "treat this as a passing validation, once the warnings have been reviewed.",
            len(validation_result.warnings),
        )
        return 1

    logger.info("Dataset validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
