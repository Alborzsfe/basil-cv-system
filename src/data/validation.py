"""Segmentation dataset validation.

This module scans an ``images/`` and ``masks/`` directory pair and detects
problems that must be fixed before any training run, including:

* Missing masks (an image with no corresponding mask file)
* Orphan masks (a mask with no corresponding image file)
* Corrupt / unreadable images or masks
* Mask/image spatial dimension mismatches
* Empty masks (no visible leaf pixels at all)
* Masks containing pixel values outside the expected set
* Duplicate/ambiguous filenames (same stem, different extensions)

No file is ever silently discarded: every problem is recorded as a
:class:`ValidationIssue` with a severity level, and it is the caller's
responsibility to decide whether to abort (see ``scripts/validate_dataset.py``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.utils.io_utils import ImageReadError, find_image_files, safe_read_image, safe_read_mask
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Pixel value sets accepted for a binary leaf mask. Some annotation tools
# export 0/1, others export 0/255. Both are accepted and normalized.
ACCEPTED_BINARY_VALUE_SETS: list[set[int]] = [{0}, {1}, {0, 1}, {0, 255}, {255}]

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


@dataclass
class ValidationIssue:
    """A single problem found while validating the dataset."""

    severity: str  # "error" or "warning"
    category: str
    filename: str
    message: str


@dataclass
class ImageStats:
    """Per-image statistics collected during validation, used for the dataset report."""

    image_filename: str
    mask_filename: str
    image_height: int
    image_width: int
    mask_height: int
    mask_width: int
    leaf_pixels: int
    total_pixels: int
    leaf_coverage: float
    is_empty_mask: bool


@dataclass
class ValidationResult:
    """The full outcome of validating a segmentation dataset."""

    issues: list[ValidationIssue] = field(default_factory=list)
    per_image_stats: list[ImageStats] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == SEVERITY_ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == SEVERITY_WARNING]

    @property
    def is_valid(self) -> bool:
        """A dataset is considered valid for training if it has no error-level issues."""
        return len(self.errors) == 0


class DatasetValidator:
    """Validates a ``images/`` + ``masks/`` directory pair for the basil segmentation dataset."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        mask_suffix_map: dict[str, str] | None = None,
    ) -> None:
        """Initialize the validator.

        Args:
            images_dir: Directory containing RGB images.
            masks_dir: Directory containing binary segmentation masks.
            mask_suffix_map: Optional mapping used to derive a mask filename
                from an image filename when they do not share an identical
                stem (e.g. ``{"_image": "_mask"}``). If None, masks are
                matched purely by filename stem (extension-independent).
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix_map = mask_suffix_map or {}
        if any(not suffix for suffix in self.mask_suffix_map):
            raise ValueError("mask_suffix_map keys cannot be empty.")

    def _mask_stem_for_image(self, image_stem: str) -> str:
        for image_suffix, mask_suffix in self.mask_suffix_map.items():
            if image_stem.lower().endswith(image_suffix.lower()):
                return f"{image_stem[: -len(image_suffix)]}{mask_suffix}"
        return image_stem

    def validate(self) -> ValidationResult:
        """Run all validation checks and return the aggregated result."""
        result = ValidationResult()

        image_files = find_image_files(self.images_dir)
        mask_files = find_image_files(self.masks_dir)

        if not image_files:
            result.issues.append(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    category="empty_dataset",
                    filename=str(self.images_dir),
                    message="No image files found in the images directory.",
                )
            )
            return result

        self._check_duplicate_stems(image_files, "image", result)
        self._check_duplicate_stems(mask_files, "mask", result)

        mask_by_stem = {mask_path.stem.lower(): mask_path for mask_path in mask_files}
        expected_mask_stems = {
            self._mask_stem_for_image(image_path.stem).lower() for image_path in image_files
        }

        # Check for orphan masks (a mask with no matching image).
        for mask_path in mask_files:
            if mask_path.stem.lower() not in expected_mask_stems:
                result.issues.append(
                    ValidationIssue(
                        severity=SEVERITY_WARNING,
                        category="orphan_mask",
                        filename=mask_path.name,
                        message="Mask file has no corresponding image file.",
                    )
                )

        for image_path in image_files:
            self._validate_single_pair(image_path, mask_by_stem, result)

        logger.info(
            "Validation finished: %d images scanned, %d errors, %d warnings.",
            len(image_files),
            len(result.errors),
            len(result.warnings),
        )
        return result

    def _validate_single_pair(
        self,
        image_path: Path,
        mask_by_stem: dict[str, Path],
        result: ValidationResult,
    ) -> None:
        """Validate a single image and its corresponding mask, if present."""
        expected_stem = self._mask_stem_for_image(image_path.stem)
        mask_path = mask_by_stem.get(expected_stem.lower())
        if mask_path is None:
            result.issues.append(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    category="missing_mask",
                    filename=image_path.name,
                    message="No corresponding mask file found for this image.",
                )
            )
            return

        try:
            image = safe_read_image(image_path)
        except ImageReadError as exc:
            result.issues.append(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    category="corrupt_image",
                    filename=image_path.name,
                    message=str(exc),
                )
            )
            return

        try:
            raw_mask = safe_read_mask(mask_path)
        except ImageReadError as exc:
            result.issues.append(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    category="corrupt_mask",
                    filename=mask_path.name,
                    message=str(exc),
                )
            )
            return

        image_height, image_width = image.shape[:2]
        mask_height, mask_width = raw_mask.shape[:2]

        if (mask_height, mask_width) != (image_height, image_width):
            result.issues.append(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    category="dimension_mismatch",
                    filename=mask_path.name,
                    message=(
                        f"Mask dimensions ({mask_height}x{mask_width}) do not match "
                        f"image dimensions ({image_height}x{image_width})."
                    ),
                )
            )
            return

        unique_values: set[int] = set(np.unique(raw_mask).tolist())
        if unique_values not in ACCEPTED_BINARY_VALUE_SETS:
            result.issues.append(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    category="unexpected_mask_values",
                    filename=mask_path.name,
                    message=(
                        f"Mask contains unexpected pixel values {sorted(unique_values)}; "
                        f"expected a binary mask (0/1 or 0/255)."
                    ),
                )
            )
            return

        binary_mask = (raw_mask > 0).astype(np.uint8)
        leaf_pixels = int(binary_mask.sum())
        total_pixels = int(binary_mask.size)
        leaf_coverage = leaf_pixels / total_pixels if total_pixels > 0 else 0.0
        is_empty_mask = leaf_pixels == 0

        if is_empty_mask:
            result.issues.append(
                ValidationIssue(
                    severity=SEVERITY_WARNING,
                    category="empty_mask",
                    filename=mask_path.name,
                    message="Mask contains no leaf pixels (fully background).",
                )
            )

        result.per_image_stats.append(
            ImageStats(
                image_filename=image_path.name,
                mask_filename=mask_path.name,
                image_height=image_height,
                image_width=image_width,
                mask_height=mask_height,
                mask_width=mask_width,
                leaf_pixels=leaf_pixels,
                total_pixels=total_pixels,
                leaf_coverage=leaf_coverage,
                is_empty_mask=is_empty_mask,
            )
        )

    @staticmethod
    def _check_duplicate_stems(files: list[Path], file_kind: str, result: ValidationResult) -> None:
        """Flag filenames that share a stem but differ only by extension or case.

        For example ``leaf_001.jpg`` and ``leaf_001.PNG`` present in the same
        directory are ambiguous: it is unclear which one should be used.
        """
        stem_counts: dict[str, list[Path]] = defaultdict(list)
        for file_path in files:
            stem_counts[file_path.stem.lower()].append(file_path)

        for stem, paths in stem_counts.items():
            if len(paths) > 1:
                result.issues.append(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        category="duplicate_filename",
                        filename=", ".join(p.name for p in paths),
                        message=(
                            f"Multiple {file_kind} files share the same stem '{stem}': "
                            f"{[p.name for p in paths]}. This is ambiguous and must be resolved."
                        ),
                    )
                )
