"""Dataset report generation.

Produces the summary statistics required before any segmentation training
run: image/mask counts, dimension ranges, empty-mask ratio, and leaf
coverage statistics. The report is saved as both JSON (machine-readable)
and Markdown (human-readable) files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.validation import ImageStats, ValidationResult
from src.utils.io_utils import save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetReport:
    """Summary statistics for a segmentation dataset."""

    num_images: int
    num_masks: int
    num_valid_pairs: int
    num_errors: int
    num_warnings: int
    image_dimensions: list[list[int]]
    mask_dimensions: list[list[int]]
    percentage_empty_masks: float
    mean_leaf_coverage: float
    min_leaf_coverage: float
    max_leaf_coverage: float
    median_leaf_coverage: float


def build_dataset_report(
    validation_result: ValidationResult, num_image_files: int, num_mask_files: int
) -> DatasetReport:
    """Compute a :class:`DatasetReport` from a completed validation pass.

    Args:
        validation_result: The result returned by
            :meth:`src.data.validation.DatasetValidator.validate`.
        num_image_files: Total number of image files found on disk.
        num_mask_files: Total number of mask files found on disk.

    Returns:
        The aggregated :class:`DatasetReport`.
    """
    stats: list[ImageStats] = validation_result.per_image_stats

    if not stats:
        logger.warning("No valid image/mask pairs were available to compute report statistics.")
        return DatasetReport(
            num_images=num_image_files,
            num_masks=num_mask_files,
            num_valid_pairs=0,
            num_errors=len(validation_result.errors),
            num_warnings=len(validation_result.warnings),
            image_dimensions=[],
            mask_dimensions=[],
            percentage_empty_masks=0.0,
            mean_leaf_coverage=0.0,
            min_leaf_coverage=0.0,
            max_leaf_coverage=0.0,
            median_leaf_coverage=0.0,
        )

    leaf_coverages = np.array([item.leaf_coverage for item in stats], dtype=np.float64)
    num_empty_masks = sum(1 for item in stats if item.is_empty_mask)

    image_dimensions = sorted({(item.image_height, item.image_width) for item in stats})
    mask_dimensions = sorted({(item.mask_height, item.mask_width) for item in stats})

    return DatasetReport(
        num_images=num_image_files,
        num_masks=num_mask_files,
        num_valid_pairs=len(stats),
        num_errors=len(validation_result.errors),
        num_warnings=len(validation_result.warnings),
        image_dimensions=[list(dim) for dim in image_dimensions],
        mask_dimensions=[list(dim) for dim in mask_dimensions],
        percentage_empty_masks=100.0 * num_empty_masks / len(stats),
        mean_leaf_coverage=float(leaf_coverages.mean()),
        min_leaf_coverage=float(leaf_coverages.min()),
        max_leaf_coverage=float(leaf_coverages.max()),
        median_leaf_coverage=float(np.median(leaf_coverages)),
    )


def save_dataset_report(
    report: DatasetReport,
    validation_result: ValidationResult,
    output_dir: str | Path,
    report_name: str = "dataset_report",
) -> dict[str, Path]:
    """Save the dataset report as JSON and Markdown, plus the full issue list as JSON.

    Args:
        report: The computed :class:`DatasetReport`.
        validation_result: The validation result the report was built from
            (used to also persist the full list of issues for debugging).
        output_dir: Directory to write the report files into.
        report_name: Base filename (without extension) for the report files.

    Returns:
        A dictionary mapping output kind ("json", "markdown", "issues") to
        the path of the corresponding file.
    """
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    json_path = output_directory / f"{report_name}.json"
    markdown_path = output_directory / f"{report_name}.md"
    issues_path = output_directory / f"{report_name}_issues.json"

    save_json(asdict(report), json_path)

    issues_payload: dict[str, Any] = {
        "errors": [asdict(issue) for issue in validation_result.errors],
        "warnings": [asdict(issue) for issue in validation_result.warnings],
    }
    save_json(issues_payload, issues_path)

    markdown_path.write_text(_render_markdown(report, validation_result), encoding="utf-8")

    logger.info("Dataset report saved to: %s", output_directory)
    return {"json": json_path, "markdown": markdown_path, "issues": issues_path}


def _render_markdown(report: DatasetReport, validation_result: ValidationResult) -> str:
    """Render the dataset report as a human-readable Markdown document."""
    lines = [
        "# Basil Segmentation Dataset Report",
        "",
        "## Summary",
        "",
        f"- Number of image files found: {report.num_images}",
        f"- Number of mask files found: {report.num_masks}",
        f"- Number of valid image/mask pairs: {report.num_valid_pairs}",
        f"- Errors found: {report.num_errors}",
        f"- Warnings found: {report.num_warnings}",
        "",
        "## Dimensions",
        "",
        f"- Distinct image dimensions (H x W): {report.image_dimensions}",
        f"- Distinct mask dimensions (H x W): {report.mask_dimensions}",
        "",
        "## Leaf Coverage Statistics",
        "",
        f"- Percentage of empty masks: {report.percentage_empty_masks:.2f}%",
        f"- Mean leaf coverage: {report.mean_leaf_coverage:.4f}",
        f"- Median leaf coverage: {report.median_leaf_coverage:.4f}",
        f"- Minimum leaf coverage: {report.min_leaf_coverage:.4f}",
        f"- Maximum leaf coverage: {report.max_leaf_coverage:.4f}",
        "",
    ]

    if validation_result.errors:
        lines.append("## Errors (must be fixed before training)")
        lines.append("")
        for issue in validation_result.errors:
            lines.append(f"- **[{issue.category}]** `{issue.filename}`: {issue.message}")
        lines.append("")

    if validation_result.warnings:
        lines.append("## Warnings")
        lines.append("")
        for issue in validation_result.warnings:
            lines.append(f"- **[{issue.category}]** `{issue.filename}`: {issue.message}")
        lines.append("")

    if not validation_result.errors and not validation_result.warnings:
        lines.append(
            "No pair-level issues were found. Complete metadata and a group-aware split before training."
        )
        lines.append("")

    return "\n".join(lines)
