#!/usr/bin/env python
"""Scaffold the basil dataset directory structure and metadata template.

This script does not require any model or GPU. It:

1. Ensures the expected directory structure exists under the configured
   dataset paths.
2. Scans the images/ directory and generates (or updates) a metadata CSV
   template with the grouping columns required for leakage-free splitting
   (plant_id, duct_id, hole_id, capture_session_id, timestamp), filled
   with a placeholder value where unknown.

It intentionally does NOT invent plant/duct/hole identifiers: the
placeholder value must be replaced by real metadata (e.g. exported from
the annotation tool or filled in manually) before splitting is performed.

Usage:
    python scripts/prepare_dataset.py --config configs/segmentation.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import ALL_METADATA_COLUMNS, GROUP_METADATA_COLUMNS, UNKNOWN_GROUP_VALUE
from src.utils.config import load_config
from src.utils.io_utils import ensure_dir, find_image_files
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _resolve_mask_filename(image_stem: str, masks_dir: Path) -> str:
    """Find the actual mask filename matching an image's stem.

    Masks may use a different file extension than their corresponding
    image (e.g. ``plant_001.jpg`` -> ``plant_001.png``), so the mask
    directory is searched by stem rather than assuming a shared extension.
    If no matching mask file exists yet, the image's own extension is used
    as a placeholder suggestion and a warning is logged.
    """
    candidates = [p for p in find_image_files(masks_dir) if p.stem.lower() == image_stem.lower()]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple masks match image stem '{image_stem}': {[item.name for item in candidates]}."
        )
    if candidates:
        return candidates[0].name
    return f"{image_stem}.MISSING"


def update_metadata(
    image_files: list[Path], masks_dir: Path, existing_metadata: pd.DataFrame
) -> tuple[pd.DataFrame, int, int]:
    """Append new images and refresh unresolved masks without losing user metadata."""
    if "image_filename" not in existing_metadata.columns and not existing_metadata.empty:
        raise ValueError("Existing metadata has no image_filename column.")
    metadata = existing_metadata.copy()
    for column in ALL_METADATA_COLUMNS:
        if column not in metadata.columns:
            metadata[column] = "" if column == "mask_filename" else UNKNOWN_GROUP_VALUE
    if metadata["image_filename"].duplicated().any():
        duplicates = sorted(
            metadata.loc[metadata["image_filename"].duplicated(), "image_filename"].astype(str)
        )
        raise ValueError(f"Existing metadata contains duplicate images: {duplicates}.")
    known_rows = {str(value): index for index, value in metadata["image_filename"].items()}
    new_rows: list[dict[str, str]] = []
    refreshed = 0
    for image_path in image_files:
        mask_filename = _resolve_mask_filename(image_path.stem, masks_dir)
        if image_path.name in known_rows:
            index = known_rows[image_path.name]
            value = metadata.at[index, "mask_filename"]
            current = "" if pd.isna(value) else str(value)
            if (not current.strip() or current.endswith(".MISSING")) and not mask_filename.endswith(
                ".MISSING"
            ):
                metadata.at[index, "mask_filename"] = mask_filename
                refreshed += 1
            continue
        row = {"image_filename": image_path.name, "mask_filename": mask_filename}
        row.update({column: UNKNOWN_GROUP_VALUE for column in GROUP_METADATA_COLUMNS})
        new_rows.append(row)
    if new_rows:
        metadata = pd.concat([metadata, pd.DataFrame(new_rows)], ignore_index=True)
    return metadata, len(new_rows), refreshed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaffold the basil segmentation dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/segmentation.yaml",
        help="Path to the segmentation configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    images_dir = ensure_dir(config.dataset.images_dir)
    masks_dir = ensure_dir(config.dataset.masks_dir)
    metadata_csv_path = Path(config.dataset.metadata_csv)

    logger.info("Images directory ready at: %s", images_dir)
    logger.info("Masks directory ready at: %s", masks_dir)

    image_files = find_image_files(images_dir)
    if not image_files:
        logger.warning(
            "No images found in '%s' yet. Place raw basil images and masks there, "
            "then re-run this script to generate the metadata template.",
            images_dir,
        )
        return

    if metadata_csv_path.exists():
        existing_metadata = pd.read_csv(metadata_csv_path)
        logger.info(
            "Existing metadata found with %d rows; new images and masks will be merged.",
            len(existing_metadata),
        )
    else:
        existing_metadata = pd.DataFrame(columns=ALL_METADATA_COLUMNS)

    updated_metadata, added_count, refreshed_count = update_metadata(
        image_files, masks_dir, existing_metadata
    )
    for image_path in image_files:
        mask_filename = str(
            updated_metadata.loc[
                updated_metadata["image_filename"] == image_path.name, "mask_filename"
            ].iloc[0]
        )
        if mask_filename.endswith(".MISSING"):
            logger.warning(
                "No mask file found yet for image '%s' (expected a file named '%s.<ext>' in '%s').",
                image_path.name,
                image_path.stem,
                masks_dir,
            )
    ensure_dir(metadata_csv_path.parent)
    updated_metadata.to_csv(metadata_csv_path, index=False)

    logger.info(
        "Metadata saved to '%s' (%d total rows, %d added, %d masks refreshed). "
        "Grouping columns are filled with the placeholder value '%s' and must be "
        "updated with real plant/duct/hole/session identifiers before splitting.",
        metadata_csv_path,
        len(updated_metadata),
        added_count,
        refreshed_count,
        UNKNOWN_GROUP_VALUE,
    )


if __name__ == "__main__":
    main()
