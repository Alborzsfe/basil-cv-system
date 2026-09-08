#!/usr/bin/env python
"""Perform a group-aware train/val/test split and save it to disk.

This script must be run only after ``prepare_dataset.py`` (metadata
template exists) and after the placeholder grouping values have been
replaced with real ``plant_id`` / ``capture_session_id`` values. Splitting
with all rows still set to the placeholder "UNKNOWN" group would put the
entire dataset in a single group, which this script deliberately refuses to
do (see ``group_aware_split`` in ``src/data/splitting.py``).

Usage:
    python scripts/split_dataset.py --config configs/segmentation.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import UNKNOWN_GROUP_VALUE
from src.data.splitting import group_aware_split, save_split, split_counts
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group-aware dataset splitting.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/segmentation.yaml",
        help="Path to the segmentation configuration file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    metadata_csv_path = Path(config.dataset.metadata_csv)
    if not metadata_csv_path.exists():
        logger.error(
            "Metadata file not found: %s. Run scripts/prepare_dataset.py first.",
            metadata_csv_path,
        )
        return 1

    metadata = pd.read_csv(metadata_csv_path)
    group_column = config.split.group_column

    if group_column not in metadata.columns:
        logger.error("Group column '%s' not present in metadata.", group_column)
        return 1

    group_values = metadata[group_column]
    unresolved = group_values.isna() | group_values.astype(str).str.strip().str.upper().eq(
        UNKNOWN_GROUP_VALUE
    )
    if unresolved.any():
        logger.error(
            "%d row(s) in column '%s' have a missing or placeholder value '%s'. "
            "Fill every grouping identifier before splitting.",
            int(unresolved.sum()),
            group_column,
            UNKNOWN_GROUP_VALUE,
        )
        return 1

    split_metadata = group_aware_split(
        metadata=metadata,
        group_column=group_column,
        test_size=config.split.test_size,
        val_size=config.split.val_size,
        seed=config.experiment.seed,
    )

    output_path = save_split(split_metadata, config.split.split_file)
    counts = split_counts(split_metadata)

    logger.info("Split saved to: %s", output_path)
    logger.info("Split image counts: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
