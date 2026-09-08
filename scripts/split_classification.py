#!/usr/bin/env python
"""Validate classification metadata and create a leakage-free split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import UNKNOWN_GROUP_VALUE
from src.data.splitting import group_aware_split, save_split, split_counts
from src.utils.config import load_config


def _path_value(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_metadata(frame: pd.DataFrame, config: object) -> None:
    required = {"image_path", "label", config.split.group_column}
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(
            f"Invalid classification metadata; missing={missing}, empty={frame.empty}."
        )
    if frame["image_path"].duplicated().any():
        raise ValueError("Duplicate image_path values found.")
    classes = list(config.classes)
    if (
        len(classes) < 2
        or len(classes) != len(set(classes))
        or any(
            not isinstance(name, str) or not name.strip() or name == "unknown" for name in classes
        )
    ):
        raise ValueError(
            "Configure at least two unique verified classes; unknown is inference-only."
        )
    observed = set(frame["label"].astype(str))
    if observed != set(classes):
        raise ValueError(
            f"Configured/observed label mismatch: configured={classes}, observed={sorted(observed)}."
        )
    groups = frame[config.split.group_column]
    normalized = groups.astype(str).str.strip()
    if (
        groups.isna().any()
        or normalized.eq("").any()
        or normalized.str.upper().eq(UNKNOWN_GROUP_VALUE).any()
    ):
        raise ValueError("Every row needs a real grouping identifier.")
    root = Path(config.dataset.get("root_dir", "."))
    mode = config.dataset.input_mode
    required_paths = ["image_path"]
    if mode == "leaf_only_mask_aware":
        if "mask_path" not in frame:
            raise ValueError("Mask-aware classification requires mask_path.")
        required_paths.append("mask_path")
    elif mode == "leaf_only":
        if not {"mask_path", "leaf_image_path"} & set(frame.columns):
            raise ValueError("Leaf-only classification requires mask_path or leaf_image_path.")
        for index, row in frame.iterrows():
            mask, leaf = row.get("mask_path"), row.get("leaf_image_path")
            selected = leaf if _path_value(leaf) else mask
            if not _path_value(selected) or not (root / str(selected)).is_file():
                raise FileNotFoundError(f"Missing leaf-only input for row {index}.")
    if mode in {"leaf_only", "leaf_only_mask_aware"} and "leaf_image_path" in frame:
        for value in frame["leaf_image_path"]:
            if _path_value(value) and not (root / str(value)).is_file():
                raise FileNotFoundError(f"Missing leaf image: {root / str(value)}.")
    for column in required_paths:
        missing_files = [
            str(root / str(value))
            for value in frame[column]
            if not _path_value(value) or not (root / str(value)).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(f"Missing files referenced by {column}: {missing_files[:10]}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and split classification metadata.")
    parser.add_argument("--config", default="configs/classification.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    source = Path(config.dataset.metadata_csv)
    if not source.is_file():
        raise FileNotFoundError(f"Classification metadata not found: {source}.")
    metadata = pd.read_csv(source)
    validate_metadata(metadata, config)
    split = group_aware_split(
        metadata,
        config.split.group_column,
        float(config.split.test_size),
        float(config.split.val_size),
        int(config.experiment.seed),
    )
    for name in ("train", "val", "test"):
        observed = set(split.loc[split["split"] == name, "label"].astype(str))
        missing = sorted(set(config.classes) - observed)
        if missing:
            raise ValueError(f"Split '{name}' has no samples for {missing}.")
    output = save_split(split, config.split.split_file)
    print(f"Saved {output} with counts {split_counts(split)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
