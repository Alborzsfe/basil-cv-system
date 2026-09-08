"""Group-aware train/validation/test splitting.

A naive random split at the image level can leak information: if several
images of the same plant cluster (or the same capture session) end up in
both the training and test sets, test performance will be overly
optimistic and will not reflect performance on genuinely unseen plants.

This module always splits by a group column (``plant_id``,
``capture_session_id``, etc.) using scikit-learn's ``GroupShuffleSplit``, so
that all images belonging to the same group stay together in a single
split.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

VALID_SPLIT_NAMES = ("train", "val", "test")


def group_aware_split(
    metadata: pd.DataFrame,
    group_column: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> pd.DataFrame:
    """Split a metadata table into train/val/test groups without leakage.

    Args:
        metadata: DataFrame with one row per image, containing at least
            ``group_column``.
        group_column: Column identifying the group that must not be split
            across train/val/test (e.g. ``plant_id`` or
            ``capture_session_id``).
        test_size: Fraction of *groups* (not images) to allocate to the
            test split.
        val_size: Fraction of *all groups* allocated to validation. It is
            rescaled internally after test groups are removed.
        seed: Random seed for reproducibility.

    Returns:
        A copy of ``metadata`` with an added ``split`` column containing
        one of ``"train"``, ``"val"``, ``"test"``.

    Raises:
        ValueError: If ``group_column`` is missing, or if there are too
            few distinct groups to perform the requested split.
    """
    if group_column not in metadata.columns:
        raise ValueError(
            f"Group column '{group_column}' not found in metadata columns: {list(metadata.columns)}"
        )

    if not 0 < float(test_size) < 1:
        raise ValueError("test_size must be between zero and one.")
    if not 0 < float(val_size) < 1:
        raise ValueError("val_size must be between zero and one.")
    if float(test_size) + float(val_size) >= 1:
        raise ValueError("test_size + val_size must be lower than one.")
    groups = metadata[group_column]
    normalized_groups = groups.astype(str).str.strip()
    if (
        groups.isna().any()
        or normalized_groups.eq("").any()
        or normalized_groups.str.upper().eq("UNKNOWN").any()
    ):
        raise ValueError(
            f"Group column '{group_column}' contains missing, empty, or placeholder values."
        )

    num_groups = metadata[group_column].nunique()
    if num_groups < 3:
        raise ValueError(
            f"Group-aware splitting requires at least 3 distinct groups in "
            f"'{group_column}' (found {num_groups}). Provide more plant_id / "
            f"capture_session_id diversity, or use a smaller test/val split."
        )
    test_group_count = math.ceil(float(test_size) * num_groups)
    remaining_group_count = num_groups - test_group_count
    relative_val_size = float(val_size) / (1.0 - float(test_size))
    val_group_count = math.ceil(relative_val_size * remaining_group_count)
    if remaining_group_count < 2 or remaining_group_count - val_group_count < 1:
        raise ValueError(
            "Split fractions leave no group for training; add groups or reduce test/val sizes."
        )

    metadata = metadata.reset_index(drop=True).copy()

    test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(test_splitter.split(metadata, groups=metadata[group_column]))

    train_val_df = metadata.iloc[train_val_idx]
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_idx_rel, val_idx_rel = next(
        val_splitter.split(train_val_df, groups=train_val_df[group_column])
    )

    train_idx = train_val_df.iloc[train_idx_rel].index
    val_idx = train_val_df.iloc[val_idx_rel].index

    metadata["split"] = "unassigned"
    metadata.loc[train_idx, "split"] = "train"
    metadata.loc[val_idx, "split"] = "val"
    metadata.loc[test_idx, "split"] = "test"

    validate_persisted_split(metadata, group_column)
    _log_split_summary(metadata, group_column)

    return metadata


def validate_persisted_split(metadata: pd.DataFrame, group_column: str) -> None:
    """Verify split labels, group completeness, and absence of leakage."""
    missing = sorted({group_column, "split"} - set(metadata.columns))
    if missing:
        raise ValueError(f"Persisted split is missing columns: {missing}.")
    if metadata.empty:
        raise ValueError("Persisted split contains no rows.")
    groups = metadata[group_column]
    normalized = groups.astype(str).str.strip()
    if groups.isna().any() or normalized.eq("").any() or normalized.str.upper().eq("UNKNOWN").any():
        raise ValueError(f"Persisted split has unresolved values in '{group_column}'.")
    observed = set(metadata["split"].astype(str))
    expected = set(VALID_SPLIT_NAMES)
    if observed != expected:
        raise ValueError(
            f"Persisted split labels must be {sorted(expected)}; got {sorted(observed)}."
        )
    group_to_splits = metadata.groupby(group_column)["split"].nunique()
    leaking_groups = group_to_splits[group_to_splits > 1]
    if not leaking_groups.empty:
        raise ValueError(
            f"Data leakage detected: the following groups appear in more than "
            f"one split: {list(leaking_groups.index)}"
        )


def _log_split_summary(metadata: pd.DataFrame, group_column: str) -> None:
    for split_name in VALID_SPLIT_NAMES:
        split_df = metadata[metadata["split"] == split_name]
        logger.info(
            "Split '%s': %d images, %d distinct groups ('%s').",
            split_name,
            len(split_df),
            split_df[group_column].nunique(),
            group_column,
        )


def save_split(metadata: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the split assignment to a CSV file for full reproducibility.

    Every experiment must persist exactly which images were used for
    train/val/test, so results can be audited and reproduced later.
    """
    output_file = Path(output_path)
    ensure_dir(output_file.parent)
    metadata.to_csv(output_file, index=False)
    logger.info("Split assignment saved to: %s", output_file)
    return output_file


def load_split(input_path: str | Path) -> pd.DataFrame:
    """Load a previously saved split assignment CSV."""
    return pd.read_csv(input_path)


def split_counts(metadata: pd.DataFrame) -> dict[str, int]:
    """Return the number of images per split, for quick reporting."""
    return {
        split_name: int((metadata["split"] == split_name).sum()) for split_name in VALID_SPLIT_NAMES
    }
