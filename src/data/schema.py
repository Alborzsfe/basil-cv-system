"""Metadata schema for the basil segmentation dataset.

Group-aware dataset splitting (see ``splitting.py``) requires knowing which
plant, duct, hole, and capture session each image belongs to. This module
defines that schema in one place so every part of the pipeline agrees on
column names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Required columns in the dataset metadata CSV/DataFrame. These identify an
# image/mask pair and the physical/temporal context it was captured in.
REQUIRED_METADATA_COLUMNS: list[str] = [
    "image_filename",
    "mask_filename",
]

# Grouping columns used to prevent data leakage between train/val/test
# splits. Not all of them need to be known for every dataset, but at least
# one should be populated to enable a meaningful group-aware split.
GROUP_METADATA_COLUMNS: list[str] = [
    "plant_id",
    "duct_id",
    "hole_id",
    "capture_session_id",
    "timestamp",
]

ALL_METADATA_COLUMNS: list[str] = REQUIRED_METADATA_COLUMNS + GROUP_METADATA_COLUMNS

# Value used to mark a grouping field as not yet known. This keeps the
# metadata file schema-complete even before annotators fill in the real
# plant/duct/hole identifiers.
UNKNOWN_GROUP_VALUE = "UNKNOWN"


@dataclass
class ImageRecord:
    """A single image/mask pair with its grouping metadata.

    Attributes:
        image_filename: Filename of the RGB image, relative to the
            ``images/`` directory.
        mask_filename: Filename of the corresponding binary mask, relative
            to the ``masks/`` directory.
        plant_id: Identifier of the individual plant cluster, if known.
        duct_id: Identifier of the growing duct, if known.
        hole_id: Identifier of the planting hole within the duct, if known.
        capture_session_id: Identifier grouping images captured together
            (e.g. same camera pass), if known.
        timestamp: ISO-8601 capture timestamp, if known.
    """

    image_filename: str
    mask_filename: str
    plant_id: str | None = None
    duct_id: str | None = None
    hole_id: str | None = None
    capture_session_id: str | None = None
    timestamp: str | None = None
    extra: dict = field(default_factory=dict)
