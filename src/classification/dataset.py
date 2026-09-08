"""PyTorch dataset for raw, leaf-only, and mask-aware basil inputs."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.background import apply_background, normalize_binary_mask
from src.segmentation.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.utils.io_utils import safe_read_image, safe_read_mask

VALID_INPUT_MODES = {"raw_image", "leaf_only", "leaf_only_mask_aware"}


class BasilClassificationDataset(Dataset):
    def __init__(
        self,
        records: str | Path | pd.DataFrame,
        *,
        class_names: Sequence[str],
        root_dir: str | Path = ".",
        split: str | None = None,
        image_size: Sequence[int] = (384, 384),
        input_mode: str = "leaf_only",
        background_mode: str = "black",
        augment: bool = False,
        augmentation: Mapping[str, Any] | None = None,
    ) -> None:
        frame = pd.read_csv(records) if not isinstance(records, pd.DataFrame) else records.copy()
        missing = sorted({"image_path", "label"} - set(frame.columns))
        if missing:
            raise ValueError(f"Classification metadata is missing columns: {missing}.")
        if split is not None:
            if "split" not in frame.columns:
                raise ValueError("A split was requested but metadata has no 'split' column.")
            frame = frame[frame["split"] == split].copy()
        if frame.empty:
            raise ValueError(f"No classification records found for split={split!r}.")
        if input_mode not in VALID_INPUT_MODES:
            raise ValueError(f"Unsupported input_mode '{input_mode}'.")
        names = list(class_names)
        if len(names) < 2 or len(names) != len(set(names)):
            raise ValueError("class_names must contain at least two unique labels.")
        unknown = sorted(set(frame["label"].astype(str)) - set(names))
        if unknown:
            raise ValueError(f"Metadata contains labels absent from class_names: {unknown}.")
        if len(image_size) != 2 or min(int(value) for value in image_size) <= 0:
            raise ValueError("image_size must contain two positive integers.")
        self.records = frame.reset_index(drop=True)
        self.class_names = names
        self.class_to_index = {name: index for index, name in enumerate(names)}
        self.root_dir = Path(root_dir)
        self.height, self.width = int(image_size[0]), int(image_size[1])
        self.input_mode = input_mode
        self.background_mode = background_mode
        self.augment = augment
        self.augmentation = dict(augmentation or {})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records.iloc[index]
        image_path = self.root_dir / str(row["image_path"])
        image = safe_read_image(image_path)
        mask: np.ndarray | None = None
        if self.input_mode != "raw_image":
            leaf_value = row.get("leaf_image_path")
            if isinstance(leaf_value, str) and leaf_value.strip():
                image = safe_read_image(self.root_dir / leaf_value)
            else:
                mask_value = row.get("mask_path")
                if not isinstance(mask_value, str) or not mask_value.strip():
                    raise ValueError(
                        f"Record '{image_path.name}' needs mask_path or leaf_image_path."
                    )
                mask = normalize_binary_mask(safe_read_mask(self.root_dir / mask_value))
                if mask.shape != image.shape[:2]:
                    raise ValueError(f"Image/mask dimensions differ for '{image_path.name}'.")
                image = apply_background(image, mask, mode=self.background_mode)
        if self.input_mode == "leaf_only_mask_aware" and mask is None:
            mask_value = row.get("mask_path")
            if not isinstance(mask_value, str) or not mask_value.strip():
                raise ValueError("Mask-aware input requires mask_path even with leaf_image_path.")
            mask = normalize_binary_mask(safe_read_mask(self.root_dir / mask_value))
            if mask.shape != image.shape[:2]:
                raise ValueError(f"Leaf image/mask dimensions differ for '{image_path.name}'.")
        if self.augment:
            image, mask = _augment(image, mask, self.augmentation)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        normalized = (image.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        channels = normalized.transpose(2, 0, 1)
        if self.input_mode == "leaf_only_mask_aware":
            assert mask is not None
            resized = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            channels = np.concatenate((channels, resized[None].astype(np.float32)), axis=0)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(channels)).float(),
            "target": torch.tensor(self.class_to_index[str(row["label"])], dtype=torch.long),
            "filename": image_path.name,
        }


def _augment(
    image: np.ndarray, mask: np.ndarray | None, options: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray | None]:
    if random.random() < float(options.get("horizontal_flip_prob", 0.5)):
        image, mask = np.fliplr(image), None if mask is None else np.fliplr(mask)
    if random.random() < float(options.get("vertical_flip_prob", 0)):
        image, mask = np.flipud(image), None if mask is None else np.flipud(mask)
    rotation = float(options.get("rotation_limit_degrees", 0))
    if rotation:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D(
            (width / 2, height / 2), random.uniform(-rotation, rotation), 1
        )
        image = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
        if mask is not None:
            mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
    brightness = float(options.get("brightness_limit", 0))
    contrast = float(options.get("contrast_limit", 0))
    image = np.clip(
        image.astype(np.float32) * (1 + random.uniform(-contrast, contrast))
        + random.uniform(-brightness, brightness) * 255,
        0,
        255,
    ).astype(np.uint8)
    saturation = float(options.get("saturation_limit", 0))
    hue = float(options.get("hue_shift_limit", 0))
    if saturation or hue:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = np.mod(hsv[..., 0] + random.uniform(-hue, hue), 180)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + random.uniform(-saturation, saturation)), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if random.random() < float(options.get("gaussian_noise_prob", 0)):
        image = np.clip(
            image.astype(np.float32) + np.random.normal(0, 5, image.shape), 0, 255
        ).astype(np.uint8)
    return np.ascontiguousarray(image), None if mask is None else np.ascontiguousarray(mask)
