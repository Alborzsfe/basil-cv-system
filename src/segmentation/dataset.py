"""Leakage-aware PyTorch dataset for paired basil images and leaf masks."""

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

from src.preprocessing.background import normalize_binary_mask
from src.utils.io_utils import safe_read_image, safe_read_mask

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class BasilSegmentationDataset(Dataset):
    def __init__(
        self,
        records: str | Path | pd.DataFrame,
        *,
        images_dir: str | Path,
        masks_dir: str | Path,
        split: str | None = None,
        image_size: Sequence[int] = (512, 512),
        augment: bool = False,
        augmentation: Mapping[str, Any] | None = None,
    ) -> None:
        frame = pd.read_csv(records) if not isinstance(records, pd.DataFrame) else records.copy()
        missing = sorted({"image_filename", "mask_filename"} - set(frame.columns))
        if missing:
            raise ValueError(f"Segmentation metadata is missing columns: {missing}.")
        if split is not None:
            if "split" not in frame.columns:
                raise ValueError("A split was requested but metadata has no 'split' column.")
            frame = frame[frame["split"] == split].copy()
        if frame.empty:
            raise ValueError(f"No segmentation records found for split={split!r}.")
        if len(image_size) != 2 or min(int(value) for value in image_size) <= 0:
            raise ValueError("image_size must contain two positive integers.")
        self.records = frame.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.height, self.width = int(image_size[0]), int(image_size[1])
        self.augment = augment
        self.augmentation = dict(augmentation or {})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records.iloc[index]
        image_path = self.images_dir / str(row["image_filename"])
        mask_path = self.masks_dir / str(row["mask_filename"])
        image = safe_read_image(image_path)
        mask = normalize_binary_mask(safe_read_mask(mask_path))
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask dimensions differ for '{image_path.name}'.")
        if self.augment:
            image, mask = _augment_pair(image, mask, self.augmentation)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        normalized = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))),
            "mask": torch.from_numpy(np.ascontiguousarray(mask[None].astype(np.float32))),
            "filename": image_path.name,
        }


def _augment_pair(
    image: np.ndarray, mask: np.ndarray, options: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < float(options.get("horizontal_flip_prob", 0.5)):
        image, mask = np.fliplr(image), np.fliplr(mask)
    if random.random() < float(options.get("vertical_flip_prob", 0.0)):
        image, mask = np.flipud(image), np.flipud(mask)
    rotation = float(options.get("rotation_limit_degrees", 0))
    translation = float(options.get("translate_percent", 0))
    scale_limit = float(options.get("scale_limit", 0))
    if rotation or translation or scale_limit:
        height, width = mask.shape
        matrix = cv2.getRotationMatrix2D(
            (width / 2, height / 2),
            random.uniform(-rotation, rotation),
            1 + random.uniform(-scale_limit, scale_limit),
        )
        matrix[0, 2] += random.uniform(-translation, translation) * width
        matrix[1, 2] += random.uniform(-translation, translation) * height
        image = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
    brightness = float(options.get("brightness_limit", 0))
    contrast = float(options.get("contrast_limit", 0))
    if brightness or contrast:
        alpha = 1 + random.uniform(-contrast, contrast)
        beta = random.uniform(-brightness, brightness) * 255
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
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
    if random.random() < float(options.get("blur_prob", 0)):
        image = cv2.GaussianBlur(image, (3, 3), 0)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)
