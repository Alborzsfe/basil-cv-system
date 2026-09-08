"""Reproducibility utilities.

Every experiment must record and apply a fixed random seed so results can
be reproduced later. This module seeds Python's ``random``, NumPy, and
(when available) PyTorch, including CUDA devices.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all relevant random number generators for reproducibility.

    Args:
        seed: The random seed to apply everywhere.
        deterministic: If True, also configure PyTorch/cuDNN for
            deterministic (but potentially slower) execution. This should
            be True for research runs whose results must be reproducible,
            and can be set to False for a small speed gain when strict
            reproducibility is not required.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # Imported lazily: torch may not be installed in every context.

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        # PyTorch is not available in this environment (e.g. lightweight
        # dataset-validation-only usage). Non-torch seeding is still applied.
        pass


def get_device_info() -> dict[str, Any]:
    """Return a small dictionary describing the available compute device.

    Intended to be stored alongside experiment metadata so every run
    records whether it executed on CPU or GPU, and which GPU.
    """
    info: dict[str, Any] = {"device": "cpu", "cuda_available": False, "gpu_name": None}
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info
