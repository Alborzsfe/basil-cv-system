"""Create a privacy-conscious manifest for reproducible experiment runs."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def build_run_manifest(*, seed: int, device: str) -> dict[str, Any]:
    packages = {}
    for distribution in (
        "numpy",
        "opencv-python-headless",
        "pandas",
        "scikit-learn",
        "torch",
        "torchvision",
    ):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = None
    details: dict[str, Any] = {"requested_or_resolved": str(device)}
    try:
        import torch

        details["cuda_available"] = bool(torch.cuda.is_available())
        details["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            details["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        details.update(cuda_available=False, cuda_version=None)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("GITHUB_SHA"),
        "seed": int(seed),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "device": details,
    }
