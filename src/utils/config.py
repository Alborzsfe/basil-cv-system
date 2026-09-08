"""Central configuration system.

All training, evaluation, and inference parameters must be read from YAML
configuration files under ``configs/``. Nothing in ``src/`` or ``scripts/``
should hard-code paths, hyperparameters, or thresholds.

Usage:
    from src.utils.config import load_config

    config = load_config("configs/segmentation.yaml")
    print(config.dataset.image_size)
    print(config["dataset"]["image_size"])  # dict-style access also works
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml


class Config:
    """A thin wrapper around a nested dictionary that supports both.

    ``config.section.key`` and ``config["section"]["key"]`` access styles.
    This is a data holder only; it performs no validation of the schema
    beyond what individual modules choose to check.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data: dict[str, Any] = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(
                f"Configuration has no key '{name}'. Available keys: {sorted(self._data.keys())}"
            ) from exc
        return self._wrap(value)

    def __getitem__(self, key: str) -> Any:
        return self._wrap(self._data[key])

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for ``key`` or ``default`` if it is missing."""
        if key not in self._data:
            return default
        return self._wrap(self._data[key])

    def to_dict(self) -> dict[str, Any]:
        """Return a plain (deep-copied) dictionary representation."""
        return copy.deepcopy(self._data)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [Config(item) if isinstance(item, dict) else item for item in value]
        return value


def load_config(path: str | Path) -> Config:
    """Load a YAML configuration file into a :class:`Config` object.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        A :class:`Config` wrapping the parsed YAML content.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file does not contain a top-level mapping.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file_handle:
        raw_data = yaml.safe_load(file_handle)

    if raw_data is None:
        raw_data = {}
    if not isinstance(raw_data, dict):
        raise ValueError(
            f"Configuration file '{config_path}' must contain a top-level mapping, "
            f"got {type(raw_data).__name__} instead."
        )

    return Config(raw_data)


def save_config(config: Config | dict[str, Any], path: str | Path) -> None:
    """Persist a configuration object to disk as YAML.

    Used by experiment tracking so that every run stores the exact
    configuration that produced it.
    """
    data = config.to_dict() if isinstance(config, Config) else config
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(data, file_handle, sort_keys=False, allow_unicode=False)
