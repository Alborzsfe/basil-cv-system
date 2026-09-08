"""Consistent logging setup shared across scripts and modules.

All log messages produced by this project must be written in English, as
required by the project language policy.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create (or fetch) a logger with a consistent format.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        log_file: Optional path to also write logs to a file, in addition
            to standard output. Parent directories are created if needed.
        level: Logging level, defaults to ``logging.INFO``.

    Returns:
        A configured :class:`logging.Logger` instance. Calling this
        function multiple times with the same ``name`` will not duplicate
        handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger
