"""Logging setup: rotating file log plus console output."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(logs_dir: Path, level: str = "INFO", console: bool = True) -> Path:
    """Configure the root logger once and return the log file path."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "app.log"

    root = logging.getLogger()
    if getattr(root, "_airec_configured", False):
        return log_file

    root.setLevel(level.upper())
    formatter = logging.Formatter(FORMAT, DATEFMT)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if console and sys.stderr is not None:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        root.addHandler(stream)

    logging.captureWarnings(True)
    root._airec_configured = True  # type: ignore[attr-defined]
    return log_file
