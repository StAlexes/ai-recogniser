"""Model registry and downloader."""

from __future__ import annotations

from airec.models.manager import ModelManager
from airec.models.manifest import find_model, load_manifest

__all__ = ["ModelManager", "find_model", "load_manifest"]
