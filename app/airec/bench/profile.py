"""Persisted machine profile (hardware snapshot for recommendations)."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from airec.bench.hardware import hardware_profile
from airec.version import __version__

log = logging.getLogger(__name__)
SCHEMA_VERSION = 1


def build_profile() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "app_version": __version__,
        "hardware": hardware_profile(),
        # Filled by the M5 benchmark:
        "tier": None,
        "recommendation": None,
    }


def save_profile(path: Path, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def load_profile(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def refresh_profile(path: Path) -> dict[str, Any]:
    profile = build_profile()
    save_profile(path, profile)
    log.info("hardware profile refreshed: %s", path)
    return profile
