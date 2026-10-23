"""User settings stored as JSON next to the application data."""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

log = logging.getLogger(__name__)

Theme = Literal["system", "light", "dark"]
Language = Literal["ru", "en"]


class Settings(BaseModel):
    """Persisted user settings. Unknown keys are ignored for forward compatibility."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1

    theme: Theme = "system"
    language: Language = "ru"
    autostart: bool = False
    notifications: bool = True
    close_to_tray: bool = True
    pause_on_battery: bool = False

    results_dir: str = ""
    keep_extracted_audio: bool = False

    asr_model: str = "gigaam-v3-e2e-ctc"
    llm_model: str = "auto"
    cpu_threads: int = Field(default=0, ge=0, le=256)
    max_parallel_jobs: int = Field(default=1, ge=1, le=8)
    external_llm_enabled: bool = False


def load_settings(path: Path) -> Settings:
    """Read settings, falling back to defaults and preserving a corrupt file."""
    if not path.exists():
        return Settings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        return Settings.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        backup = path.with_name(f"config.broken-{datetime.now(UTC):%Y%m%d-%H%M%S}.json")
        try:
            shutil.copy2(path, backup)
        except OSError:
            log.exception("cannot back up broken config")
        log.warning("settings are broken (%s); using defaults, backup: %s", exc, backup)
        return Settings()


def save_settings(path: Path, settings: Settings) -> None:
    """Atomically write settings so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    payload = json.dumps(settings.model_dump(), ensure_ascii=False, indent=2)
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
