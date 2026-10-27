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
    external_llm_base_url: str = "http://127.0.0.1:11434"
    external_llm_api_key: str = ""
    llm_provider: str = "llamacpp-local"
    llm_ctx_size: int = Field(default=4096, ge=512, le=131072)
    llm_threads: int = Field(default=0, ge=0, le=256)

    # Long recordings are split into speech segments with Silero VAD.  The
    # segment cap is deliberately far below GigaAM's positional limit (~200 s).
    vad_enabled: bool = True
    vad_max_segment_s: int = Field(default=30, ge=5, le=120)
    vad_min_silence_ms: int = Field(default=300, ge=50, le=5000)
    vad_speech_pad_ms: int = Field(default=100, ge=0, le=1000)

    # Speaker diarization: cluster VAD segments by speaker identity using
    # Wespeaker embeddings (optional, disabled by default).
    diarize_enabled: bool = False
    diarize_threshold: float = Field(default=0.70, ge=0.1, le=0.99)
    diarize_model_path: str = ""

    # ragkb runs as a separate, local HTTP service.  It is optional: when it is
    # unavailable, transcription and the rest of the core keep working.
    rag_enabled: bool = True
    rag_url: str = "http://127.0.0.1:8000"
    rag_k: int = Field(default=5, ge=1, le=20)

    # Automatically start recognition when a file is added via drag-drop or
    # context menu.  When disabled, files are queued but not started until the
    # user clicks "Распознать".
    recognize_immediately: bool = True


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
