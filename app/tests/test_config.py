from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from airec.core.config import Settings, load_settings, save_settings


def test_settings_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    settings = Settings(theme="dark", close_to_tray=False, max_parallel_jobs=2)
    save_settings(path, settings)
    loaded = load_settings(path)

    assert loaded.theme == "dark"
    assert loaded.close_to_tray is False
    assert loaded.max_parallel_jobs == 2


def test_broken_settings_fall_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")

    loaded = load_settings(path)

    assert loaded.theme == "system"
    assert list(tmp_path.glob("config.broken-*.json"))


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"theme": "light", "future_option": 42}), encoding="utf-8")
    assert load_settings(path).theme == "light"


def test_vad_settings_defaults_and_bounds() -> None:
    settings = Settings()

    assert settings.vad_enabled is True
    assert settings.vad_max_segment_s == 30
    assert settings.vad_min_silence_ms == 300
    assert settings.vad_speech_pad_ms == 100

    with pytest.raises(ValidationError):
        Settings(vad_max_segment_s=1000)
    with pytest.raises(ValidationError):
        Settings(vad_min_silence_ms=0)


def test_diarization_model_default_and_legacy_path() -> None:
    assert Settings().diarize_model == "pyannote-segmentation-3.0"

    legacy = Settings.model_validate({"diarize_model_path": "D:/models/wespeaker"})

    assert legacy.diarize_model == "custom"
    assert legacy.diarize_model_path == "D:/models/wespeaker"

    explicit = Settings.model_validate(
        {"diarize_model": "wespeaker-voxceleb-campplus", "diarize_model_path": "D:/old"}
    )
    assert explicit.diarize_model == "wespeaker-voxceleb-campplus"


def test_settings_with_utf8_bom_are_read(tmp_path: Path) -> None:
    """Notepad and some editors prepend a BOM; that must not reset the settings."""
    path = tmp_path / "config.json"
    payload = json.dumps({"theme": "dark", "close_to_tray": False})
    path.write_text(payload, encoding="utf-8-sig")

    loaded = load_settings(path)

    assert loaded.theme == "dark"
    assert loaded.close_to_tray is False
    assert not list(tmp_path.glob("config.broken-*.json"))
