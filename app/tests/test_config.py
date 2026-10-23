from __future__ import annotations

import json
from pathlib import Path

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
