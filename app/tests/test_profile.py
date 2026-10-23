from __future__ import annotations

from pathlib import Path

from airec.bench.profile import load_profile, refresh_profile, save_profile
from airec.version import __version__


def test_refresh_profile_persists(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("airec.bench.profile.hardware_profile", lambda: {"cpu": "test-cpu"})
    path = tmp_path / "profile.json"

    profile = refresh_profile(path)

    assert profile["hardware"] == {"cpu": "test-cpu"}
    assert profile["app_version"] == __version__
    assert load_profile(path) == profile


def test_missing_or_broken_profile(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    assert load_profile(path) is None

    path.write_text("{broken", encoding="utf-8")
    assert load_profile(path) is None

    save_profile(path, {"ok": True})
    assert load_profile(path) == {"ok": True}
