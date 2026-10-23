from __future__ import annotations

import sys
from pathlib import Path

from airec.core.paths import AppPaths, data_root, resource_root


def test_data_root_override(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "custom-data"
    monkeypatch.setenv("AIREC_DATA_DIR", str(target))
    assert data_root() == target.resolve()


def test_app_paths_layout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AIREC_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("AIREC_PORTABLE", raising=False)
    paths = AppPaths.resolve()
    paths.ensure()

    assert paths.data == tmp_path
    assert paths.models == tmp_path / "models"
    assert paths.logs.is_dir()
    assert paths.results.is_dir()
    assert paths.temp.is_dir()
    assert paths.config_file == tmp_path / "config.json"
    assert paths.session_file == tmp_path / "session.json"


def test_frozen_ui_resolves_from_meipass(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AIREC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    (tmp_path / "ui" / "dist").mkdir(parents=True)

    assert resource_root() == tmp_path
    assert AppPaths.resolve().ui == tmp_path / "ui" / "dist"
