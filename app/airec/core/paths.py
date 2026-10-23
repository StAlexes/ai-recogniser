"""Filesystem layout and portable-mode detection.

Rules:
- installed mode: data lives in %LOCALAPPDATA%\\AIRecogniser (never in Program Files);
- portable mode: data lives next to the executable in .\\data, nothing in the registry;
- tests/dev may override the data root with AIREC_DATA_DIR.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "AIRecogniser"
APP_TITLE = "AI Recogniser"

# .../AI Recogniser/app/airec/core/paths.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resource_root() -> Path:
    """Read-only bundled resources (UI, icons), also inside a PyInstaller build."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return PROJECT_ROOT


def app_root() -> Path:
    """Directory of the executable in a frozen build, repo root in development."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return PROJECT_ROOT


def is_portable() -> bool:
    flag = os.environ.get("AIREC_PORTABLE", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    return (app_root() / "portable.flag").exists()


def data_root() -> Path:
    override = os.environ.get("AIREC_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if is_portable():
        return app_root() / "data"
    return Path(user_data_dir(APP_NAME, appauthor=False, roaming=False))


@dataclass(frozen=True)
class AppPaths:
    root: Path
    data: Path
    models: Path
    cache: Path
    logs: Path
    results: Path
    temp: Path
    bin: Path
    ui: Path

    @classmethod
    def resolve(cls) -> AppPaths:
        root = app_root()
        data = data_root()
        # In a frozen build the UI ships inside _internal (sys._MEIPASS).
        ui = resource_root() / "ui" / "dist"
        if not ui.exists():
            # Running from source: the UI build lives at the repo root.
            ui = PROJECT_ROOT / "ui" / "dist"
        return cls(
            root=root,
            data=data,
            models=data / "models",
            cache=data / "cache",
            logs=data / "logs",
            results=data / "results",
            temp=data / "temp",
            bin=root / "bin",
            ui=ui,
        )

    @property
    def config_file(self) -> Path:
        return self.data / "config.json"

    @property
    def session_file(self) -> Path:
        return self.data / "session.json"

    @property
    def portable(self) -> bool:
        return is_portable()

    def ensure(self) -> None:
        for directory in (self.data, self.models, self.cache, self.logs, self.results, self.temp):
            directory.mkdir(parents=True, exist_ok=True)
