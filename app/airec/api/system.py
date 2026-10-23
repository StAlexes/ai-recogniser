"""System endpoints: health, version, hardware profile."""

from __future__ import annotations

import platform
import sys
import time

from fastapi import APIRouter, Request

from airec import __version__
from airec.bench.hardware import hardware_profile
from airec.core.paths import AppPaths

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    paths: AppPaths = request.app.state.paths
    started = float(request.app.state.started_at)
    return {
        "status": "ok",
        "version": __version__,
        "pid": request.app.state.pid,
        "port": request.app.state.port,
        "uptime_s": round(time.time() - started, 1),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "portable": paths.portable,
        "data_dir": str(paths.data),
        "ui_built": paths.ui.exists(),
    }


@router.get("/system/profile")
def profile() -> dict[str, object]:
    return hardware_profile()
