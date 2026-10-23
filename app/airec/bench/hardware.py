"""Hardware probing for the first-run benchmark and diagnostics (M0 subset)."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from functools import lru_cache
from typing import Any

import psutil

log = logging.getLogger(__name__)

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _gpu_info() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=_CREATE_NO_WINDOW,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return []
        data = json.loads(completed.stdout)
        if isinstance(data, dict):
            data = [data]
        return [
            {
                "name": item.get("Name", ""),
                "vram_gb": round(item.get("AdapterRAM", 0) / 1024**3, 1)
                if item.get("AdapterRAM")
                else None,
                "driver": item.get("DriverVersion", ""),
            }
            for item in data
        ]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        log.debug("GPU probe failed: %s", exc)
        return []


@lru_cache(maxsize=1)
def hardware_profile() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disks = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        disks.append(
            {
                "mount": part.mountpoint,
                "fs": part.fstype,
                "total_gb": round(usage.total / 1024**3, 1),
                "free_gb": round(usage.free / 1024**3, 1),
            }
        )
    return {
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(memory.total / 1024**3, 1),
        "ram_available_gb": round(memory.available / 1024**3, 1),
        "disks": disks,
        "gpu": _gpu_info(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "architecture": platform.machine(),
    }
