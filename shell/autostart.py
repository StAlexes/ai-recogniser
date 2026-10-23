"""Windows autostart integration (HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run).

Autostart is strictly opt-in: the shell never writes the registry unless the
user explicitly runs ``--autostart on``. Portable mode leaves the registry
untouched.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from airec.core.paths import is_portable

LOG = logging.getLogger(__name__)

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "AIRecogniser"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def autostart_command() -> str:
    """Command line stored in the Run key (quoted, ready for Win32)."""
    executable = Path(sys.executable).resolve()
    if getattr(sys, "frozen", False):
        return f'"{executable}"'
    script = repo_root() / "shell" / "main.py"
    return f'"{executable}" "{script}"'


def _query() -> str | None:
    """Return the raw registry value or None when the entry is absent."""
    if os.name != "nt":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _kind = winreg.QueryValueEx(key, VALUE_NAME)
    except FileNotFoundError:
        return None
    except OSError:
        LOG.exception("cannot read the autostart registry value")
        return None
    return str(value)


def is_enabled() -> bool:
    return _query() is not None


def enable() -> bool:
    if is_portable():
        LOG.warning("portable mode: autostart is not written to the registry")
        return True
    if os.name != "nt":
        LOG.warning("autostart is only supported on Windows")
        return False

    import winreg

    command = autostart_command()
    try:
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command)
    except OSError:
        LOG.exception("cannot enable autostart")
        return False
    LOG.info("autostart enabled: %s", command)
    return True


def disable() -> bool:
    if is_portable():
        LOG.warning("portable mode: autostart is not managed in the registry")
        return True
    if os.name != "nt":
        return False

    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.DeleteValue(key, VALUE_NAME)
    except FileNotFoundError:
        pass
    except OSError:
        LOG.exception("cannot disable autostart")
        return False
    LOG.info("autostart disabled")
    return True


def run_cli(action: str) -> int:
    """Handle ``--autostart {on,off,status}``; status never mutates the registry."""
    if action == "status":
        state = "enabled" if is_enabled() else "disabled"
        if is_portable():
            state += " (portable)"
        print(f"autostart: {state}")
        print(f"command: {_query() or autostart_command()}")
        return 0
    if action == "on":
        if not enable():
            print("autostart: unchanged", file=sys.stderr)
            return 1
        print("autostart: enabled")
        return 0
    if action == "off":
        if not disable():
            print("autostart: unchanged", file=sys.stderr)
            return 1
        print("autostart: disabled")
        return 0
    print(f"unknown autostart action: {action}", file=sys.stderr)
    return 2
