"""Windows shell integration: context menu, URI protocol and toast AUMID.

All writes go to ``HKCU`` (no administrator rights required).  Every helper is
a no-op outside Windows so the core stays importable on other platforms and in
tests.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from airec.core.paths import PROJECT_ROOT

log = logging.getLogger(__name__)

APP_USER_MODEL_ID = "AIRecogniser.Desktop"
PROTOCOL = "airec"
CONTEXT_MENU_KEY = "AIRecogniser"
CONTEXT_MENU_LABEL = "Распознать в AI Recogniser"

SUPPORTED_EXTENSIONS = (
    ".mp3",
    ".mp4",
    ".m4a",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
)


def _shell_executable() -> tuple[str, list[str]]:
    """Return (executable, leading arguments) that launch the desktop shell."""
    if getattr(sys, "frozen", False):
        return sys.executable, []
    python = Path(sys.executable)
    pythonw = python.with_name("pythonw.exe")
    runner = pythonw if pythonw.is_file() else python
    return str(runner), [str(PROJECT_ROOT / "shell" / "main.py")]


def _command(*, protocol: bool = False, file_arg: bool = False) -> str:
    executable, prefix = _shell_executable()
    command = subprocess.list2cmdline([executable, *prefix])
    if protocol:
        command += ' --protocol "%1"'
    elif file_arg:
        command += ' "%1"'
    return command


def _winreg():
    import winreg  # local import: only available on Windows

    return winreg


def _set_value(key, name: str, value: str) -> None:
    winreg = _winreg()
    winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)


def register_context_menu() -> bool:
    """Register the Explorer context-menu item for supported file types."""
    if os.name != "nt":
        return False
    winreg = _winreg()
    command = _command(file_arg=True)
    icon = f'"{_shell_executable()[0]}",0'
    try:
        for extension in SUPPORTED_EXTENSIONS:
            path = rf"Software\Classes\{extension}\shell\{CONTEXT_MENU_KEY}"
            with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_WRITE) as key:
                winreg.SetValueEx(key, "", 0, winreg.REG_SZ, CONTEXT_MENU_LABEL)
                winreg.SetValueEx(key, "Icon", 0, winreg.REG_SZ, icon)
            with winreg.CreateKeyEx(
                winreg.HKEY_CURRENT_USER, path + r"\command", 0, winreg.KEY_WRITE
            ) as key:
                winreg.SetValueEx(key, "", 0, winreg.REG_SZ, command)
    except OSError:
        log.exception("cannot register the Explorer context menu")
        return False
    log.info("context menu registered for %d extensions", len(SUPPORTED_EXTENSIONS))
    return True


def unregister_context_menu() -> bool:
    """Remove the Explorer context-menu item."""
    if os.name != "nt":
        return False
    # ``reg.exe delete`` handles a whole subtree and works even when Explorer
    # or another process keeps a handle on the key open.  Plain
    # ``winreg.DeleteKey`` fails with access denied in that situation.
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    ok = True
    for extension in SUPPORTED_EXTENSIONS:
        path = rf"HKCU\Software\Classes\{extension}\shell\{CONTEXT_MENU_KEY}"
        try:
            completed = subprocess.run(
                ["reg.exe", "delete", path, "/f"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                creationflags=creationflags,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            log.debug("cannot run reg.exe delete for %s", path, exc_info=True)
            ok = False
            continue
        if completed.returncode not in {0, 1}:  # 1 = key did not exist
            log.debug("reg.exe delete failed for %s (%s)", path, completed.returncode)
            ok = False
    log.info("context menu removed")
    return ok


def context_menu_registered() -> bool:
    """True when at least one supported extension has the menu entry."""
    if os.name != "nt":
        return False
    winreg = _winreg()
    for extension in SUPPORTED_EXTENSIONS:
        path = rf"Software\Classes\{extension}\shell\{CONTEXT_MENU_KEY}\command"
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path):
                return True
        except OSError:
            continue
    return False


def register_protocol() -> bool:
    """Register ``airec://`` so toast buttons can activate the running app."""
    if os.name != "nt":
        return False
    winreg = _winreg()
    command = _command(protocol=True)
    try:
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, rf"Software\Classes\{PROTOCOL}", 0, winreg.KEY_WRITE
        ) as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "URL:AI Recogniser Protocol")
            winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            rf"Software\Classes\{PROTOCOL}\DefaultIcon",
            0,
            winreg.KEY_WRITE,
        ) as key:
            winreg.SetValueEx(
                key, "", 0, winreg.REG_SZ, f'"{_shell_executable()[0]}",0'
            )
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            rf"Software\Classes\{PROTOCOL}\shell\open\command",
            0,
            winreg.KEY_WRITE,
        ) as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, command)
    except OSError:
        log.exception("cannot register the airec:// protocol")
        return False
    return True


def register_toast_app_id() -> bool:
    """Register the AppUserModelID used by native Windows toast notifications."""
    if os.name != "nt":
        return False
    winreg = _winreg()
    try:
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            rf"Software\Classes\AppUserModelId\{APP_USER_MODEL_ID}",
            0,
            winreg.KEY_WRITE,
        ) as key:
            _set_value(key, "DisplayName", "AI Recogniser")
            _set_value(key, "IconUri", _shell_executable()[0])
            _set_value(key, "IconBackgroundColor", "FF1F6FEB")
    except OSError:
        log.exception("cannot register the toast AppUserModelID")
        return False
    return True


def apply(settings) -> None:
    """Apply registry-side integration after settings are loaded or changed."""
    if os.name != "nt":
        return
    register_protocol()
    register_toast_app_id()
    if getattr(settings, "context_menu_enabled", True):
        register_context_menu()
    else:
        unregister_context_menu()
