"""AI Recogniser desktop shell.

A thin WebView2 host around the local core: single instance, core lifecycle,
window, tray and autostart. No business logic lives here.

Usage:
    .\\.venv\\Scripts\\python.exe shell\\main.py
    .\\.venv\\Scripts\\python.exe shell\\main.py --smoke-test
    .\\.venv\\Scripts\\python.exe shell\\main.py --autostart status

Environment:
    AIREC_CORE_URL  use an already running core instead of spawning one
    AIREC_TOKEN     token for the external core (falls back to session.json)
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any
from urllib.parse import urlsplit

from airec.core.paths import APP_TITLE, AppPaths
from airec.version import __version__

try:  # `python -m shell.main`
    from shell import autostart
except ImportError:  # `python shell/main.py` puts shell/ itself on sys.path
    import autostart  # type: ignore[no-redef]

if TYPE_CHECKING:
    from airec.core.security import Session

LOG = logging.getLogger("airec.shell")

MUTEX_NAME = r"Local\AIRecogniser.Shell.v1"
ERROR_ALREADY_EXISTS = 183
CREATE_NO_WINDOW = 0x08000000
SW_RESTORE = 9
MB_ICONERROR = 0x10
MB_ICONINFORMATION = 0x40

CORE_SESSION_TIMEOUT_S = 40.0
CORE_HEALTH_TIMEOUT_S = 10.0
HEALTH_POLL_INTERVAL_S = 0.3
CORE_STOP_TIMEOUT_S = 5.0

LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5

_logging_configured = False
_mutex_handle: int | None = None


class CoreError(RuntimeError):
    """The core could not be started or did not become healthy."""


@dataclass
class CoreHandle:
    base_url: str
    token: str
    port: int | None
    version: str
    pid: int | None = None
    process: subprocess.Popen[bytes] | None = None
    log_stream: IO[str] | None = None
    spawned: bool = False


# --------------------------------------------------------------------------- #
# Logging, message boxes and single instance
# --------------------------------------------------------------------------- #


def setup_logging(paths: AppPaths, *, console: bool) -> Path:
    """Configure shell logging: 5 MB x 5 rotating file, plus console in dev."""
    global _logging_configured

    paths.logs.mkdir(parents=True, exist_ok=True)
    log_file = paths.logs / "shell.log"
    if _logging_configured:
        return log_file

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if console and not getattr(sys, "frozen", False):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    # Keep the core probe noise out of the console and the log.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _logging_configured = True
    return log_file


def message_box(text: str, *, error: bool = True) -> None:
    """Show a native MessageBox; fall back to stderr off Windows."""
    if os.name != "nt":
        print(text, file=sys.stderr)
        return
    flags = MB_ICONERROR if error else MB_ICONINFORMATION
    try:
        ctypes.windll.user32.MessageBoxW(None, text, APP_TITLE, flags)
    except Exception:
        LOG.exception("cannot show a message box")
        print(text, file=sys.stderr)


def acquire_single_instance() -> bool:
    """Create the named mutex. False means another shell instance is running."""
    global _mutex_handle
    if os.name != "nt":
        return True

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    kernel32.CreateMutexW.restype = ctypes.c_void_p

    handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    last_error = kernel32.GetLastError()
    if not handle:
        LOG.warning(
            "CreateMutexW failed (error %s); continuing without the guard", last_error
        )
        return True

    _mutex_handle = handle
    return last_error != ERROR_ALREADY_EXISTS


def activate_existing_window() -> bool:
    """Restore and focus the window of the first instance, if it can be found."""
    if os.name != "nt":
        return False
    try:
        user32 = ctypes.windll.user32
        user32.FindWindowW.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p)
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.GetClassNameW.argtypes = (
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_int,
        )

        hwnd = user32.FindWindowW(None, APP_TITLE)
        if not hwnd:
            return False

        # Never touch Explorer windows for a folder that happens to share the
        # application title; pywebview's WinForms window has its own class.
        class_name = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_name, 256)
        if not class_name.value.startswith("WindowsForms10"):
            LOG.info(
                "ignoring window %r of class %r while activating the first instance",
                APP_TITLE,
                class_name.value,
            )
            return False

        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetForegroundWindow(hwnd)
        return True
    except Exception:
        LOG.exception("cannot activate the existing window")
        return False


# --------------------------------------------------------------------------- #
# Core lifecycle
# --------------------------------------------------------------------------- #


def repo_root() -> Path:
    """Repository root in development, install directory in a frozen build."""
    if getattr(sys, "frozen", False):
        return AppPaths.resolve().root
    return Path(__file__).resolve().parents[1]


def discard_stale_session(session_file: Path) -> None:
    """Remove session.json left behind by a crashed core."""
    import psutil

    from airec.core.security import read_session

    session = read_session(session_file)
    if session is None:
        return
    if not psutil.pid_exists(session.pid):
        with contextlib.suppress(OSError):
            session_file.unlink()
        LOG.info("removed a stale session.json (pid %s is gone)", session.pid)


def wait_health(base_url: str, *, timeout: float) -> bool:
    """Poll GET /healthz until the core answers or the timeout expires."""
    import httpx

    deadline = time.monotonic() + timeout
    last_error = "no response"
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(f"{base_url}/healthz")
                if response.status_code == 200 and response.json().get("status") == "ok":
                    return True
                last_error = f"HTTP {response.status_code}"
            except (httpx.HTTPError, ValueError) as exc:
                last_error = str(exc) or exc.__class__.__name__
            time.sleep(HEALTH_POLL_INTERVAL_S)
    LOG.warning("core health check failed: %s", last_error)
    return False


def core_health(base_url: str, token: str) -> dict[str, Any]:
    """Read /api/v1/health (version, pid, ui_built); {} when unavailable."""
    import httpx

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(timeout=5.0, trust_env=False) as client:
            response = client.get(f"{base_url}/api/v1/health", headers=headers)
            if response.status_code == 200:
                return dict(response.json())
    except (httpx.HTTPError, ValueError) as exc:
        LOG.warning("cannot read core health details: %s", exc)
    return {}


def _session_token(base_url: str, session_file: Path) -> str:
    """Reuse the token written by a local core when only AIREC_CORE_URL is set."""
    host = (urlsplit(base_url).hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return ""
    from airec.core.security import read_session

    session = read_session(session_file)
    return session.token if session else ""


def _pid_is_descendant(pid: int, ancestor: int) -> bool:
    """True when `pid` is `ancestor` or one of its children.

    The venv's python.exe may be a launcher that spawns the real interpreter as
    a child, so Popen.pid and the pid written by the core can differ.
    """
    import psutil

    if pid == ancestor:
        return True
    try:
        return ancestor in {parent.pid for parent in psutil.Process(pid).parents()}
    except psutil.Error:
        return False


def _terminate_pid(pid: int) -> None:
    import psutil

    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    LOG.info("terminating core (pid %s)", pid)
    with contextlib.suppress(psutil.Error):
        process.terminate()
    try:
        process.wait(timeout=CORE_STOP_TIMEOUT_S)
        return
    except psutil.TimeoutExpired:
        LOG.warning("core did not exit after terminate; killing it")
    with contextlib.suppress(psutil.Error):
        process.kill()
    with contextlib.suppress(psutil.Error):
        process.wait(timeout=CORE_STOP_TIMEOUT_S)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Stop our direct child (the venv launcher shim), if it is still alive."""
    if process.poll() is not None:
        return
    with contextlib.suppress(OSError):
        process.terminate()
    try:
        process.wait(timeout=CORE_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=CORE_STOP_TIMEOUT_S)


def _terminate_tree(process: subprocess.Popen[bytes]) -> None:
    """Best-effort cleanup when the core pid was never obtained."""
    import psutil

    with contextlib.suppress(psutil.Error):
        for child in psutil.Process(process.pid).children(recursive=True):
            _terminate_pid(child.pid)
    _terminate_process(process)


def stop_core(handle: CoreHandle | None, paths: AppPaths) -> None:
    """Terminate the core we started and drop its session file (pid-guarded)."""
    if handle is None:
        return
    if handle.spawned:
        if handle.pid:
            _terminate_pid(handle.pid)
        if handle.process is not None:
            _terminate_process(handle.process)
    if handle.log_stream is not None:
        with contextlib.suppress(OSError):
            handle.log_stream.close()
        handle.log_stream = None
    if handle.pid:
        from airec.core.security import remove_session

        remove_session(paths.session_file, pid=handle.pid)
        LOG.info("core stopped (pid %s)", handle.pid)


def _wait_for_session(
    process: subprocess.Popen[bytes],
    session_file: Path,
    log_path: Path,
    spawn_time: float,
) -> Session:
    import psutil

    from airec.core.security import read_session

    deadline = time.monotonic() + CORE_SESSION_TIMEOUT_S
    while time.monotonic() < deadline:
        session = read_session(session_file)
        if session is not None and psutil.pid_exists(session.pid):
            if _pid_is_descendant(session.pid, process.pid):
                return session
            # The venv launcher may have been reparented already; a session
            # file that appeared after our spawn still belongs to this core.
            with contextlib.suppress(OSError):
                if session_file.stat().st_mtime >= spawn_time - 1.0:
                    return session
        code = process.poll()
        if code is not None and code != 0:
            raise CoreError(
                f"core exited with code {code} before writing session.json "
                f"(see {log_path})"
            )
        time.sleep(0.2)
    raise CoreError(
        f"core did not write session.json within {CORE_SESSION_TIMEOUT_S:.0f} s "
        f"(see {log_path})"
    )


def _spawn_core(paths: AppPaths) -> CoreHandle:
    discard_stale_session(paths.session_file)

    root = repo_root()
    env = os.environ.copy()
    app_dir = str(root / "app")
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = app_dir + (os.pathsep + python_path if python_path else "")

    paths.logs.mkdir(parents=True, exist_ok=True)
    log_path = paths.logs / "core.out.log"
    log_stream = log_path.open("a", encoding="utf-8", buffering=1)

    command = [sys.executable, "-m", "airec"]
    LOG.info("starting core: %s (cwd=%s, log=%s)", subprocess.list2cmdline(command), root, log_path)
    spawn_time = time.time()
    try:
        process = subprocess.Popen(
            command,
            cwd=str(root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except OSError as exc:
        log_stream.close()
        raise CoreError(f"cannot start the core: {exc}") from exc

    core_pid: int | None = None
    try:
        session = _wait_for_session(
            process, paths.session_file, log_path, spawn_time
        )
        core_pid = session.pid
        base_url = f"http://127.0.0.1:{session.port}"
        if not wait_health(base_url, timeout=CORE_HEALTH_TIMEOUT_S):
            raise CoreError(
                f"core did not pass /healthz within {CORE_HEALTH_TIMEOUT_S:.0f} s "
                f"(see {log_path})"
            )
        return CoreHandle(
            base_url=base_url,
            token=session.token,
            port=session.port,
            version=session.version,
            pid=session.pid,
            process=process,
            log_stream=log_stream,
            spawned=True,
        )
    except BaseException:
        if core_pid is not None:
            _terminate_pid(core_pid)
        _terminate_tree(process)
        log_stream.close()
        raise


def start_core(paths: AppPaths) -> CoreHandle:
    """Attach to AIREC_CORE_URL or spawn `python -m airec` and wait for health."""
    external_url = os.environ.get("AIREC_CORE_URL", "").strip()
    if external_url:
        base_url = external_url.rstrip("/")
        token = os.environ.get("AIREC_TOKEN", "").strip() or _session_token(
            base_url, paths.session_file
        )
        LOG.info("using the external core at %s", base_url)
        if not wait_health(base_url, timeout=CORE_HEALTH_TIMEOUT_S):
            raise CoreError(f"the core at {base_url} is not responding")
        info = core_health(base_url, token)
        return CoreHandle(
            base_url=base_url,
            token=token,
            port=urlsplit(base_url).port,
            version=str(info.get("version", "")),
        )
    return _spawn_core(paths)


# --------------------------------------------------------------------------- #
# Tray and window state
# --------------------------------------------------------------------------- #


def _asset(name: str) -> Path | None:
    candidates = [Path(__file__).resolve().parent / "assets" / name]
    if getattr(sys, "frozen", False):
        candidates.insert(0, Path(sys.executable).resolve().parent / "assets" / name)
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_tray_image() -> Any:
    from PIL import Image, ImageDraw

    path = _asset("icon.png")
    if path is not None:
        try:
            with Image.open(path) as source:
                return source.convert("RGBA")
        except OSError:
            LOG.warning("cannot read the tray icon: %s", path)

    # Minimal fallback so the tray keeps working if assets are missing.
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2, 2, 61, 61), radius=15, fill=(15, 108, 189, 255))
    for x, height in ((18, 14), (26, 26), (34, 36), (42, 26), (50, 14)):
        draw.rounded_rectangle(
            (x, 32 - height / 2, x + 5, 32 + height / 2), radius=2, fill="white"
        )
    return image


def _noop(*_args: Any) -> None:
    return None


class TrayIcon:
    """pystray icon running in a daemon thread."""

    def __init__(self, window: Any, app: ShellApp) -> None:
        self._window = window
        self._app = app
        self._icon: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import pystray
        except ImportError:
            LOG.warning("pystray is not installed; running without a tray icon")
            return

        menu = pystray.Menu(
            pystray.MenuItem("Открыть", self._on_open, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Пауза очереди", _noop, enabled=False),
            pystray.MenuItem("Последний результат", _noop, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Настройки", self._on_settings),
            pystray.MenuItem("Выход", self._on_quit),
        )
        self._icon = pystray.Icon(
            "airec", _load_tray_image(), f"{APP_TITLE} — очередь: 0", menu
        )
        self._thread = threading.Thread(target=self._run, name="tray", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._icon.run()
        except Exception:
            LOG.exception("tray icon failed")

    def _on_open(self, _icon: Any = None, _item: Any = None) -> None:
        self.show_window()

    def _on_settings(self, _icon: Any = None, _item: Any = None) -> None:
        self.show_window()
        try:
            self._window.evaluate_js("window.location.hash='#/settings'")
        except Exception:
            LOG.debug("cannot switch the window to the settings route", exc_info=True)

    def _on_quit(self, _icon: Any = None, _item: Any = None) -> None:
        # Leave the pystray callback before stopping the icon to avoid deadlocks.
        threading.Thread(target=self._app.request_quit, name="quit", daemon=True).start()

    def show_window(self) -> None:
        try:
            self._window.show()
        except Exception:
            LOG.debug("cannot show the window", exc_info=True)

    def notify(self, message: str, title: str = APP_TITLE) -> None:
        if self._icon is None:
            return
        try:
            self._icon.notify(message, title)
        except Exception:
            LOG.debug("tray notification failed", exc_info=True)

    def stop(self) -> None:
        icon, self._icon = self._icon, None
        if icon is None:
            return
        try:
            icon.stop()
        except Exception:
            LOG.debug("cannot stop the tray icon", exc_info=True)


class ShellApp:
    """Window/tray glue shared by the close handler and the tray menu."""

    def __init__(
        self,
        window: Any,
        core: CoreHandle,
        paths: AppPaths,
        *,
        close_to_tray: bool,
    ) -> None:
        self.window = window
        self.core = core
        self.paths = paths
        self.close_to_tray = close_to_tray
        self.tray = TrayIcon(window, self)
        self._lock = threading.Lock()
        self._quitting = False
        self._cleaned_up = False
        self._tray_hint_shown = False

    @property
    def quitting(self) -> bool:
        return self._quitting

    def on_closing(self) -> bool:
        """pywebview closing hook: False cancels the close, True allows it."""
        if self._quitting:
            return True
        if not self.close_to_tray:
            self._quitting = True
            return True

        with contextlib.suppress(Exception):
            self.window.hide()
        if not self._tray_hint_shown:
            self._tray_hint_shown = True
            self.tray.notify("Приложение продолжает работать в трее")
        return False

    def request_quit(self) -> None:
        """Stop the tray and close the window; the main thread finishes cleanup."""
        with self._lock:
            if self._quitting:
                return
            self._quitting = True
        LOG.info("quit requested")
        self.tray.stop()
        with contextlib.suppress(Exception):
            self.window.destroy()

    def cleanup(self) -> None:
        with self._lock:
            if self._cleaned_up:
                return
            self._cleaned_up = True
        self.tray.stop()
        stop_core(self.core, self.paths)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


def run_gui(paths: AppPaths) -> int:
    # Imported lazily so --help / --smoke-test / --autostart stay fast and work
    # even when WebView2 or pythonnet is unavailable.
    import webview

    from airec.core.config import load_settings

    settings = load_settings(paths.config_file)
    core = start_core(paths)
    LOG.info("core ready: version=%s port=%s", core.version, core.port)

    app: ShellApp | None = None
    try:
        fragment = f"#token={core.token}" if core.token else ""
        window = webview.create_window(
            APP_TITLE,
            f"{core.base_url}/{fragment}",
            width=1200,
            height=800,
            min_size=(960, 640),
            background_color="#F3F3F3",
        )
        app = ShellApp(window, core, paths, close_to_tray=settings.close_to_tray)
        window.events.closing += app.on_closing
        app.tray.start()

        icon = _asset("icon.ico")
        webview.start(debug=False, icon=str(icon) if icon else None)
        LOG.info("window loop finished")
        return 0
    finally:
        if app is not None:
            app.cleanup()
        else:
            stop_core(core, paths)


def run_smoke_test(paths: AppPaths) -> int:
    """Headless end-to-end check: core boot, session.json, /healthz, cleanup."""
    core: CoreHandle | None = None
    try:
        core = start_core(paths)
        info = core_health(core.base_url, core.token)
        payload = {
            "ok": True,
            "port": int(info.get("port") or core.port or 0),
            "version": str(info.get("version") or core.version or __version__),
            "ui_built": bool(info.get("ui_built", (paths.ui / "index.html").exists())),
        }
        LOG.info("smoke test passed: %s", payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        LOG.exception("smoke test failed")
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    finally:
        stop_core(core, paths)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="AI Recogniser",
        description="AI Recogniser desktop shell: window, tray, core lifecycle, autostart.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="start the core, wait for /healthz, print one line of JSON and exit",
    )
    parser.add_argument(
        "--autostart",
        choices=("on", "off", "status"),
        metavar="{on,off,status}",
        help="manage autostart in HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
    )
    parser.add_argument(
        "--version", action="version", version=f"{APP_TITLE} {__version__}"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.autostart is not None:
        return autostart.run_cli(args.autostart)

    paths = AppPaths.resolve()
    try:
        paths.ensure()
    except OSError as exc:
        print(
            f"{APP_TITLE}: cannot create the data directory {paths.data}: {exc}",
            file=sys.stderr,
        )
        return 1

    log_file = setup_logging(paths, console=not args.smoke_test)
    LOG.info(
        "%s shell %s starting (frozen=%s, portable=%s, data=%s)",
        APP_TITLE,
        __version__,
        bool(getattr(sys, "frozen", False)),
        paths.portable,
        paths.data,
    )

    try:
        if args.smoke_test:
            return run_smoke_test(paths)

        if not acquire_single_instance():
            LOG.info("another shell instance is already running")
            if not activate_existing_window():
                message_box(f"{APP_TITLE} уже запущен.", error=False)
            return 0

        return run_gui(paths)
    except Exception as exc:
        LOG.exception("fatal shell error")
        message_box(
            f"Не удалось запустить {APP_TITLE}.\n\n{exc}\n\nПодробности: {log_file}"  # noqa: RUF001
        )
        return 1
    finally:
        logging.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
