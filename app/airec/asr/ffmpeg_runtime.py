"""On-demand FFmpeg runtime (LGPL) discovery and installation.

The application no longer ships FFmpeg.  The previously bundled ``imageio-ffmpeg``
binary was built with ``--enable-gpl --enable-version3``, which would have made
the whole distribution subject to the GPL even though the project is MIT.

Instead an **LGPL** shared build (BtbN FFmpeg-Builds, pinned autobuild) is
downloaded on demand into the writable application data directory the first time
a non-WAV file needs decoding.  PCM WAV input keeps working without FFmpeg.
Advanced users can also drop their own ``ffmpeg.exe`` into ``bin`` next to the
executable, or rely on ``PATH``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from airec.core.paths import AppPaths, app_root, resource_root
from airec.models.manifest import find_model, load_manifest, manifest_dir

log = logging.getLogger(__name__)

FFMPEG_ID = "ffmpeg-lgpl-x64"
FFMPEG_FILENAME = "ffmpeg-lgpl-x64.zip"

# BtbN FFmpeg-Builds, immutable autobuild tag.  Stable 8.1 branch, LGPL shared
# (ffmpeg.exe + av*.dll); no --enable-gpl, no x264/x265.
DEFAULT_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/"
    "autobuild-2026-10-28-19-47/"
    "ffmpeg-n8.1.3-25-g611cec792a-win64-lgpl-shared-8.1.zip"
)
DEFAULT_SHA256 = "0448C3F3B9BDC724CC0189D5CE6E6BFA5B8937E7EC408E1E97FB08719BDB3B92"
DEFAULT_SIZE_BYTES = 81_649_021

EXECUTABLE_NAMES = ("ffmpeg.exe", "ffmpeg")
ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], None]
CHUNK_BYTES = 1024 * 1024


class FfmpegInstallError(RuntimeError):
    """FFmpeg download/install error."""


class FfmpegCancelled(RuntimeError):
    """The FFmpeg download was cancelled by the caller."""


def runtime_dir(paths: AppPaths) -> Path:
    """Writable directory for the downloaded FFmpeg runtime."""
    return paths.data / "runtime" / "ffmpeg"


def _candidate_dirs(paths: AppPaths | None = None) -> list[Path]:
    resolved = paths or AppPaths.resolve()
    candidates = [
        runtime_dir(resolved),
        resolved.bin,
        resolved.root / "bin",
        resource_root() / "bin",
        app_root() / "bin",
        app_root() / "_internal" / "bin",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def find_ffmpeg(
    paths: AppPaths | None = None, custom: str | Path | None = None
) -> Path | None:
    """Return the first usable ``ffmpeg`` executable, or ``None``."""
    if custom:
        candidate = Path(custom).expanduser()
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            for name in EXECUTABLE_NAMES:
                found = candidate / name
                if found.is_file():
                    return found
    for directory in _candidate_dirs(paths):
        for name in EXECUTABLE_NAMES:
            found = directory / name
            if found.is_file():
                return found
    found = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    return Path(found) if found else None


def ffmpeg_installed(paths: AppPaths | None = None) -> bool:
    return find_ffmpeg(paths) is not None


def _runtime_entry() -> dict[str, Any]:
    try:
        manifest = load_manifest(manifest_dir())
        entry = find_model(manifest, FFMPEG_ID)
        if entry is not None:
            return entry
        for candidate in manifest.get("models", []):
            if candidate.get("id") == FFMPEG_ID:
                return candidate
    except (OSError, ValueError):
        log.debug("cannot read FFmpeg entry from manifest", exc_info=True)
    return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def install_ffmpeg_zip(zip_path: Path, paths: AppPaths) -> Path:
    """Extract the FFmpeg ``bin`` files (ffmpeg.exe + av*.dll) flat."""
    target = runtime_dir(paths)
    staging = target.with_name(target.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            extracted = 0
            for info in archive.infolist():
                if info.is_dir():
                    continue
                parts = Path(info.filename).parts
                name = parts[-1] if parts else ""
                if not name:
                    continue
                # BtbN archives keep the binaries under ``<root>/bin/``; a flat
                # archive has no directories at all.  Ignore headers, docs, etc.
                in_bin = "bin" in parts[:-1]
                is_flat = len(parts) == 1
                if not (in_bin or is_flat):
                    continue
                destination = staging / name
                with archive.open(info) as source, destination.open("wb") as out:
                    shutil.copyfileobj(source, out)
                extracted += 1
            if extracted == 0 or not any(
                (staging / name).is_file() for name in EXECUTABLE_NAMES
            ):
                raise FfmpegInstallError(
                    "В архиве FFmpeg не найден ffmpeg.exe. "
                    "Скачайте официальную LGPL-сборку BtbN."
                )
        shutil.rmtree(target, ignore_errors=True)
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    found = find_ffmpeg(paths)
    if found is None:
        raise FfmpegInstallError("FFmpeg установлен, но не найден после распаковки.")
    log.info("ffmpeg installed: %s", found)
    return found


def _download(
    url: str,
    target: Path,
    *,
    progress: ProgressCallback | None,
    client: httpx.Client | None = None,
    cancel: CancelCallback | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.stat().st_size if target.is_file() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    notify = progress or (lambda _value, _phase: None)
    owned = client is None
    http = client or httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(120.0, connect=30.0),
    )
    try:
        with http.stream("GET", url, headers=headers) as response:
            if response.status_code not in {200, 206}:
                raise FfmpegInstallError(
                    f"Не удалось скачать FFmpeg: HTTP {response.status_code}"
                )
            if response.status_code == 200 and existing:
                existing = 0
                target.unlink(missing_ok=True)
            total_header = response.headers.get("content-length")
            total = (
                existing + int(total_header)
                if total_header and total_header.isdigit()
                else 0
            )
            downloaded = existing
            mode = "ab" if existing and response.status_code == 206 else "wb"
            with target.open(mode) as handle:
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    if cancel is not None:
                        cancel()
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        notify(
                            min(downloaded / total * 0.8, 0.8),
                            "Скачивание FFmpeg",
                        )
            notify(0.82, "Проверка FFmpeg")
    finally:
        if owned:
            http.close()


def ensure_ffmpeg(
    paths: AppPaths,
    *,
    progress: ProgressCallback | None = None,
    client: httpx.Client | None = None,
    cancel: CancelCallback | None = None,
    force: bool = False,
) -> Path:
    """Return an installed ffmpeg, downloading the LGPL build on first use."""
    found = find_ffmpeg(paths)
    if found is not None and not force:
        return found

    entry = _runtime_entry()
    url = str(entry.get("url") or DEFAULT_URL)
    expected_sha = str(entry.get("sha256") or DEFAULT_SHA256).upper()
    expected_size = int(entry.get("size_bytes") or DEFAULT_SIZE_BYTES)
    filename = str(entry.get("filename") or FFMPEG_FILENAME)
    partial = paths.temp / f"{filename}.partial"
    notify = progress or (lambda _value, _phase: None)

    skip_download = False
    if partial.is_file() and expected_size:
        partial_size = partial.stat().st_size
        if partial_size > expected_size:
            partial.unlink(missing_ok=True)
        elif partial_size == expected_size:
            # A previous attempt downloaded the whole archive but crashed
            # before verification.  Do not send a Range request at EOF (416).
            skip_download = True
    if not skip_download:
        _download(url, partial, progress=progress, client=client, cancel=cancel)
    if cancel is not None:
        cancel()
    notify(0.86, "Проверка размера FFmpeg")
    if expected_size and partial.stat().st_size != expected_size:
        partial.unlink(missing_ok=True)
        raise FfmpegInstallError(
            "Размер скачанного FFmpeg не совпадает с ожидаемым. Повторите скачивание."
        )
    if expected_sha:
        notify(0.9, "Проверка контрольной суммы FFmpeg")
        actual = _sha256(partial)
        if actual.upper() != expected_sha:
            partial.unlink(missing_ok=True)
            raise FfmpegInstallError(
                "Контрольная сумма FFmpeg не совпадает. Скачайте FFmpeg заново."
            )
    notify(0.94, "Установка FFmpeg")
    found = install_ffmpeg_zip(partial, paths)
    partial.unlink(missing_ok=True)
    notify(1.0, "FFmpeg готов")
    return found
