"""Local llama.cpp runtime (``llama-server``) discovery and installation.

The app does not ship llama.cpp binaries: they are downloaded on demand from
the pinned llama.cpp CPU release and unpacked into the writable application
data directory.  The provider also looks next to the executable and in
``PATH`` so advanced users can use their own build.
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

RUNTIME_ID = "llama-server-cpu-x64"
RUNTIME_FILENAME = "llama-server-cpu-x64.zip"
DEFAULT_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/b11946/"
    "llama-b11946-bin-win-cpu-x64.zip"
)
DEFAULT_SHA256 = "8A702BB07C873B133C56EB3ACD92A48D7C5C9AD339C6435EDDC5185E46E1A043"
DEFAULT_SIZE_BYTES = 24_376_086

# Vulkan-сборка работает и на дискретных GPU, и на iGPU (AMD Radeon 780M,
# Intel Arc/Iris Xe) через единый драйвер — отдельные CUDA-сборки не нужны.
RUNTIME_ID_VULKAN = "llama-server-vulkan-x64"
RUNTIME_FILENAME_VULKAN = "llama-server-vulkan-x64.zip"
DEFAULT_URL_VULKAN = (
    "https://github.com/ggml-org/llama.cpp/releases/download/b11946/"
    "llama-b11946-bin-win-vulkan-x64.zip"
)
DEFAULT_SHA256_VULKAN = "B65B2837C584CA2D37033B8A947513FC02649199077A17455F416921AA6EB8F5"
DEFAULT_SIZE_BYTES_VULKAN = 37_303_240

EXECUTABLE_NAMES = ("llama-server.exe", "llama-server")
ProgressCallback = Callable[[float, str], None]
CHUNK_BYTES = 1024 * 1024


class RuntimeInstallError(RuntimeError):
    """Runtime download/install error."""


def runtime_dir(paths: AppPaths, *, device: str = "cpu") -> Path:
    """Writable directory for the downloaded llama.cpp runtime.

    CPU- и Vulkan-сборки живут в разных папках: обе могут быть установлены
    одновременно, переключение устройства в настройках не требует повторного
    скачивания.
    """
    subdir = "llama-server-vulkan" if device == "vulkan" else "llama-server"
    return paths.data / "runtime" / subdir


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


def find_llama_server(
    paths: AppPaths | None = None,
    custom: str | Path | None = None,
    *,
    device: str = "cpu",
) -> Path | None:
    """Return the first usable ``llama-server`` executable, or ``None``.

    *device* выбирает, какую сборку искать первой: при ``vulkan`` —
    Vulkan-папку runtime, при ``cpu`` — обычную.  Пользовательский путь
    (*custom*) и PATH имеют приоритет над обеими.
    """
    if custom:
        candidate = Path(custom).expanduser()
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            for name in EXECUTABLE_NAMES:
                found = candidate / name
                if found.is_file():
                    return found
    resolved = paths or AppPaths.resolve()
    preferred = runtime_dir(resolved, device=device)
    others = [d for d in _candidate_dirs(resolved) if d != preferred]
    for directory in [preferred, *others]:
        for name in EXECUTABLE_NAMES:
            found = directory / name
            if found.is_file():
                return found
    found = shutil.which("llama-server") or shutil.which("llama-server.exe")
    return Path(found) if found else None


def runtime_installed(paths: AppPaths | None = None, *, device: str = "cpu") -> bool:
    return find_llama_server(paths, device=device) is not None


def _runtime_entry(*, device: str = "cpu") -> dict[str, Any]:
    runtime_id = RUNTIME_ID_VULKAN if device == "vulkan" else RUNTIME_ID
    try:
        manifest = load_manifest(manifest_dir())
        entry = find_model(manifest, runtime_id)
        if entry is not None:
            return entry
        for candidate in manifest.get("models", []):
            if candidate.get("kind") == "runtime" and (
                candidate.get("id") == runtime_id or device == "cpu"
            ):
                return candidate
    except (OSError, ValueError):
        log.debug("cannot read runtime entry from manifest", exc_info=True)
    if device == "vulkan":
        return {
            "id": RUNTIME_ID_VULKAN,
            "filename": RUNTIME_FILENAME_VULKAN,
            "url": DEFAULT_URL_VULKAN,
            "sha256": DEFAULT_SHA256_VULKAN,
            "size_bytes": DEFAULT_SIZE_BYTES_VULKAN,
        }
    return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def install_runtime_zip(
    zip_path: Path, paths: AppPaths, *, device: str = "cpu"
) -> Path:
    """Extract a llama.cpp Windows archive into the runtime directory."""
    target = runtime_dir(paths, device=device)
    staging = target.with_name(target.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if not name:
                    continue
                destination = staging / name
                with archive.open(info) as source, destination.open("wb") as out:
                    shutil.copyfileobj(source, out)
        if not any((staging / name).is_file() for name in EXECUTABLE_NAMES):
            raise RuntimeInstallError(
                "В архиве llama.cpp не найден llama-server.exe. "
                "Скачайте официальный Windows CPU-архив."
            )
        shutil.rmtree(target, ignore_errors=True)
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    found = find_llama_server(paths, device=device)
    if found is None:
        raise RuntimeInstallError("llama-server установлен, но не найден после распаковки.")
    log.info("llama-server (%s) installed: %s", device, found)
    return found


def _download(
    url: str,
    target: Path,
    *,
    progress: ProgressCallback | None,
    client: httpx.Client | None = None,
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
                raise RuntimeInstallError(
                    f"Не удалось скачать llama-server: HTTP {response.status_code}"
                )
            if response.status_code == 200 and existing:
                existing = 0
                target.unlink(missing_ok=True)
            total_header = response.headers.get("content-length")
            total = existing + int(total_header) if total_header and total_header.isdigit() else 0
            downloaded = existing
            mode = "ab" if existing and response.status_code == 206 else "wb"
            with target.open(mode) as handle:
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        notify(
                            min(downloaded / total * 0.8, 0.8),
                            "Скачивание llama-server",
                        )
            notify(0.82, "Проверка llama-server")
    finally:
        if owned:
            http.close()


def ensure_runtime(
    paths: AppPaths,
    *,
    progress: ProgressCallback | None = None,
    client: httpx.Client | None = None,
    force: bool = False,
    device: str = "cpu",
) -> Path:
    """Return an installed llama-server, downloading it on first use.

    *device* — ``cpu`` или ``vulkan``: Vulkan-сборка нужна для GPU-ускорения
    саммари и скачивается отдельно при первом запуске с устройством ``vulkan``.
    """
    found = find_llama_server(paths, device=device)
    if found is not None and not force:
        return found

    entry = _runtime_entry(device=device)
    if device == "vulkan" and not entry.get("url"):
        raise RuntimeInstallError(
            "Vulkan-сборка llama.cpp недоступна в манифесте моделей."
        )
    url = str(entry.get("url") or DEFAULT_URL)
    expected_sha = str(entry.get("sha256") or DEFAULT_SHA256).upper()
    expected_size = int(entry.get("size_bytes") or 0)
    fallback_size = (
        DEFAULT_SIZE_BYTES_VULKAN if device == "vulkan" else DEFAULT_SIZE_BYTES
    )
    if not expected_size:
        expected_size = fallback_size
    filename = str(entry.get("filename") or RUNTIME_FILENAME)
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
        _download(url, partial, progress=progress, client=client)
    notify(0.86, "Проверка размера llama-server")
    if expected_size and partial.stat().st_size != expected_size:
        partial.unlink(missing_ok=True)
        raise RuntimeInstallError(
            "Размер скачанного llama-server не совпадает с ожидаемым. "
            "Повторите скачивание."
        )
    if expected_sha:
        notify(0.9, "Проверка контрольной суммы llama-server")
        actual = _sha256(partial)
        if actual.upper() != expected_sha:
            partial.unlink(missing_ok=True)
            raise RuntimeInstallError(
                "Контрольная сумма llama-server не совпадает. "
                "Скачайте runtime заново."
            )
    notify(0.94, "Установка llama-server")
    found = install_runtime_zip(partial, paths, device=device)
    partial.unlink(missing_ok=True)
    notify(1.0, "llama-server готов")
    return found

