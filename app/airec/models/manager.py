"""Model downloader with resume, SHA-256 verification and offline import.

Models are downloaded to ``data/models/<id>/``.  Progress is reported through
a callback so the UI can show a progress bar.  After successful verification
the model is ready to use; failed downloads leave partial files for resume.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

import httpx

log = logging.getLogger(__name__)

CHUNK_BYTES = 1024 * 1024


class ModelError(RuntimeError):
    """Model download/verification error."""


class ModelManager:
    """Manage model downloads and verification."""

    def __init__(
        self,
        models_dir: Path,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        )
        self._lock = threading.RLock()
        self._downloads: dict[str, threading.Thread] = {}
        self._progress: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #
    def installed_models(self) -> list[dict[str, Any]]:
        """List locally installed models (directory per model id)."""
        installed: list[dict[str, Any]] = []
        if not self.models_dir.exists():
            return installed
        for directory in sorted(self.models_dir.iterdir()):
            if not directory.is_dir():
                continue
            complete = [
                f
                for f in directory.rglob("*")
                if f.is_file()
                and f.suffix != ".partial"
                and (
                    f.suffix.lower() in {".onnx", ".gguf", ".safetensors"}
                    or f.name.lower().endswith((".onnx.data", ".onnx_data"))
                )
            ]
            if not complete:
                continue
            size = sum(f.stat().st_size for f in complete)
            installed.append(
                {
                    "id": directory.name,
                    "path": str(directory),
                    "size_bytes": size,
                    "installed": True,
                }
            )
        return installed

    def model_path(self, model_id: str) -> Path:
        return self.models_dir / model_id

    def is_installed(self, model_id: str, filename: str | None = None) -> bool:
        directory = self.models_dir / model_id
        if not directory.is_dir():
            return False
        if filename:
            candidate = directory / Path(filename).name
            return candidate.is_file() and candidate.suffix != ".partial"
        return any(
            f.is_file() and f.suffix != ".partial" for f in directory.rglob("*")
        )
    # ------------------------------------------------------------------ #
    # Download
    # ------------------------------------------------------------------ #
    def download(
        self,
        model_id: str,
        url: str,
        *,
        sha256: str | None = None,
        size_bytes: int | None = None,
        filename: str | None = None,
        extra_files: list[dict[str, str]] | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> None:
        """Download *url* into ``models_dir/<model_id>/`` with resume.

        Runs in a background thread; progress callback receives (0..1, phase).
        *filename* keeps the original model file name (``model.onnx``,
        ``*.gguf``) so onnx-asr and llama.cpp can use the downloaded folder
        directly. *extra_files* are small companions such as ``config.json``
        and vocabulary files required by onnx-asr's local-directory loader.
        """
        with self._lock:
            if model_id in self._downloads:
                raise ModelError(f"model {model_id} is already being downloaded")
            self._progress[model_id] = {
                "value": 0.0,
                "phase": "Подготовка",
                "status": "running",
            }
            thread = threading.Thread(
                target=self._download_thread,
                args=(
                    model_id,
                    url,
                    sha256,
                    size_bytes,
                    filename,
                    extra_files or [],
                    progress,
                ),
                name=f"model-download-{model_id}",
                daemon=True,
            )
            self._downloads[model_id] = thread
            thread.start()

    def _download_thread(
        self,
        model_id: str,
        url: str,
        sha256: str | None,
        size_bytes: int | None,
        filename: str | None,
        extra_files: list[dict[str, str]],
        progress: Callable[[float, str], None] | None,
    ) -> None:
        notify = progress or (lambda _value, _phase: None)
        target_dir = self.model_path(model_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        # Keep the real filename (onnx-asr resolves `*.onnx`, llama.cpp needs a
        # `.gguf`).  Fall back to `<id>.bin` for entries without a filename.
        safe_name = Path(filename).name if filename else f"{model_id}.bin"
        target_file = target_dir / safe_name
        tmp_file = target_dir / f"{safe_name}.partial"

        def report(value: float, phase: str, status: str = "running") -> None:
            with self._lock:
                self._progress[model_id] = {
                    "value": max(0.0, min(float(value), 1.0)),
                    "phase": phase,
                    "status": status,
                }
            notify(value, phase)

        report(0.0, "Скачивание модели")
        try:
            # Companions first: a failed main download must not leave a folder
            # that looks installed but cannot be loaded offline.
            for index, extra in enumerate(extra_files):
                extra_url = str(extra.get("url") or "")
                extra_name = Path(str(extra.get("filename") or "")).name
                if not extra_url or not extra_name:
                    raise ModelError("у дополнительного файла нет url/filename")
                report(
                    0.02 * (index / max(len(extra_files), 1)),
                    f"Загрузка {extra_name}",
                )
                extra_tmp = target_dir / f"{extra_name}.partial"
                extra_tmp.unlink(missing_ok=True)
                self._perform_download(
                    extra_url,
                    extra_tmp,
                    lambda _value, _phase: report(
                        0.01 + 0.02 * (index / max(len(extra_files), 1)),
                        f"Загрузка {extra_name}",
                    ),
                )
                extra_tmp.replace(target_dir / extra_name)

            main_notify = (
                (lambda value, phase: report(0.02 + 0.88 * min(value, 0.9), phase))
                if extra_files
                else report
            )
            self._perform_download(url, tmp_file, main_notify)
            if size_bytes and tmp_file.stat().st_size != size_bytes:
                raise ModelError(
                    "размер модели не совпадает: "
                    f"ожидалось {size_bytes}, получено {tmp_file.stat().st_size}"
                )
            if sha256:
                report(0.95, "Проверка контрольной суммы")
                actual = self._sha256(tmp_file, report)
                if actual.lower() != sha256.lower():
                    raise ModelError(
                        f"контрольная сумма не совпадает: ожидалось {sha256}, "
                        f"получено {actual}"
                    )
            tmp_file.replace(target_file)
            report(1.0, "Готово", "done")
            log.info("model %s downloaded and verified to %s", model_id, target_file)
        except Exception as exc:  # download boundary
            log.exception("model %s download failed: %s", model_id, exc)
            report(1.0, f"Ошибка: {exc}", "error")
        finally:
            with self._lock:
                self._downloads.pop(model_id, None)

    def _perform_download(
        self,
        url: str,
        destination: Path,
        notify: Callable[[float, str], None],
    ) -> None:
        resume_from = destination.stat().st_size if destination.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        with self._client.stream("GET", url, headers=headers) as response:
            if response.status_code not in {200, 206}:
                raise ModelError(f"HTTP {response.status_code} для {url}")
            content_length = int(response.headers.get("content-length", "0") or "0")
            if response.status_code == 206 and resume_from:
                # The server honoured the Range request: keep the partial file.
                downloaded = resume_from
                total = resume_from + content_length
                mode = "ab"
            else:
                # No range support (or a fresh download): restart from scratch
                # instead of appending a full body to a partial file.
                if resume_from:
                    log.info("server did not resume %s; restarting download", url)
                downloaded = 0
                total = content_length
                mode = "wb"
            with destination.open(mode) as handle:
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        notify(min(downloaded / total, 0.9), "Скачивание модели")

    def _sha256(self, path: Path, notify: Callable[[float, str], None]) -> str:
        h = hashlib.sha256()
        total = path.stat().st_size
        read = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(CHUNK_BYTES)
                if not chunk:
                    break
                h.update(chunk)
                read += len(chunk)
                if total:
                    notify(0.9 + 0.08 * min(read / total, 1.0), "Проверка суммы")
        return h.hexdigest()

    def download_status(self, model_id: str) -> dict[str, Any] | None:
        """Return the latest download state for *model_id* or None."""
        with self._lock:
            progress = self._progress.get(model_id)
            return dict(progress) if progress is not None else None

    def cancel_download(self, model_id: str) -> None:
        """Cancel an in-progress download (best effort)."""
        with self._lock:
            thread = self._downloads.pop(model_id, None)
        # Threads cannot be forcibly killed in Python; the partial file remains
        # for resume.  We just stop tracking it.
        log.info("download %s cancelled (partial file kept for resume)", model_id)

    def import_model(self, source: Path, model_id: str) -> None:
        """Import a model from a local directory/archive (offline)."""
        source = Path(source)
        if not source.exists():
            raise ModelError(f"источник не найден: {source}")
        target_dir = self.model_path(model_id)
        if target_dir.exists():
            raise ModelError(f"модель {model_id} уже установлена")
        if source.is_dir():
            shutil.copytree(source, target_dir)
        elif source.is_file() and source.suffix.lower() in {".zip", ".7z"}:
            import zipfile

            with zipfile.ZipFile(source) as zf:
                zf.extractall(target_dir)
        else:
            shutil.copy2(source, target_dir / source.name)
        log.info("model %s imported from %s", model_id, source)
