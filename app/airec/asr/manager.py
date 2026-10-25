"""ASR job queue backed by SQLite.

State lives in ``data/airec_jobs.sqlite`` (WAL mode) so jobs survive process
restarts and crashes.  In-flight futures remain in memory (process-local); on
startup any jobs left ``queued``/``running`` are marked ``error`` with a clear
message so the user can re-upload them.

Per-job result folders (``data/results/<job_id>/`` with ``job.json`` and
``transcript.txt``) are kept for compatibility; ``job.json`` is now written as a
side-effect of the SQLite upsert so existing tooling keeps working.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from airec.asr.audio import SUPPORTED_EXTENSIONS, find_ffmpeg
from airec.asr.service import AsrError, TranscriptionResult, transcribe
from airec.core.config import Settings
from airec.core.db import Database
from airec.core.paths import AppPaths

log = logging.getLogger(__name__)

JOB_SCHEMA = 1
MAX_INPUT_BYTES = 4 * 1024 * 1024 * 1024
ACTIVE_STATUSES = {"queued", "running"}

Transcriber = Callable[..., TranscriptionResult]


class JobValidationError(ValueError):
    """The uploaded input is not a supported audio/video file."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _safe_name(name: str) -> str:
    candidate = Path(name or "audio").name.strip()
    if candidate in {"", ".", ".."}:
        return "audio"
    return candidate


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"
    progress: float = 0.0
    phase: str = "В очереди"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    model: str = ""
    duration_s: float | None = None
    elapsed_s: float | None = None
    rtf: float | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)
    size_bytes: int = 0
    audio_format: str = ""
    text: str = ""
    error: str | None = None
    result_file: str | None = None
    source_path: str | None = None

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("text", None)
        data.pop("result_file", None)
        data["segment_count"] = len(self.segments)
        data.pop("segments", None)
        return data

    def detail(self) -> dict[str, Any]:
        data = asdict(self)
        data["result_url"] = (
            f"/api/v1/jobs/{self.id}/result" if self.status == "done" else None
        )
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Job:
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in payload.items() if key in known})


class JobManager:
    def __init__(
        self,
        paths: AppPaths,
        settings_provider: Callable[[], Settings],
        *,
        transcriber: Transcriber = transcribe,
    ) -> None:
        self._paths = paths
        self._settings = settings_provider
        self._transcriber = transcriber
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="airec-asr")
        self._ffmpeg = find_ffmpeg(paths.bin)
        self._db = Database(paths.data / "airec_jobs.sqlite")
        self._load_existing()

    @property
    def results_dir(self) -> Path:
        return self._paths.results

    @property
    def max_input_bytes(self) -> int:
        return MAX_INPUT_BYTES

    def _job_dir(self, job_id: str) -> Path:
        return self._paths.results / job_id

    def _job_meta_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _persist(self, job: Job) -> None:
        """Write job state to SQLite and keep job.json in sync."""
        self._db.upsert_job(asdict(job))
        self._write_job_json(job)

    def _write_job_json(self, job: Job) -> None:
        """Side-car JSON for external tooling; best-effort, never fatal."""
        try:
            directory = self._job_dir(job.id)
            directory.mkdir(parents=True, exist_ok=True)
            target = self._job_meta_path(job.id)
            tmp = target.with_suffix(".json.tmp")
            payload = json.dumps(asdict(job), ensure_ascii=False, indent=2)
            with tmp.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        except OSError:
            log.exception("cannot write side-car job.json for %s", job.id)

    def _load_existing(self) -> None:
        """Load jobs from SQLite; migrate legacy job.json files on first run."""
        db_jobs = self._db.list_jobs()
        if db_jobs:
            for row in db_jobs:
                self._rehydrate(row)
            return

        # First run with SQLite: migrate any legacy per-job job.json files.
        if not self.results_dir.exists():
            return
        migrated = 0
        for directory in sorted(self.results_dir.iterdir(), reverse=True):
            if not directory.is_dir():
                continue
            meta = directory / "job.json"
            if not meta.is_file():
                continue
            try:
                payload = json.loads(meta.read_text(encoding="utf-8-sig"))
                job = Job.from_dict(payload)
            except (OSError, ValueError, TypeError) as exc:
                log.warning("cannot read legacy job metadata %s: %s", meta, exc)
                continue
            if job.status in ACTIVE_STATUSES:
                job.status = "error"
                job.phase = "Прервано"
                job.error = (
                    "Задача была прервана при закрытии приложения. Запустите её снова."
                )
                job.finished_at = utc_now()
            self._db.upsert_job(asdict(job))
            self._write_job_json(job)
            self._rehydrate(asdict(job))
            migrated += 1
        if migrated:
            log.info("migrated %d legacy job.json files to SQLite", migrated)

    def _rehydrate(self, row: dict[str, Any]) -> None:
        """Add a job row from SQLite into the in-memory index (metadata only)."""
        segments = row.get("segments")
        if isinstance(segments, str):
            try:
                segments = json.loads(segments)
            except (TypeError, ValueError):
                segments = []
        elif not isinstance(segments, list):
            segments = []
        row["segments"] = segments
        job = Job.from_dict(row)
        with self._lock:
            self._jobs[job.id] = job

    # ------------------------------------------------------------------ #
    # Submission
    # ------------------------------------------------------------------ #
    def submit(self, uploaded: Path, original_name: str, model_id: str) -> Job:
        filename = _safe_name(original_name)
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise JobValidationError(
                "Неподдерживаемый формат файла. Используйте mp3, mp4, m4a, wav, "
                "flac, ogg, opus, mkv, mov, webm или avi."
            )
        if not uploaded.is_file():
            raise JobValidationError("Загруженный файл не найден.")
        size = uploaded.stat().st_size
        if size <= 0:
            raise JobValidationError("Файл пуст.")
        if size > MAX_INPUT_BYTES:
            raise JobValidationError("Файл больше 4 ГБ.")

        job_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
        work_dir = self._paths.temp / "jobs" / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        source = work_dir / f"input{suffix}"
        shutil.move(str(uploaded), source)

        # Keep a copy of the source file next to the results so the job can be
        # retried even after the temp folder is cleaned up.
        result_dir = self._job_dir(job_id)
        result_dir.mkdir(parents=True, exist_ok=True)
        source_copy = result_dir / f"source{suffix}"
        shutil.copy2(source, source_copy)

        job = Job(
            id=job_id,
            filename=filename,
            model=model_id,
            size_bytes=size,
            audio_format=suffix.lstrip("."),
            source_path=str(source_copy),
        )
        with self._lock:
            self._jobs[job.id] = job
            try:
                self._persist(job)
            except OSError:
                log.exception("cannot persist new job %s", job.id)
            future = self._executor.submit(self._run, job.id, source, work_dir)
            self._futures[job.id] = future
        return job

    # ------------------------------------------------------------------ #
    # Cancel / retry
    # ------------------------------------------------------------------ #
    def cancel(self, job_id: str) -> None:
        """Отменить задачу (queued или running)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Job {job_id} not found.")
            if job.status not in ACTIVE_STATUSES:
                raise ValueError(
                    f"Нельзя отменить задачу со статусом '{job.status}'."
                )
            future = self._futures.pop(job_id, None)
            if future is not None:
                future.cancel()
            job.status = "cancelled"
            job.phase = "Отменено"
            job.finished_at = utc_now()
            job.error = "Задача отменена пользователем."
            try:
                self._persist(job)
            except OSError:
                log.exception("cannot persist cancelled job %s", job_id)
            # Освобождаем временные файлы.
            work_dir = self._paths.temp / "jobs" / job_id
            self._cleanup_source(work_dir)

    def retry(self, job_id: str) -> Job:
        """Повтор задачи со статусом error или cancelled.

        Исходный файл берётся из results/<job_id>/source<ext>, который
        хранится независимо от временной папки.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Job {job_id} not found.")
            if job.status not in {"error", "cancelled"}:
                raise ValueError(
                    f"Повтор доступен только для задач со статусом 'error' или "
                    f"'cancelled', получено '{job.status}'."
                )
            if not job.source_path:
                raise ValueError(
                    f"У задачи {job_id} нет сохранённого исходного файла."
                )
            source = Path(job.source_path)
            if not source.is_file():
                raise FileNotFoundError(
                    f"Исходный файл задачи не найден: {source}"
                )

            # Создаём новую задачу на основе старой.
            new_job_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
            new_work_dir = self._paths.temp / "jobs" / new_job_id
            new_work_dir.mkdir(parents=True, exist_ok=True)
            new_source = new_work_dir / f"input{source.suffix}"
            shutil.copy2(source, new_source)

            new_job = Job(
                id=new_job_id,
                filename=job.filename,
                model=job.model,
                size_bytes=source.stat().st_size,
                audio_format=job.audio_format,
                source_path=str(source),
            )
            self._jobs[new_job.id] = new_job
            try:
                self._persist(new_job)
            except OSError:
                log.exception("cannot persist retried job %s", new_job.id)
            future = self._executor.submit(
                self._run, new_job.id, new_source, new_work_dir
            )
            self._futures[new_job.id] = future
            log.info("job %s retried as %s", job_id, new_job.id)
            return new_job

    # ------------------------------------------------------------------ #
    # Worker
    # ------------------------------------------------------------------ #
    def _run(self, job_id: str, source: Path, work_dir: Path) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "running"
            job.phase = "Запуск"
            job.started_at = utc_now()
            job.progress = 0.02
            try:
                self._persist(job)
            except OSError:
                log.exception("cannot persist started job %s", job.id)
            settings = self._settings()

        def progress(value: float, phase: str) -> None:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is None:
                    return
                current.progress = max(current.progress, min(max(value, 0.0), 0.99))
                current.phase = phase
                try:
                    self._db.upsert_job(asdict(current))
                except OSError:
                    log.exception("cannot persist progress for job %s", job_id)

        try:
            result = self._transcriber(
                source,
                model_id=job.model,
                cpu_threads=settings.cpu_threads,
                ffmpeg=self._ffmpeg,
                work_dir=work_dir,
                progress=progress,
                vad_enabled=settings.vad_enabled,
                vad_max_segment_s=settings.vad_max_segment_s,
                vad_min_silence_ms=settings.vad_min_silence_ms,
                vad_speech_pad_ms=settings.vad_speech_pad_ms,
                diarize_enabled=settings.diarize_enabled,
                diarize_threshold=settings.diarize_threshold,
                diarize_model_path=settings.diarize_model_path,
            )
            transcript_path = self._job_dir(job_id) / "transcript.txt"
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text(result.text + "\n", encoding="utf-8", newline="\n")
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job.status = "done"
                job.phase = "Готово"
                job.progress = 1.0
                job.finished_at = utc_now()
                job.text = result.text
                job.duration_s = result.duration_s
                job.elapsed_s = result.elapsed_s
                job.rtf = result.rtf
                job.segments = [asdict(segment) for segment in result.segments]
                job.result_file = str(transcript_path)
                try:
                    self._persist(job)
                except OSError:
                    log.exception("cannot persist finished job %s", job.id)
            if not settings.keep_extracted_audio:
                self._cleanup_source(work_dir)
        except Exception as exc:  # worker boundary: never let a job kill the queue
            message = str(exc) if isinstance(exc, AsrError) else f"Ошибка обработки: {exc}"
            log.exception("job %s failed: %s", job_id, exc)
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job.status = "error"
                job.phase = "Ошибка"
                job.progress = 1.0
                job.error = message
                job.finished_at = utc_now()
                try:
                    self._persist(job)
                except OSError:
                    log.exception("cannot persist failed job %s", job.id)

    @staticmethod
    def _cleanup_source(work_dir: Path) -> None:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except OSError:
            log.warning("cannot remove temporary job directory %s", work_dir)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            # Prefer the authoritative SQLite state, merged with in-memory jobs
            # (freshly submitted ones that may not be committed yet).
            rows = self._db.list_jobs()
            jobs = [Job.from_dict(row) for row in rows]
            for job in self._jobs.values():
                if job.id not in {j.id for j in jobs}:
                    jobs.append(job)
            jobs.sort(key=lambda item: item.created_at, reverse=True)
            return [job.summary() for job in jobs]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job
            row = self._db.get_job(job_id)
            if row is None:
                return None
            job = Job.from_dict(row)
            self._jobs[job.id] = job
            return job

    def result_path(self, job_id: str) -> Path | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                job = self.get(job_id)
            if job is None or job.status != "done" or not job.result_file:
                return None
            path = Path(job.result_file)
            return path if path.is_file() else None

    def export_result(self, job_id: str, format: str) -> tuple[Path, str, str]:
        """Export a finished job in *format*.

        Returns ``(file_path, download_name, content_type)``.  The exported
        file is written to the job's results folder next to ``transcript.txt``.
        """
        from airec.asr.export import EXPORT_CONTENT_TYPES, EXPORT_DOWNLOAD_NAMES, export

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                job = self.get(job_id)
            if job is None:
                raise FileNotFoundError(f"Job {job_id} not found.")
            if job.status != "done":
                raise ValueError(f"Job {job_id} is not finished (status={job.status}).")

        segments = job.segments or []
        content = export(segments, format, metadata={
            "job_id": job.id,
            "filename": job.filename,
            "model": job.model,
            "duration_s": job.duration_s,
            "speaker_count": len({s.get("speaker") for s in segments if s.get("speaker")}),
        })

        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(job.filename).stem or "transcript"
        ext = EXPORT_DOWNLOAD_NAMES[format]
        out_path = job_dir / f"{stem}{ext}"
        out_path.write_text(content, encoding="utf-8", newline="\n")
        return out_path, EXPORT_DOWNLOAD_NAMES[format], EXPORT_CONTENT_TYPES[format]

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        try:
            self._db.close()
        except Exception:  # never crash the shell on shutdown
            log.exception("cannot close job database")
