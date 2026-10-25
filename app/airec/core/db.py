"""SQLite storage for ASR jobs.

Replaces per-job JSON metadata with a single WAL database so job state survives
process restarts and crashes without losing in-flight work.  One connection per
process, WAL mode, synchronous=NORMAL — a good trade-off for a single-writer
desktop queue.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# All Job dataclass fields that are persisted, in column order.
JOB_COLUMNS = (
    "id",
    "filename",
    "status",
    "progress",
    "phase",
    "created_at",
    "started_at",
    "finished_at",
    "model",
    "duration_s",
    "elapsed_s",
    "rtf",
    "segments",
    "size_bytes",
    "audio_format",
    "text",
    "error",
    "result_file",
)

CREATE_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    progress REAL NOT NULL DEFAULT 0.0,
    phase TEXT NOT NULL DEFAULT 'В очереди',
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    model TEXT NOT NULL DEFAULT '',
    duration_s REAL,
    elapsed_s REAL,
    rtf REAL,
    segments TEXT,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    audio_format TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    error TEXT,
    result_file TEXT,
    segment_count INTEGER NOT NULL DEFAULT 0
);
"""

CREATE_META_TABLE = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Database:
    """Thread-safe SQLite wrapper for job persistence."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = self._connect()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(CREATE_META_TABLE)
            self._conn.execute(CREATE_JOBS_TABLE)
            # Record schema version so migrations can be added later.
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", ("schema_version",)
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    ("schema_version", str(SCHEMA_VERSION)),
                )
            log.info("SQLite job store initialised at %s (schema v%s)", self._path, SCHEMA_VERSION)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    # Jobs
    # ------------------------------------------------------------------ #
    def upsert_job(self, job: dict[str, Any]) -> None:
        """Insert or replace a job row from a Job.asdict()-style mapping."""
        values: list[Any] = []
        for column in JOB_COLUMNS:
            value = job.get(column)
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            values.append(value)
        columns = ", ".join(JOB_COLUMNS)
        placeholders = ", ".join("?" for _ in JOB_COLUMNS)
        with self._lock:
            self._conn.execute(
                f"""
                INSERT INTO jobs ({columns})
                VALUES ({placeholders})
                ON CONFLICT(id) DO UPDATE SET
                    filename=excluded.filename,
                    status=excluded.status,
                    progress=excluded.progress,
                    phase=excluded.phase,
                    created_at=excluded.created_at,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    model=excluded.model,
                    duration_s=excluded.duration_s,
                    elapsed_s=excluded.elapsed_s,
                    rtf=excluded.rtf,
                    segments=excluded.segments,
                    size_bytes=excluded.size_bytes,
                    audio_format=excluded.audio_format,
                    text=excluded.text,
                    error=excluded.error,
                    result_file=excluded.result_file
                """,
                values,
            )

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        if data.get("segments"):
            try:
                data["segments"] = json.loads(data["segments"])
            except (TypeError, ValueError):
                data["segments"] = []
        else:
            data["segments"] = []
        return data

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return self._row_to_dict(row) if row is not None else None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"
            ).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def delete_job(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def active_jobs(self) -> list[dict[str, Any]]:
        """Jobs that were queued/running when the process stopped."""
        placeholders = ",".join("?" for _ in ("queued", "running"))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM jobs WHERE status IN ({placeholders})",
                ("queued", "running"),
            ).fetchall()
            return [self._row_to_dict(row) for row in rows]
