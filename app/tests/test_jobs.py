from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from airec.api.app import create_app
from airec.asr.manager import Job, JobManager, JobValidationError
from airec.asr.service import TranscriptionResult
from airec.asr.vad import SpeechSegment
from airec.core.config import Settings
from airec.core.paths import AppPaths

TOKEN = "test-token"


def paths_for(tmp_path: Path) -> AppPaths:
    paths = AppPaths(
        root=tmp_path,
        data=tmp_path,
        models=tmp_path / "models",
        cache=tmp_path / "cache",
        logs=tmp_path / "logs",
        results=tmp_path / "results",
        temp=tmp_path / "temp",
        bin=tmp_path / "bin",
        ui=tmp_path / "ui-dist",
    )
    paths.ensure()
    return paths


def fake_transcriber(source: Path, **kwargs: object) -> TranscriptionResult:
    assert source.is_file()
    progress = kwargs["progress"]
    assert callable(progress)
    progress(0.5, "Распознавание")
    time.sleep(1.0)  # keep job running long enough to cancel
    return TranscriptionResult(
        text="Привет, это тестовая расшифровка.",
        model_id=str(kwargs["model_id"]),
        duration_s=12.0,
        elapsed_s=1.1,
        rtf=10.9,
        segments=(
            SpeechSegment(0.0, 6.0, "Привет, это"),
            SpeechSegment(6.0, 12.0, "тестовая расшифровка."),
        ),
    )


def wait_done(manager: JobManager, job_id: str, timeout: float = 5.0) -> Job:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        assert job is not None
        if job.status in {"done", "error"}:
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_manager_transcribes_and_persists(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)
    manager = JobManager(paths, lambda: Settings(), transcriber=fake_transcriber)
    upload = tmp_path / "meeting.wav"
    upload.write_bytes(b"RIFF....WAVEfmt ")

    job = manager.submit(upload, "meeting.wav", "gigaam-v3-e2e-ctc")
    finished = wait_done(manager, job.id)

    assert finished.status == "done"
    assert finished.text == "Привет, это тестовая расшифровка."
    assert finished.rtf == 10.9
    assert len(finished.segments) == 2
    assert finished.detail()["segments"][0]["text"] == "Привет, это"
    assert finished.detail()["result_url"] == f"/api/v1/jobs/{job.id}/result"
    result = manager.result_path(job.id)
    assert result is not None
    assert result.read_text(encoding="utf-8").startswith("Привет")
    summary = manager.list()[0]
    assert "text" not in summary
    assert "result_file" not in summary
    assert "segments" not in summary
    assert summary["segment_count"] == 2

    manager.shutdown()
    restored = JobManager(paths, lambda: Settings())
    restored_job = restored.get(job.id)
    assert restored_job is not None
    assert restored_job.status == "done"
    assert restored_job.text == finished.text
    assert len(restored_job.segments) == 2
    assert restored_job.segments[0]["text"] == "Привет, это"
    assert restored.result_path(job.id) is not None
    restored.shutdown()


def test_manager_cancel_queued_job(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)
    manager = JobManager(paths, lambda: Settings(), transcriber=fake_transcriber)
    upload = tmp_path / "meeting.wav"
    upload.write_bytes(b"RIFF....WAVEfmt ")

    job = manager.submit(upload, "meeting.wav", "gigaam-v3-e2e-ctc")
    # Wait until running, then cancel.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if manager.get(job.id).status == "running":
            break
        time.sleep(0.02)
    manager.cancel(job.id)

    cancelled = manager.get(job.id)
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.phase != ""
    assert "отмен" in (cancelled.error or "").lower()
    manager.shutdown()


def test_manager_cancel_non_active_job_raises(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)
    manager = JobManager(paths, lambda: Settings(), transcriber=fake_transcriber)
    upload = tmp_path / "meeting.wav"
    upload.write_bytes(b"RIFF....WAVEfmt ")

    job = manager.submit(upload, "meeting.wav", "gigaam-v3-e2e-ctc")
    finished = wait_done(manager, job.id)

    with pytest.raises(ValueError):
        manager.cancel(finished.id)
    manager.shutdown()


def test_manager_retry_error_job(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)
    manager = JobManager(paths, lambda: Settings(), transcriber=fake_transcriber)
    upload = tmp_path / "meeting.wav"
    upload.write_bytes(b"RIFF....WAVEfmt ")

    job = manager.submit(upload, "meeting.wav", "gigaam-v3-e2e-ctc")
    finished = wait_done(manager, job.id)
    assert finished.status == "done"

    # Повтор выполненной задачи должен быть ошибкой (не error/cancelled).
    with pytest.raises(ValueError):
        manager.retry(finished.id)

    # Повтор отмененной задачи — ок.
    upload2 = tmp_path / "meeting2.wav"
    upload2.write_bytes(b"RIFF....WAVEfmt ")
    job2 = manager.submit(upload2, "meeting2.wav", "gigaam-v3-e2e-ctc")
    manager.cancel(job2.id)

    retried = manager.retry(job2.id)
    assert retried.status == "queued"
    assert retried.filename == "meeting2.wav"
    assert retried.id != job2.id

    retried_done = wait_done(manager, retried.id)
    assert retried_done.status == "done"
    assert retried_done.text == "Привет, это тестовая расшифровка."
    manager.shutdown()


def test_manager_rejects_unsupported_extension(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)
    manager = JobManager(paths, lambda: Settings(), transcriber=fake_transcriber)
    upload = tmp_path / "notes.txt"
    upload.write_text("no audio", encoding="utf-8")
    with pytest.raises(JobValidationError):
        manager.submit(upload, "notes.txt", "gigaam-v3-e2e-ctc")
    manager.shutdown()


def test_interrupted_job_is_marked_as_error(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)
    job = Job(id="20260101-000000-deadbeef", filename="old.wav", status="running")
    directory = paths.results / job.id
    directory.mkdir(parents=True)
    (directory / "job.json").write_text(
        json.dumps(asdict(job), ensure_ascii=False),
        encoding="utf-8",
    )

    manager = JobManager(paths, lambda: Settings(), transcriber=fake_transcriber)
    restored = manager.get(job.id)
    assert restored is not None
    assert restored.status == "error"
    assert "прервана" in (restored.error or "")
    manager.shutdown()


def test_upload_endpoint_with_fake_worker(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)
    settings = Settings()
    app = create_app(settings=settings, paths=paths, token=TOKEN, auth_enabled=True, port=12345)
    app.state.jobs.shutdown()
    app.state.jobs = JobManager(paths, lambda: app.state.settings, transcriber=fake_transcriber)

    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        response = client.post(
            "/api/v1/jobs",
            headers=headers,
            files={"file": ("meeting.wav", b"RIFF....WAVEfmt ", "audio/wav")},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]

        detail = None
        for _ in range(100):
            detail = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
            if detail["status"] in {"done", "error"}:
                break
            time.sleep(0.02)
        assert detail is not None
        assert detail["status"] == "done", detail
        assert "тестовая" in detail["text"]

        result = client.get(f"/api/v1/jobs/{job_id}/result", headers=headers)
        assert result.status_code == 200
        assert "тестовая" in result.text
        assert "attachment" in result.headers["content-disposition"]

        text = client.get(f"/api/v1/jobs/{job_id}/text", headers=headers)
        assert text.text == detail["text"]
