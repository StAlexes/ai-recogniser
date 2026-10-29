from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
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
    assert finished.stages["recognition"]["status"] == "done"
    assert finished.stages["recognition"]["progress"] == 1.0
    assert finished.stages["diarization"]["status"] == "skipped"
    assert finished.stages["summary"]["status"] == "pending"
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
    assert restored_job.stages["recognition"]["status"] == "done"
    assert restored.result_path(job.id) is not None
    restored.shutdown()


def test_manager_starts_paused_job_manually(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)
    settings = Settings(recognize_immediately=False)
    manager = JobManager(paths, lambda: settings, transcriber=fake_transcriber)
    upload = tmp_path / "meeting.wav"
    upload.write_bytes(b"RIFF....WAVEfmt ")

    job = manager.submit(upload, "meeting.wav", "gigaam-v3-e2e-ctc")
    assert job.status == "queued"
    assert job.paused is True

    started = manager.start(job.id)
    assert started.paused is False
    finished = wait_done(manager, job.id)
    assert finished.status == "done"
    manager.shutdown()


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
    assert retried.id == job2.id
    assert retried.filename == "meeting2.wav"
    assert retried.status in {"queued", "running"}

    retried_done = wait_done(manager, retried.id)
    assert retried_done.status == "done"
    assert retried_done.text == "Привет, это тестовая расшифровка."
    manager.shutdown()


def test_manager_auto_summary(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)

    def summary_runner(
        job_id: str, progress: Callable[[float, str], None]
    ) -> dict[str, object]:
        assert job_id
        progress(0.5, "Саммаризация")
        return {"tl_dr": "Короткое содержание.", "topics": ["тест"]}

    settings = Settings(summarize_enabled=True)
    manager = JobManager(
        paths,
        lambda: settings,
        transcriber=fake_transcriber,
        summary_runner=summary_runner,
    )
    upload = tmp_path / "meeting.wav"
    upload.write_bytes(b"RIFF....WAVEfmt ")

    job = manager.submit(upload, "meeting.wav", "gigaam-v3-e2e-ctc")
    deadline = time.monotonic() + 5.0
    finished = manager.get(job.id)
    while time.monotonic() < deadline:
        finished = manager.get(job.id)
        assert finished is not None
        if finished.stages["summary"]["status"] in {"done", "error"}:
            break
        time.sleep(0.02)
    assert finished is not None
    assert finished.status == "done"
    assert finished.stages["summary"]["status"] == "done", finished.stages["summary"]
    assert finished.summary == {"tl_dr": "Короткое содержание.", "topics": ["тест"]}
    assert finished.text == "Привет, это тестовая расшифровка."
    manager.shutdown()


def test_manager_auto_summary_without_llm_keeps_transcript(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)
    settings = Settings(summarize_enabled=True)
    manager = JobManager(paths, lambda: settings, transcriber=fake_transcriber)
    upload = tmp_path / "meeting.wav"
    upload.write_bytes(b"RIFF....WAVEfmt ")

    job = manager.submit(upload, "meeting.wav", "gigaam-v3-e2e-ctc")
    finished = wait_done(manager, job.id)
    assert finished.status == "done"
    assert finished.stages["summary"]["status"] == "error"
    assert "LLM" in finished.stages["summary"]["phase"]
    assert finished.text == "Привет, это тестовая расшифровка."
    manager.shutdown()


def test_interrupted_summary_is_recovered(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)
    manager = JobManager(paths, lambda: Settings(), transcriber=fake_transcriber)
    upload = tmp_path / "meeting.wav"
    upload.write_bytes(b"RIFF....WAVEfmt ")

    job = manager.submit(upload, "meeting.wav", "gigaam-v3-e2e-ctc")
    finished = wait_done(manager, job.id)
    finished.stages["summary"].update(
        status="running", phase="Саммаризация", progress=0.5
    )
    manager._persist(finished)
    manager.shutdown()

    restored_manager = JobManager(
        paths, lambda: Settings(), transcriber=fake_transcriber
    )
    restored = restored_manager.get(job.id)
    assert restored is not None
    assert restored.status == "done"
    assert restored.stages["summary"]["status"] == "error"
    assert "прервана" in restored.stages["summary"]["phase"]
    restored_manager.shutdown()


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

        target = tmp_path / "saved" / "meeting.srt"
        saved = client.post(
            f"/api/v1/jobs/{job_id}/save",
            headers=headers,
            json={"path": str(target)},
        )
        assert saved.status_code == 200, saved.text
        assert target.is_file()
        assert "-->" in target.read_text(encoding="utf-8")
        assert saved.json()["name"] == "meeting.srt"


def wait_stage(
    manager: JobManager,
    job_id: str,
    stage: str,
    statuses: set[str],
    timeout: float = 5.0,
) -> Job:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        assert job is not None
        if job.stages[stage]["status"] in statuses:
            return job
        time.sleep(0.02)
    raise AssertionError(f"stage {stage} did not reach {statuses}")


def test_stage_controls_restart_diarization_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = paths_for(tmp_path)
    diarize_calls: list[int] = []
    summary_calls: list[str] = []
    summary_started = threading.Event()
    release_summary = threading.Event()

    def fake_diarize(source: Path, segments: list[dict[str, object]], **kwargs: object):
        assert source.is_file()
        diarize_calls.append(len(segments))
        stage_progress = kwargs["stage_progress"]
        assert callable(stage_progress)
        stage_progress("diarization", 0.5, "Спикеры")
        return tuple(
            SpeechSegment(
                float(segment["start_s"]),
                float(segment["end_s"]),
                str(segment["text"]),
                speaker="S1",
            )
            for segment in segments
        )

    def summary_runner(
        job_id: str, progress: Callable[[float, str], None]
    ) -> dict[str, object]:
        summary_calls.append(job_id)
        progress(0.4, "Саммаризация")
        if len(summary_calls) == 1:
            summary_started.set()
            assert release_summary.wait(5)
        return {"tl_dr": "Готовое саммари."}

    monkeypatch.setattr("airec.asr.manager.diarize_transcription", fake_diarize)
    settings = Settings(diarize_enabled=False, summarize_enabled=False)
    manager = JobManager(
        paths,
        lambda: settings,
        transcriber=fake_transcriber,
        summary_runner=summary_runner,
    )
    upload = tmp_path / "meeting.wav"
    upload.write_bytes(b"RIFF....WAVEfmt ")

    job = manager.submit(upload, "meeting.wav", "gigaam-v3-e2e-ctc")
    wait_done(manager, job.id)

    manager.start_stage(job.id, "diarization")
    diarized = wait_stage(manager, job.id, "diarization", {"done", "error"})
    assert diarized.stages["diarization"]["status"] == "done"
    assert diarized.segments[0]["speaker"] == "S1"
    assert diarize_calls == [2]

    manager.restart_stage(job.id, "diarization")
    wait_stage(manager, job.id, "diarization", {"done", "error"})
    assert diarize_calls == [2, 2]
    assert manager.get(job.id).id == job.id  # type: ignore[union-attr]

    manager.start_stage(job.id, "summary")
    assert summary_started.wait(2)
    manager.stop_stage(job.id, "summary")
    stopped = manager.get(job.id)
    assert stopped is not None
    assert stopped.stages["summary"]["status"] == "skipped"
    release_summary.set()
    time.sleep(0.15)
    stopped = manager.get(job.id)
    assert stopped is not None
    assert stopped.summary is None

    manager.restart_stage(job.id, "summary")
    finished = wait_stage(manager, job.id, "summary", {"done", "error"})
    assert finished.stages["summary"]["status"] == "done"
    assert finished.summary == {"tl_dr": "Готовое саммари."}
    assert summary_calls == [job.id, job.id]
    manager.shutdown()


def test_auto_diarization_runs_before_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = paths_for(tmp_path)
    summary_speakers: list[str | None] = []

    def fake_diarize(source: Path, segments: list[dict[str, object]], **kwargs: object):
        stage_progress = kwargs["stage_progress"]
        assert callable(stage_progress)
        stage_progress("diarization", 0.5, "Спикеры")
        return tuple(
            SpeechSegment(
                float(segment["start_s"]),
                float(segment["end_s"]),
                str(segment["text"]),
                speaker="S1",
            )
            for segment in segments
        )

    def summary_runner(
        job_id: str, progress: Callable[[float, str], None]
    ) -> dict[str, object]:
        progress(1.0, "Готово")
        job = manager.get(job_id)
        assert job is not None
        summary_speakers.append(job.segments[0].get("speaker"))
        return {"tl_dr": "Саммари после спикеров."}

    monkeypatch.setattr("airec.asr.manager.diarize_transcription", fake_diarize)
    settings = Settings(diarize_enabled=True, summarize_enabled=True)
    manager = JobManager(
        paths,
        lambda: settings,
        transcriber=fake_transcriber,
        summary_runner=summary_runner,
    )
    upload = tmp_path / "meeting.wav"
    upload.write_bytes(b"RIFF....WAVEfmt ")

    job = manager.submit(upload, "meeting.wav", "gigaam-v3-e2e-ctc")
    finished = wait_stage(manager, job.id, "summary", {"done", "error"})
    assert finished.stages["diarization"]["status"] == "done"
    assert finished.stages["summary"]["status"] == "done"
    assert summary_speakers == ["S1"]
    manager.shutdown()


def test_recognition_stage_restart_keeps_job_id(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)
    manager = JobManager(paths, lambda: Settings(), transcriber=fake_transcriber)
    upload = tmp_path / "meeting.wav"
    upload.write_bytes(b"RIFF....WAVEfmt ")

    job = manager.submit(upload, "meeting.wav", "gigaam-v3-e2e-ctc")
    wait_done(manager, job.id)

    restarted = manager.restart_stage(job.id, "recognition")
    assert restarted.id == job.id
    assert restarted.status in {"queued", "running"}
    finished = wait_done(manager, job.id)
    assert finished.status == "done"
    assert finished.text == "Привет, это тестовая расшифровка."
    manager.shutdown()


def test_stage_action_endpoint(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)
    settings = Settings()

    def summary_runner(
        job_id: str, progress: Callable[[float, str], None]
    ) -> dict[str, object]:
        progress(1.0, "Готово")
        return {"tl_dr": f"Саммари для {job_id}"}

    app = create_app(
        settings=settings, paths=paths, token=TOKEN, auth_enabled=True, port=12345
    )
    app.state.jobs.shutdown()
    app.state.jobs = JobManager(
        paths,
        lambda: app.state.settings,
        transcriber=fake_transcriber,
        summary_runner=summary_runner,
    )

    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        response = client.post(
            "/api/v1/jobs",
            headers=headers,
            files={"file": ("meeting.wav", b"RIFF....WAVEfmt ", "audio/wav")},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        for _ in range(100):
            detail = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
            if detail["status"] == "done":
                break
            time.sleep(0.02)
        assert detail["status"] == "done", detail

        started = client.post(
            f"/api/v1/jobs/{job_id}/stages/summary/start", headers=headers
        )
        assert started.status_code == 200, started.text
        for _ in range(100):
            detail = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
            if detail["stages"]["summary"]["status"] in {"done", "error"}:
                break
            time.sleep(0.02)
        assert detail["stages"]["summary"]["status"] == "done", detail
        assert detail["summary"]["tl_dr"] == f"Саммари для {job_id}"
