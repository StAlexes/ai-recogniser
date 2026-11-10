"""Режим проверки спикеров: отчёт diarization_review.json и эндпоинты."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from airec.api.app import create_app
from airec.asr.manager import REVIEW_FILE_NAME, Job
from airec.core.config import Settings
from airec.core.paths import AppPaths

TOKEN = "test-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


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


def make_done_job(tmp_path: Path, paths: AppPaths) -> Job:
    """Завершённая задача с двумя спикерами; исходник лежит рядом."""
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"RIFF....WAVEfmt ")
    job = Job(
        id="20260101-000000-review1",
        filename="meeting.wav",
        status="done",
        text="Фрагмент один Фрагмент два Фрагмент три",
        original_path=str(source),
    )
    job.segments = [
        {"start_s": 0.0, "end_s": 6.0, "text": "Фрагмент один", "speaker": "Спикер 1"},
        {"start_s": 6.0, "end_s": 12.0, "text": "Фрагмент два", "speaker": "Спикер 2"},
        {"start_s": 12.0, "end_s": 18.0, "text": "Фрагмент три", "speaker": "Спикер 1"},
        # Фрагмент без текста в отчёт не попадает (его не видно в UI).
        {"start_s": 18.0, "end_s": 20.0, "text": "  ", "speaker": "Спикер 1"},
    ]
    directory = paths.results / job.id
    directory.mkdir(parents=True)
    (directory / "job.json").write_text(
        json.dumps(asdict(job), ensure_ascii=False),
        encoding="utf-8",
    )
    return job


@pytest.fixture()
def client(tmp_path: Path) -> tuple[TestClient, Path]:
    paths = paths_for(tmp_path)
    make_done_job(tmp_path, paths)
    app = create_app(
        settings=Settings(), paths=paths, token=TOKEN, auth_enabled=True, port=12345
    )
    with TestClient(app) as test_client:
        yield test_client, tmp_path


def test_review_404_before_first_save(client: tuple[TestClient, Path]) -> None:
    test_client, _tmp_path = client
    job_id = "20260101-000000-review1"
    assert (
        test_client.get(f"/api/v1/jobs/{job_id}/review", headers=HEADERS).status_code
        == 404
    )


def test_review_roundtrip(client: tuple[TestClient, Path]) -> None:
    test_client, tmp_path = client
    job_id = "20260101-000000-review1"
    url = f"/api/v1/jobs/{job_id}/review"

    saved = test_client.put(
        url,
        headers=HEADERS,
        json={
            "corrections": [
                {"index": 1, "correct": "Спикер 1", "comment": "это тот же голос"}
            ]
        },
    )
    assert saved.status_code == 200, saved.text
    report = saved.json()

    # Исправление заполнено данными задачи: сервер, а не UI, знает метку.
    assert report["corrections"] == [
        {
            "index": 1,
            "start_s": 6.0,
            "end_s": 12.0,
            "text": "Фрагмент два",
            "assigned": "Спикер 2",
            "correct": "Спикер 1",
            "comment": "это тот же голос",
        }
    ]
    assert report["version"] == 1
    assert report["job"]["filename"] == "meeting.wav"
    # Фрагмент без текста не входит в исходную разметку.
    assert len(report["original_labels"]) == 3
    assert report["stats"]["fragments"] == 3
    assert report["stats"]["corrected"] == 1
    assert report["stats"]["speaker_counts_original"] == {"Спикер 1": 2, "Спикер 2": 1}
    assert report["stats"]["speaker_counts_corrected"] == {"Спикер 1": 3}
    # Снимок настроек: активный движок и его группа гиперпараметров.
    assert report["settings"]["engine"] == "pyannote"
    assert report["settings"]["group"] is not None
    assert report["settings"]["group"]["id"] == "meeting"

    assert (tmp_path / "results" / job_id / REVIEW_FILE_NAME).is_file()

    loaded = test_client.get(url, headers=HEADERS)
    assert loaded.status_code == 200
    assert loaded.json() == report


def test_review_created_at_stable_across_saves(
    client: tuple[TestClient, Path],
) -> None:
    test_client, _tmp_path = client
    url = "/api/v1/jobs/20260101-000000-review1/review"
    first = test_client.put(url, headers=HEADERS, json={"corrections": []}).json()
    second = test_client.put(
        url,
        headers=HEADERS,
        json={"corrections": [{"index": 0, "correct": "Спикер 2"}]},
    ).json()
    assert first["created_at"] == second["created_at"]
    assert second["corrections"][0]["assigned"] == "Спикер 1"


def test_review_validates_corrections(client: tuple[TestClient, Path]) -> None:
    test_client, _tmp_path = client
    url = "/api/v1/jobs/20260101-000000-review1/review"

    # Номер вне списка фрагментов (в списке три фрагмента).
    assert (
        test_client.put(
            url, headers=HEADERS, json={"corrections": [{"index": 9, "correct": "Спикер 1"}]}
        ).status_code
        == 400
    )
    # Спикер, которого нет в разметке, и не следующий новый номер.
    assert (
        test_client.put(
            url, headers=HEADERS, json={"corrections": [{"index": 0, "correct": "Спикер 7"}]}
        ).status_code
        == 400
    )
    # Дубликат фрагмента.
    assert (
        test_client.put(
            url,
            headers=HEADERS,
            json={
                "corrections": [
                    {"index": 0, "correct": "Спикер 2"},
                    {"index": 0, "correct": "Спикер 1"},
                ]
            },
        ).status_code
        == 400
    )
    # Пустой правильный спикер отсекается валидацией схемы.
    assert (
        test_client.put(
            url, headers=HEADERS, json={"corrections": [{"index": 0, "correct": ""}]}
        ).status_code
        == 422
    )
    # Новый спикер — только следующий по номеру.
    ok = test_client.put(
        url, headers=HEADERS, json={"corrections": [{"index": 2, "correct": "Спикер 3"}]}
    )
    assert ok.status_code == 200
    assert ok.json()["stats"]["speaker_counts_corrected"] == {
        "Спикер 1": 1,
        "Спикер 2": 1,
        "Спикер 3": 1,
    }
    # Введённая метка действительна и для следующих исправлений.
    ok = test_client.put(
        url,
        headers=HEADERS,
        json={
            "corrections": [
                {"index": 2, "correct": "Спикер 3"},
                {"index": 1, "correct": "Спикер 3"},
            ]
        },
    )
    assert ok.status_code == 200
    assert ok.json()["stats"]["speaker_counts_corrected"] == {
        "Спикер 1": 1,
        "Спикер 3": 2,
    }


def test_review_unknown_job_is_404(client: tuple[TestClient, Path]) -> None:
    test_client, _tmp_path = client
    assert (
        test_client.put(
            "/api/v1/jobs/unknown/review", headers=HEADERS, json={"corrections": []}
        ).status_code
        == 404
    )


def test_review_export_via_save_and_export(
    client: tuple[TestClient, Path],
) -> None:
    test_client, tmp_path = client
    job_id = "20260101-000000-review1"
    test_client.put(
        f"/api/v1/jobs/{job_id}/review",
        headers=HEADERS,
        json={"corrections": [{"index": 1, "correct": "Спикер 1"}]},
    )

    target = tmp_path / "out" / "review.json"
    saved = test_client.post(
        f"/api/v1/jobs/{job_id}/save",
        headers=HEADERS,
        json={"path": str(target), "view": "review"},
    )
    assert saved.status_code == 200, saved.text
    exported = json.loads(target.read_text(encoding="utf-8"))
    assert exported["stats"]["corrected"] == 1

    response = test_client.get(
        f"/api/v1/jobs/{job_id}/export", params={"view": "review", "format": "json"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert json.loads(response.text)["stats"]["corrected"] == 1


def test_audio_endpoint_serves_source(
    client: tuple[TestClient, Path],
) -> None:
    test_client, tmp_path = client
    job_id = "20260101-000000-review1"
    response = test_client.get(f"/api/v1/jobs/{job_id}/audio", headers=HEADERS)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")

    (tmp_path / "meeting.wav").unlink()
    missing = test_client.get(f"/api/v1/jobs/{job_id}/audio", headers=HEADERS)
    assert missing.status_code == 409

    unknown = test_client.get("/api/v1/jobs/unknown/audio", headers=HEADERS)
    assert unknown.status_code == 404


def test_review_requires_finished_job(tmp_path: Path) -> None:
    """Для выполняющейся задачи отчёт недоступен — разметка ещё меняется."""
    paths = paths_for(tmp_path)
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"RIFF....WAVEfmt ")
    job = Job(
        id="20260101-000000-review2",
        filename="meeting.wav",
        status="running",
        original_path=str(source),
    )
    job.segments = [
        {"start_s": 0.0, "end_s": 6.0, "text": "Фрагмент один", "speaker": "Спикер 1"}
    ]
    directory = paths.results / job.id
    directory.mkdir(parents=True)
    (directory / "job.json").write_text(
        json.dumps(asdict(job), ensure_ascii=False), encoding="utf-8"
    )
    app = create_app(
        settings=Settings(), paths=paths, token=TOKEN, auth_enabled=True, port=12345
    )
    with TestClient(app) as test_client:
        assert (
            test_client.get(
                f"/api/v1/jobs/{job.id}/review", headers=HEADERS
            ).status_code
            == 409
        )
        assert (
            test_client.put(
                f"/api/v1/jobs/{job.id}/review",
                headers=HEADERS,
                json={"corrections": []},
            ).status_code
            == 409
        )
        # Аудио доступно и до завершения — по нему можно проверять.
        assert (
            test_client.get(f"/api/v1/jobs/{job.id}/audio", headers=HEADERS).status_code
            == 200
        )
