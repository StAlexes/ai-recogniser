"""Upload and job endpoints for local ASR."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from airec.asr.audio import SUPPORTED_EXTENSIONS
from airec.asr.manager import ACTIVE_STATUSES, JobManager, JobValidationError

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

EXPORT_FORMATS = {"txt", "srt", "vtt", "md", "json"}

UPLOAD_CHUNK_BYTES = 1024 * 1024


class SaveResultRequest(BaseModel):
    """Destination selected by the user in the native save dialog."""

    path: str
    format: str | None = None


def _manager(request: Request) -> JobManager:
    return request.app.state.jobs


@router.get("")
def list_jobs(request: Request) -> dict[str, object]:
    items = _manager(request).list()
    return {"items": items, "total": len(items)}


@router.post("", status_code=202)
async def create_job(
    request: Request, file: Annotated[UploadFile, File()]
) -> dict[str, object]:
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Неподдерживаемый формат. Используйте mp3, mp4, m4a, wav, flac, "
            "ogg, opus, mkv, mov, webm или avi.",
        )

    paths = request.app.state.paths
    manager = _manager(request)
    upload_dir = paths.temp / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temporary = upload_dir / f"{uuid.uuid4().hex}{suffix}"

    size = 0
    try:
        with temporary.open("wb") as handle:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > manager.max_input_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Файл больше допустимых 4 ГБ.",
                    )
                handle.write(chunk)
    except HTTPException:
        temporary.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        log.exception("upload failed")
        raise HTTPException(status_code=500, detail="Не удалось сохранить файл.") from exc
    finally:
        await file.close()

    if size == 0:
        temporary.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Файл пуст.")

    settings = request.app.state.settings
    try:
        job = manager.submit(temporary, filename, settings.asr_model)
    except JobValidationError as exc:
        temporary.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        log.exception("cannot queue uploaded file")
        raise HTTPException(status_code=500, detail="Не удалось поставить файл в очередь.") from exc

    log.info("job %s queued: %s (%s bytes)", job.id, filename, size)
    return job.detail()


@router.get("/{job_id}")
def get_job(request: Request, job_id: str) -> dict[str, object]:
    job = _manager(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    return job.detail()


@router.get("/{job_id}/result")
def download_result(request: Request, job_id: str) -> FileResponse:
    manager = _manager(request)
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    path = manager.result_path(job_id)
    if path is None:
        raise HTTPException(status_code=409, detail="Результат ещё не готов.")
    download_name = f"{Path(job.filename).stem or 'transcript'}.txt"
    return FileResponse(
        path,
        media_type="text/plain; charset=utf-8",
        filename=download_name,
    )


@router.get("/{job_id}/text", response_class=PlainTextResponse)
def get_result_text(request: Request, job_id: str) -> PlainTextResponse:
    manager = _manager(request)
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    if job.status in ACTIVE_STATUSES:
        raise HTTPException(status_code=409, detail="Задача ещё выполняется.")
    return PlainTextResponse(job.text, media_type="text/plain; charset=utf-8")


@router.post("/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> dict[str, object]:
    """Отменить задачу (queued или running)."""
    manager = _manager(request)
    try:
        manager.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    job = manager.get(job_id)
    return job.detail() if job else {"id": job_id, "status": "cancelled"}


@router.post("/{job_id}/retry")
def retry_job(request: Request, job_id: str) -> dict[str, object]:
    """Повтор задачи со статусом error или cancelled."""
    manager = _manager(request)
    try:
        new_job = manager.retry(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return new_job.detail()


@router.post("/{job_id}/start")
def start_job(request: Request, job_id: str) -> dict[str, object]:
    """Запустить задачу, добавленную без автозапуска."""
    manager = _manager(request)
    try:
        job = manager.start(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.detail()


@router.post("/{job_id}/reveal")
def reveal_job_folder(request: Request, job_id: str) -> dict[str, object]:
    """Open the job result folder in Explorer (local desktop app only)."""
    manager = _manager(request)
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    directory = manager.result_dir(job_id)
    try:
        if os.name == "nt":
            os.startfile(str(directory))  # type: ignore[attr-defined]
        else:
            raise OSError("открытие папки поддерживается только в Windows")
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Не удалось открыть папку: {exc}"
        ) from exc
    return {"id": job_id, "path": str(directory)}


@router.post("/{job_id}/save")
def save_job_result(
    request: Request, job_id: str, payload: SaveResultRequest
) -> dict[str, object]:
    """Save a finished transcript to a user-selected path."""
    manager = _manager(request)
    try:
        target = manager.save_result(
            job_id, Path(payload.path), format=payload.format
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "id": job_id,
        "path": str(target),
        "name": target.name,
        "size_bytes": target.stat().st_size,
    }


@router.get("/{job_id}/export")
def export_job(
    request: Request, job_id: str, format: str = "txt"
) -> FileResponse:
    """Export a finished job transcript in the requested format.

    Supported formats: txt, srt, vtt, md, json.  Defaults to txt.
    """
    if format not in EXPORT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный формат экспорта. Доступно: {', '.join(sorted(EXPORT_FORMATS))}.",
        )

    manager = _manager(request)
    try:
        path, download_name, content_type = manager.export_result(job_id, format)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return FileResponse(
        path,
        media_type=content_type,
        filename=download_name,
    )
