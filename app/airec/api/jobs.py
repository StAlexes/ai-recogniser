"""Upload and job endpoints for local ASR."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from airec.asr.audio import SUPPORTED_EXTENSIONS
from airec.asr.manager import ACTIVE_STATUSES, JobManager, JobValidationError

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

UPLOAD_CHUNK_BYTES = 1024 * 1024


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
