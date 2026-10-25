"""Models API: list installed models, download with progress, offline import."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from airec.core.paths import app_root
from airec.models.manager import ModelError, ModelManager
from airec.models.manifest import find_model, load_manifest

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["models"])

_DOWNLOAD_PROGRESS: dict[str, dict[str, Any]] = {}

MANIFEST_DIR = app_root() / "models"


def _manager(request: Request) -> ModelManager:
    return request.app.state.models


@router.get("")
def list_models(request: Request) -> dict[str, Any]:
    """List available manifest entries and installed models."""
    manifest = load_manifest(MANIFEST_DIR)
    manager = _manager(request)
    installed = {m["id"]: m for m in manager.installed_models()}
    items: list[dict[str, Any]] = []
    for entry in manifest.get("models", []):
        model_id = entry.get("id", "")
        info = dict(entry)
        inst = installed.get(model_id)
        info["installed"] = inst is not None
        info["size_mb"] = round((inst["size_bytes"] if inst else entry.get("size_bytes", 0)) / (1024 * 1024), 1)
        info["download_progress"] = _DOWNLOAD_PROGRESS.get(model_id)
        items.append(info)
    return {"items": items, "installed": [m["id"] for m in manager.installed_models()]}


@router.post("/download/{model_id}")
async def download_model_by_id(
    request: Request, model_id: str
) -> dict[str, Any]:
    """Start downloading *model_id* (manifest entry) or custom {id, url, sha256}."""
    manager = _manager(request)
    manifest = load_manifest(MANIFEST_DIR)

    entry = find_model(manifest, model_id)
    if entry is None:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="укажите тело запроса {id, url, sha256}"
            ) from exc
        model_id = body.get("id", model_id)
        url = body.get("url")
        sha256 = body.get("sha256")
        if not url:
            raise HTTPException(status_code=400, detail="требуется поле url")
    else:
        url = entry.get("url", "")
        sha256 = entry.get("sha256") or None
        if not url:
            raise HTTPException(status_code=400, detail="модель без URL скачивания")

    if manager.is_installed(model_id):
        raise HTTPException(status_code=409, detail="модель уже установлена")

    def progress(value: float, phase: str) -> None:
        _DOWNLOAD_PROGRESS[model_id] = {"value": value, "phase": phase}

    try:
        manager.download(model_id, url, sha256=sha256, progress=progress)
    except ModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"id": model_id, "status": "downloading", "url": url}


@router.get("/download/{model_id}/progress")
def download_progress(request: Request, model_id: str) -> dict[str, Any]:
    """Return download progress for *model_id*."""
    progress = _DOWNLOAD_PROGRESS.get(model_id)
    if progress is None:
        installed = [m["id"] for m in _manager(request).installed_models()]
        return {
            "id": model_id,
            "installed": model_id in installed,
            "phase": "готово" if model_id in installed else "неизвестно",
        }
    return {"id": model_id, **progress}


@router.post("/import")
def import_model(request: Request) -> dict[str, Any]:
    """Import a model from a local path (body: {path, id})."""
    manager = _manager(request)
    try:
        body = request.json() if hasattr(request, "json") else {}
    except Exception:
        body = {}
    source = body.get("path")
    model_id = body.get("id")
    if not source or not model_id:
        raise HTTPException(status_code=400, detail="требуются поля path и id")
    try:
        manager.import_model(Path(source), model_id)
    except ModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": model_id, "status": "installed"}
