"""Models API: list installed models, download with progress, offline import,
and Ollama models detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from airec.core.paths import app_root, resource_root
from airec.models.manager import ModelError, ModelManager
from airec.models.manifest import find_model, load_manifest

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["models"])

_DOWNLOAD_PROGRESS: dict[str, dict[str, Any]] = {}

# Manifest lives next to the repo root in development and inside the
# PyInstaller bundle (_internal/models) in a frozen build.  Probe the known
# locations so a moved/copied build keeps working.
def _manifest_dir() -> Path:
    candidates = (
        resource_root() / "models",
        app_root() / "_internal" / "models",
        app_root() / "models",
    )
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    return candidates[0]


MANIFEST_DIR = _manifest_dir()

OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def _manager(request: Request) -> ModelManager:
    return request.app.state.models


def _detect_ollama_models() -> list[dict[str, Any]]:
    """Try to detect models installed in a local Ollama instance."""
    try:
        with httpx.Client(timeout=httpx.Timeout(2.0, connect=1.0)) as client:
            resp = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = []
                for m in data.get("models", []):
                    models.append(
                        {
                            "id": f"ollama:{m.get('name', 'unknown')}",
                            "kind": "ollama",
                            "title": m.get("name", "unknown"),
                            "filename": None,
                            "url": None,
                            "size_bytes": m.get("size", 0),
                            "sha256": m.get("digest", ""),
                            "license": "unknown",
                            "note": "Модель Ollama (уже установлена на компьютере)",
                            "purpose": "саммаризация, вопросы",
                            "installed": True,
                            "download_progress": None,
                        }
                    )
                return models
    except Exception:
        # Ollama not running or not installed — silently ignore
        pass
    return []


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

    # Add Ollama models (already installed on the computer)
    ollama_models = _detect_ollama_models()
    items.extend(ollama_models)

    return {
        "items": items,
        "installed": [m["id"] for m in manager.installed_models()]
        + [m["id"] for m in ollama_models],
    }


@router.post("/download/{model_id}")
async def download_model_by_id(
    request: Request, model_id: str, background: BackgroundTasks
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

    def run_download() -> None:
        try:
            manager.download(model_id, url, sha256=sha256, progress=progress)
        except Exception as exc:  # background task: report instead of crashing
            _DOWNLOAD_PROGRESS[model_id] = {
                "value": 0.0,
                "phase": f"ошибка: {exc}",
            }
            log.exception("model download failed: %s", model_id)

    _DOWNLOAD_PROGRESS[model_id] = {"value": 0.0, "phase": "подготовка"}
    background.add_task(run_download)
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
async def import_model(request: Request) -> dict[str, Any]:
    """Import a model from a local path (body: {path, id})."""
    manager = _manager(request)
    try:
        body = await request.json()
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
