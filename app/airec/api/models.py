"""Models API: list installed models, download with progress, offline import,
and Ollama models detection."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from airec.llm import runtime
from airec.models.manager import ModelError, ModelManager
from airec.models.manifest import find_model, load_manifest, manifest_dir

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["models"])

_DOWNLOAD_PROGRESS: dict[str, dict[str, Any]] = {}

MANIFEST_DIR = manifest_dir()

OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def _manager(request: Request) -> ModelManager:
    return request.app.state.models


def _hf_cached(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Return a fake installed entry when the ONNX model is in the HF cache.

    Diarization/ASR models are downloaded by onnx-asr into the Hugging Face
    cache rather than ``data/models``; without this check the catalogue would
    report an already usable model as missing.
    """
    repo = entry.get("repo")
    filename = entry.get("filename")
    if not repo or not filename:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo, filename)
    except Exception:  # huggingface_hub is optional at runtime
        return None
    if not isinstance(cached, str):
        return None
    try:
        size = Path(cached).stat().st_size
    except OSError:
        size = 0
    return {
        "id": entry.get("id", ""),
        "path": str(Path(cached).parent),
        "size_bytes": size,
        "installed": True,
    }


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
    paths = request.app.state.paths
    disk_installed = manager.installed_models()
    installed = {m["id"]: {**m, "location": "data"} for m in disk_installed}
    items: list[dict[str, Any]] = []
    for entry in manifest.get("models", []):
        model_id = entry.get("id", "")
        info = dict(entry)
        if entry.get("kind") == "runtime":
            server = runtime.find_llama_server(paths)
            info["installed"] = server is not None
            info["local"] = server is not None
            info["location"] = "runtime" if server is not None else None
            info["size_mb"] = round(
                float(entry.get("size_bytes", 0)) / (1024 * 1024), 1
            )
            info["download_progress"] = _DOWNLOAD_PROGRESS.get(model_id)
            if server is not None:
                info["installed_path"] = str(server)
            items.append(info)
            continue
        inst = installed.get(model_id)
        if inst is not None and inst.get("location") == "data":
            for extra in entry.get("extra_files") or []:
                extra_name = Path(str(extra.get("filename", ""))).name
                if extra_name and not (manager.model_path(model_id) / extra_name).is_file():
                    inst = None
                    installed.pop(model_id, None)
                    break
        if inst is None:
            cached = _hf_cached(entry)
            if cached is not None:
                inst = {**cached, "location": "huggingface-cache"}
                installed[model_id] = inst
        info["installed"] = inst is not None
        info["local"] = inst is not None and inst.get("location") == "data"
        info["location"] = inst.get("location") if inst else None
        info["size_mb"] = round((inst["size_bytes"] if inst else entry.get("size_bytes", 0)) / (1024 * 1024), 1)
        info["download_progress"] = manager.download_status(model_id) or _DOWNLOAD_PROGRESS.get(
            model_id
        )
        items.append(info)

    # Add Ollama models (already installed on the computer)
    ollama_models = _detect_ollama_models()
    items.extend(ollama_models)

    disk_ids = {m["id"] for m in disk_installed}
    if runtime.runtime_installed(paths):
        disk_ids.add(runtime.RUNTIME_ID)
    return {
        "items": items,
        "installed": sorted(
            disk_ids
            | set(installed)
            | {m["id"] for m in ollama_models}
        ),
    }


@router.post("/download/{model_id}")
async def download_model_by_id(
    request: Request, model_id: str, background: BackgroundTasks
) -> dict[str, Any]:
    """Start downloading *model_id* (manifest entry) or custom {id, url, sha256}."""
    manager = _manager(request)
    manifest = load_manifest(MANIFEST_DIR)

    entry = find_model(manifest, model_id)
    filename: str | None = None
    extra_files: list[dict[str, str]] = []
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
        filename = body.get("filename")
        if not url:
            raise HTTPException(status_code=400, detail="требуется поле url")
    else:
        url = entry.get("url", "")
        sha256 = entry.get("sha256") or None
        filename = entry.get("filename")
        extra_files = entry.get("extra_files") or []
        if not url:
            raise HTTPException(status_code=400, detail="модель без URL скачивания")

    if entry is not None and entry.get("kind") == "runtime":
        paths = request.app.state.paths
        if runtime.find_llama_server(paths) is not None:
            raise HTTPException(status_code=409, detail="llama-server уже установлен")

        def runtime_progress(value: float, phase: str) -> None:
            _DOWNLOAD_PROGRESS[model_id] = {
                "value": value,
                "phase": phase,
                "status": "downloading",
            }

        def run_runtime_download() -> None:
            try:
                runtime.ensure_runtime(
                    paths, progress=runtime_progress, force=True
                )
                _DOWNLOAD_PROGRESS[model_id] = {
                    "value": 1.0,
                    "phase": "готово",
                    "status": "done",
                }
            except Exception as exc:  # background task: report instead of crashing
                _DOWNLOAD_PROGRESS[model_id] = {
                    "value": 0.0,
                    "phase": f"ошибка: {exc}",
                    "status": "error",
                }
                log.exception("llama-server install failed")

        _DOWNLOAD_PROGRESS[model_id] = {
            "value": 0.0,
            "phase": "подготовка",
            "status": "downloading",
        }
        background.add_task(run_runtime_download)
        return {"id": model_id, "status": "downloading", "url": url}

    if manager.is_installed(model_id, filename=filename) and not any(
        not (manager.model_path(model_id) / Path(str(extra.get("filename", ""))).name).is_file()
        for extra in extra_files
        if extra.get("filename")
    ):
        raise HTTPException(status_code=409, detail="модель уже установлена")

    def progress(value: float, phase: str) -> None:
        _DOWNLOAD_PROGRESS[model_id] = {
            "value": value,
            "phase": phase,
            "status": "downloading",
        }

    def run_download() -> None:
        try:
            manager.download(
                model_id,
                url,
                sha256=sha256,
                filename=filename,
                extra_files=extra_files,
                progress=progress,
            )
            _DOWNLOAD_PROGRESS[model_id] = {
                "value": 1.0,
                "phase": "готово",
                "status": "done",
            }
        except Exception as exc:  # background task: report instead of crashing
            _DOWNLOAD_PROGRESS[model_id] = {
                "value": 0.0,
                "phase": f"ошибка: {exc}",
                "status": "error",
            }
            log.exception("model download failed: %s", model_id)

    _DOWNLOAD_PROGRESS[model_id] = {
        "value": 0.0,
        "phase": "подготовка",
        "status": "downloading",
    }
    background.add_task(run_download)
    return {"id": model_id, "status": "downloading", "url": url}


@router.get("/download/{model_id}/progress")
def download_progress(request: Request, model_id: str) -> dict[str, Any]:
    """Return download progress for *model_id*."""
    manager = _manager(request)
    progress = manager.download_status(model_id)
    if progress is None:
        progress = _DOWNLOAD_PROGRESS.get(model_id)
    if progress is None:
        if model_id == runtime.RUNTIME_ID:
            installed_runtime = (
                runtime.find_llama_server(request.app.state.paths) is not None
            )
            return {
                "id": model_id,
                "installed": installed_runtime,
                "phase": "готово" if installed_runtime else "неизвестно",
                "status": "done" if installed_runtime else None,
                "value": 1.0 if installed_runtime else 0.0,
            }
        installed = [m["id"] for m in manager.installed_models()]
        return {
            "id": model_id,
            "installed": model_id in installed,
            "phase": "готово" if model_id in installed else "неизвестно",
        }
    return {"id": model_id, **progress}


@router.post("/reveal")
def reveal_models_folder(request: Request) -> dict[str, Any]:
    """Open the local models folder in Explorer."""
    manager = _manager(request)
    directory = manager.models_dir
    directory.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(str(directory))  # type: ignore[attr-defined]
        else:
            raise OSError("открытие папки поддерживается только в Windows")
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Не удалось открыть папку: {exc}"
        ) from exc
    return {"path": str(directory)}


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
