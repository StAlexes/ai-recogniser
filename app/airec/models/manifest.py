"""Model registry manifest.

Describes models available for download (LLM GGUF, ASR, diarization, VAD).
Each entry: id, kind, title, filename, url, size_bytes, sha256, license, note,
purpose.  Diarization entries additionally carry a Hugging Face ``repo`` id used
by onnx-asr when the model has not been downloaded through the catalogue yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from airec.core.paths import app_root, resource_root

MANIFEST_FILENAME = "manifest.json"


def manifest_dir() -> Path:
    """Return the directory containing ``manifest.json``.

    In development it lives at the repo root; in a PyInstaller build the models
    are bundled under ``_internal/models``.  A copied build may also keep them
    next to the executable, so all known locations are probed.
    """
    candidates = (
        resource_root() / "models",
        app_root() / "_internal" / "models",
        app_root() / "models",
    )
    for candidate in candidates:
        if (candidate / MANIFEST_FILENAME).is_file():
            return candidate
    return candidates[0]


def load_manifest(path: Path) -> dict[str, Any]:
    """Load model manifest from *path* (a directory containing manifest.json)."""
    manifest_path = path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return {"models": []}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read model manifest {manifest_path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {"models": []}


def find_model(manifest: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    """Return model entry by id or None."""
    for model in manifest.get("models", []):
        if model.get("id") == model_id:
            return model
    return None


def models_of_kind(manifest: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """Return all manifest entries of *kind* (e.g. ``diarization``)."""
    return [model for model in manifest.get("models", []) if model.get("kind") == kind]
