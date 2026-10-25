"""Model registry manifest.

Describes models available for download (LLM GGUF, ASR, embeddings, VAD).
Each entry: id, kind, title, filename, url, size_bytes, sha256, license, note.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "manifest.json"


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
