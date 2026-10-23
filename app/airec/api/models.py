"""Model catalog endpoint (static metadata until the registry lands in M1/M3)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/models", tags=["models"])

CATALOG: list[dict[str, object]] = [
    {
        "id": "gigaam-v3-e2e-ctc",
        "kind": "asr",
        "title": "GigaAM v3 e2e CTC",
        "size_mb": 214,
        "license": "MIT",
        "languages": ["ru"],
        "default": True,
        "description": "Русский, пунктуация и нормализация, CPU/DirectML",
    },
    {
        "id": "gigaam-multilingual-ctc",
        "kind": "asr",
        "title": "GigaAM Multilingual CTC",
        "size_mb": 214,
        "license": "MIT",
        "languages": ["ru", "en", "kk", "ky", "uz"],
        "default": False,
        "description": "5 языков, без пунктуации (char CTC)",
    },
    {
        "id": "t-lite-it-2.1-8b-q4",
        "kind": "llm",
        "title": "T-lite-it-2.1 8B (Q4)",
        "size_mb": 5000,
        "license": "check before release",
        "languages": ["ru", "en"],
        "default": False,
        "description": "Локальные саммари и Q&A, 16+ ГБ RAM",
    },
    {
        "id": "qwen3-4b-instruct-q4",
        "kind": "llm",
        "title": "Qwen3 4B Instruct (Q4)",
        "size_mb": 2600,
        "license": "Apache-2.0",
        "languages": ["ru", "en"],
        "default": False,
        "description": "Лёгкий вариант для 8 ГБ RAM",
    },
]


@router.get("")
def list_models() -> dict[str, object]:
    return {"items": CATALOG, "installed": []}
