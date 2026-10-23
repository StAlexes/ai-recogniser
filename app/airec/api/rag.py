"""RAG integration endpoints.

The heavy lifting lives in :mod:`airec.rag.client`; this module only reads the
current settings and maps failures to useful HTTP responses.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from airec.core.config import Settings
from airec.rag.client import RagError, RagKbClient, disabled_status

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

MAX_QUESTION_CHARS = 2000
MAX_QUERY_CHARS = 1000


class RagAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    k: int | None = Field(default=None, ge=1, le=20)


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _client(settings: Settings) -> RagKbClient:
    return RagKbClient(settings.rag_url)


def _ensure_enabled(settings: Settings) -> None:
    if not settings.rag_enabled:
        raise HTTPException(
            status_code=409,
            detail="Интеграция с RAG выключена. Включите её в настройках.",
        )


@router.get("/status")
def rag_status(request: Request) -> dict[str, object]:
    settings = _settings(request)
    if not settings.rag_enabled:
        return disabled_status(settings)
    try:
        return _client(settings).status()
    except RagError as exc:
        # An invalid URL is a configuration error, but the UI should still be
        # able to render the page and show the reason.
        return {
            "enabled": True,
            "configured_url": settings.rag_url,
            "reachable": False,
            "latency_ms": None,
            "chunks": None,
            "indexing": None,
            "gen_model": None,
            "error": str(exc),
        }


@router.get("/search")
def rag_search(
    request: Request,
    q: str = Query(min_length=1, max_length=MAX_QUERY_CHARS),
    k: int | None = Query(default=None, ge=1, le=20),
) -> dict[str, object]:
    settings = _settings(request)
    _ensure_enabled(settings)
    try:
        return _client(settings).search(q, k or settings.rag_k)
    except RagError as exc:
        log.warning("RAG search failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/ask")
def rag_ask(request: Request, payload: RagAskRequest) -> dict[str, object]:
    settings = _settings(request)
    _ensure_enabled(settings)
    try:
        return _client(settings).ask(payload.question, payload.k or settings.rag_k)
    except RagError as exc:
        log.warning("RAG ask failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
