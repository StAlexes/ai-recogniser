"""LLM endpoints: summarise and ask over transcription jobs."""

from __future__ import annotations

import logging
from typing import Any

from dataclasses import asdict
from fastapi import APIRouter, HTTPException, Request

from airec.asr.manager import JobManager
from airec.core.config import Settings
from airec.llm.glossary import Glossary
from airec.llm.providers import LLMError, create_provider
from airec.llm.summary import Summarizer

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/llm", tags=["llm"])

# Provider is created lazily per request (llama-server child process is
# spawned on first use).  Kept on app.state so it persists across requests.
PROVIDER_STATE_KEY = "llm_provider"


def _manager(request: Request) -> JobManager:
    return request.app.state.jobs


def _provider(request: Request, settings: Settings) -> Any:
    provider = getattr(request.app.state, PROVIDER_STATE_KEY, None)
    if provider is None:
        kind = settings.llm_provider
        if settings.external_llm_enabled and kind == "llamacpp-local":
            kind = "external"
        provider = create_provider(
            kind=kind,
            model=settings.llm_model,
            base_url=settings.external_llm_base_url,
            api_key=settings.external_llm_api_key or None,
            ctx_size=settings.llm_ctx_size,
            threads=settings.llm_threads,
        )
        setattr(request.app.state, PROVIDER_STATE_KEY, provider)
    return provider


@router.get("/providers")
def list_providers(request: Request, settings: Settings = None) -> dict[str, Any]:  # type: ignore[assignment]
    """Available LLM providers and their health."""
    s = request.app.state.settings
    kind = "external" if s.external_llm_enabled else "llamacpp-local"
    provider = create_provider(
        kind=kind,
        model=s.llm_model,
        base_url=s.rag_url,
    )
    return {"active": kind, "health": provider.health()}


@router.post("/summarize/{job_id}")
def summarize_job(request: Request, job_id: str) -> dict[str, Any]:
    """Produce structured summary for a finished transcription job."""
    manager = _manager(request)
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    if job.status != "done":
        raise HTTPException(status_code=409, detail="Задача ещё не завершена.")
    if not job.text:
        raise HTTPException(status_code=409, detail="Транскрипт пуст.")

    settings = request.app.state.settings
    try:
        provider = _provider(request, settings)
        summarizer = Summarizer(provider)
        glossary = Glossary.from_dicts([])  # TODO: load persisted glossary
        text = glossary.apply(job.text)
        summary = summarizer.summarize(text)
        # Persist summary on the job.
        job.summary = summary  # type: ignore[attr-defined]
        manager._db.upsert_job(asdict(job))  # type: ignore[attr-defined]
        manager._write_job_json(job)
        return {"job_id": job_id, "summary": summary}
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/ask/{job_id}")
def ask_job(request: Request, job_id: str, question: str) -> dict[str, Any]:
    """Answer a question about a finished transcription job."""
    manager = _manager(request)
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    if job.status != "done":
        raise HTTPException(status_code=409, detail="Задача ещё не завершена.")
    if not job.text:
        raise HTTPException(status_code=409, detail="Транскрипт пуст.")

    settings = request.app.state.settings
    try:
        provider = _provider(request, settings)
        summarizer = Summarizer(provider)
        answer = summarizer.ask(question, context=job.text)
        return {"job_id": job_id, "question": question, "answer": answer}
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/glossary")
def update_glossary(request: Request, entries: list[dict[str, str]]) -> dict[str, Any]:
    """Replace the session glossary with *entries* ([{heard, correct, note}])."""
    glossary = Glossary.from_dicts(entries)
    request.app.state.llm_glossary = glossary
    return {"entries": glossary.as_dicts(), "count": len(glossary)}
