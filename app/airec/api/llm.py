"""LLM endpoints: summarise and ask over transcription jobs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from airec.asr.manager import JobManager
from airec.core.config import Settings
from airec.llm.glossary import Glossary
from airec.llm.providers import LLMError, create_provider
from airec.llm.runtime import ensure_runtime, find_llama_server
from airec.llm.summary import Summarizer

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/llm", tags=["llm"])

# Provider is created lazily per request (llama-server child process is
# spawned on first use).  Kept on app.state so it persists across requests.
PROVIDER_STATE_KEY = "llm_provider"


def _manager(request: Request) -> JobManager:
    return request.app.state.jobs


def _app_of(target: Any) -> Any:
    """Accept either a FastAPI Request or the application itself."""
    return getattr(target, "app", target)


def _resolve_llm_model(target: Any, settings: Settings) -> str:
    """Resolve a GGUF name from Settings to the real file in ``models/``."""
    models_dir = Path(_app_of(target).state.paths.models)
    value = (settings.llm_model or "").strip()
    if value and value != "auto":
        candidate = Path(value)
        if candidate.is_file():
            return str(candidate)
        if not candidate.is_absolute():
            direct = models_dir / value
            if direct.is_file():
                return str(direct)
            matches = sorted(models_dir.rglob(value))
            if matches:
                return str(matches[0])
        return value

    matches = sorted(models_dir.rglob("*.gguf"))
    if matches:
        return str(matches[0])
    return ""


def _provider_binary(app: Any, settings: Settings) -> Path | None:
    """Explicit llama-server path from settings, or the app's bin folder."""
    if settings.llm_server_path.strip():
        return Path(settings.llm_server_path).expanduser()
    return Path(app.state.paths.bin)


def _provider(target: Any, settings: Settings) -> Any:
    app = _app_of(target)
    kind = settings.llm_provider
    if settings.external_llm_enabled and kind == "llamacpp-local":
        kind = "external"
    local_model = (
        _resolve_llm_model(app, settings)
        if kind == "llamacpp-local"
        else settings.llm_model
    )
    signature = (
        kind,
        local_model,
        settings.external_llm_base_url,
        settings.external_llm_api_key,
        settings.llm_server_path,
        settings.llm_ctx_size,
        settings.llm_threads,
    )
    cached = getattr(app.state, PROVIDER_STATE_KEY, None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == signature:
        return cached[1]
    if cached is not None:
        old = cached[1] if isinstance(cached, tuple) else cached
        close = getattr(old, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best effort cleanup
                log.debug("cannot close previous LLM provider", exc_info=True)
    provider = create_provider(
        kind=kind,
        model=local_model,
        base_url=settings.external_llm_base_url,
        api_key=settings.external_llm_api_key or None,
        ctx_size=settings.llm_ctx_size,
        threads=settings.llm_threads,
        binary=_provider_binary(app, settings) if kind == "llamacpp-local" else None,
    )
    setattr(app.state, PROVIDER_STATE_KEY, (signature, provider))
    return provider


def _warn_provider_model_mismatch(settings: Settings) -> None:
    """Log a hint when ollama/external gets a local GGUF filename.

    Первопричина 404 «model not found» в логах: пользователь выбирает
    провайдера ollama, а в ``llm_model`` остаётся имя скачанного GGUF-файла
    (например ``T-lite-it-2.1-Q4_K_M.gguf``).  Ollama ждёт имя установленной
    модели (``ollama list``), а GGUF-файлы — это провайдер llamacpp-local.
    """
    kind = settings.llm_provider
    if kind in {"ollama", "openai-compatible", "external"}:
        model = (settings.llm_model or "").strip()
        if model.lower().endswith(".gguf"):
            log.warning(
                "провайдер %s получил имя GGUF-файла %r: ollama ждёт имя модели "
                "из `ollama list`, а GGUF-файлы предназначены для провайдера "
                "llamacpp-local (менеджер моделей)",
                kind,
                model,
            )


def _summarize_text(
    target: Any,
    text: str,
    progress: Any,
) -> dict[str, Any]:
    """Run the summarizer for *text* using the configured LLM provider."""
    app = _app_of(target)
    settings = app.state.settings
    kind = settings.llm_provider
    if settings.external_llm_enabled and kind == "llamacpp-local":
        kind = "external"
    if kind == "llamacpp-local" and not _resolve_llm_model(app, settings):
        raise LLMError(
            "GGUF-модель для саммари не найдена. Скачайте модель в менеджере "
            "моделей и выберите её в настройках."
        )
    _warn_provider_model_mismatch(settings)
    provider = _provider(app, settings)
    summarizer = Summarizer(provider, instructions=settings.summary_prompt)
    glossary = getattr(app.state, "llm_glossary", None)
    if not isinstance(glossary, Glossary):
        glossary = Glossary.from_dicts([])
    return summarizer.summarize(glossary.apply(text), progress=progress)


def make_summary_runner(app: Any) -> Any:
    """Build the callback used by the job manager for automatic summaries."""

    def runner(job_id: str, progress: Any) -> dict[str, Any]:
        manager: JobManager = app.state.jobs
        job = manager.get(job_id)
        if job is None or not job.text.strip():
            raise ValueError("Транскрипт пуст.")
        settings = app.state.settings
        kind = settings.llm_provider
        if settings.external_llm_enabled and kind == "llamacpp-local":
            kind = "external"
        if kind == "llamacpp-local":
            custom = settings.llm_server_path.strip() or None
            if find_llama_server(app.state.paths, custom=custom) is None:
                # The runtime is a one-time ~24 MB download.  Doing it inside
                # the summary worker keeps the HTTP request short and gives the
                # stage a real progress bar instead of a silent 30-second pause.
                def install_progress(value: float, phase: str) -> None:
                    progress(0.02 + 0.2 * max(0.0, min(value, 1.0)), phase)

                ensure_runtime(app.state.paths, progress=install_progress)

            def summary_progress(value: float, phase: str) -> None:
                progress(0.22 + 0.78 * max(0.0, min(value, 1.0)), phase)

            return _summarize_text(app, job.text, summary_progress)
        return _summarize_text(app, job.text, progress)

    return runner


@router.get("/providers")
def list_providers(request: Request) -> dict[str, Any]:
    """Available LLM providers and their health."""
    s = request.app.state.settings
    kind = s.llm_provider
    if kind == "llamacpp-local" and s.external_llm_enabled:
        kind = "external"
    local_model = (
        _resolve_llm_model(request, s) if kind == "llamacpp-local" else s.llm_model
    )
    provider = create_provider(
        kind=kind,
        model=local_model,
        base_url=s.external_llm_base_url,
        api_key=s.external_llm_api_key or None,
        binary=(
            _provider_binary(request.app, s) if kind == "llamacpp-local" else None
        ),
    )
    try:
        health = provider.health()
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best effort cleanup
                log.debug("cannot close probe provider", exc_info=True)
    return {"active": kind, "health": health}


@router.post("/summarize/{job_id}", status_code=202)
def summarize_job(request: Request, job_id: str) -> dict[str, Any]:
    """Start summary generation in the background.

    The UI polls the job; keeping the request short lets the user stop or
    restart the summary stage while llama.cpp is working.
    """
    manager = _manager(request)
    try:
        job = manager.start_stage(job_id, "summary")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Задача не найдена.") from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job_id": job_id, "status": "running", "job": job.detail()}


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


@router.post("/summarize-preview")
def summarize_preview(request: Request) -> dict[str, Any]:
    """Инструкция, которая уйдёт в модель при следующем саммари.

    Возвращает собранный промпт (системный + пользовательский) для проверки
    кастомного промпта из настроек без запуска LLM.
    """
    settings = request.app.state.settings
    from airec.llm.providers import _summary_messages

    messages = _summary_messages(
        "…текст задачи…", instructions=settings.summary_prompt
    )
    return {
        "custom_prompt": settings.summary_prompt,
        "uses_custom_prompt": bool(settings.summary_prompt.strip()),
        "messages": messages,
        "schema_is_fixed": True,
    }


@router.post("/glossary")
def update_glossary(request: Request, entries: list[dict[str, str]]) -> dict[str, Any]:
    """Replace the session glossary with *entries* ([{heard, correct, note}])."""
    glossary = Glossary.from_dicts(entries)
    request.app.state.llm_glossary = glossary
    return {"entries": glossary.as_dicts(), "count": len(glossary)}
