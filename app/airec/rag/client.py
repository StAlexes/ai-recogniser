"""Small, dependency-light client for the local ``ragkb`` HTTP API.

The application deliberately does not import or embed ragkb's index.  It talks
to an already running service (``/health``, ``/search``, ``/ask``) and keeps the
integration optional: an unreachable RAG must never stop transcription or make
the rest of the core unusable.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from airec.core.config import Settings

log = logging.getLogger(__name__)

STATUS_TIMEOUT_S = 6.0
SEARCH_TIMEOUT_S = 45.0
ASK_TIMEOUT_S = 240.0


class RagError(RuntimeError):
    """A human-readable RAG failure.

    ``status_code`` is the HTTP status the core should return.  A missing
    service is a bad gateway; a timeout is a gateway timeout.
    """

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def normalize_base_url(value: str) -> str:
    """Return a normalized base URL or raise ``RagError`` for invalid input."""
    url = value.strip().rstrip("/")
    if not url:
        raise RagError("Адрес RAG не задан в настройках.", status_code=400)
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise RagError("Некорректный адрес RAG.", status_code=400) from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RagError(
            "Адрес RAG должен начинаться с http:// или https://.",
            status_code=400,
        )
    if parsed.username or parsed.password:
        raise RagError("Адрес RAG не должен содержать логин или пароль.", status_code=400)
    if parsed.query or parsed.fragment:
        raise RagError("Адрес RAG не должен содержать query или fragment.", status_code=400)
    return url


def _error_detail(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    return None


class RagKbClient:
    """Synchronous client for one ragkb instance.

    ``transport`` exists for tests; production code should not pass it.
    """

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self._transport = transport

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        timeout_s: float,
    ) -> dict[str, Any]:
        try:
            with httpx.Client(
                timeout=httpx.Timeout(timeout_s),
                transport=self._transport,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                response = client.request(
                    method,
                    f"{self.base_url}{path}",
                    params=params,
                    json=json,
                )
        except httpx.TimeoutException as exc:
            raise RagError(
                f"RAG не ответил за {int(timeout_s)} с. "
                "Проверьте, что ragkb и Ollama запущены.",
                status_code=504,
            ) from exc
        except httpx.RequestError as exc:
            raise RagError(
                f"RAG недоступен по адресу {self.base_url}. "
                "Запустите ragkb или проверьте адрес в настройках.",
                status_code=502,
            ) from exc

        if response.status_code >= 400:
            detail = _error_detail(response)
            message = detail or f"RAG вернул HTTP {response.status_code}."
            raise RagError(message, status_code=502)

        try:
            payload = response.json()
        except ValueError as exc:
            raise RagError("RAG вернул некорректный JSON.", status_code=502) from exc
        if not isinstance(payload, dict):
            raise RagError("RAG вернул неожиданный формат ответа.", status_code=502)
        return payload

    def status(self) -> dict[str, Any]:
        """Return a non-throwing connection summary for the UI."""
        started = time.perf_counter()
        try:
            payload = self._request("GET", "/health", timeout_s=STATUS_TIMEOUT_S)
        except RagError as exc:
            return {
                "enabled": True,
                "configured_url": self.base_url,
                "reachable": False,
                "latency_ms": None,
                "chunks": None,
                "indexing": None,
                "gen_model": None,
                "error": str(exc),
            }

        latency_ms = int((time.perf_counter() - started) * 1000)
        chunks = payload.get("chunks")
        return {
            "enabled": True,
            "configured_url": self.base_url,
            "reachable": True,
            "latency_ms": latency_ms,
            "chunks": chunks if isinstance(chunks, int) else None,
            "indexing": payload.get("indexing") if isinstance(payload.get("indexing"), bool) else None,
            "gen_model": payload.get("gen_model")
            if isinstance(payload.get("gen_model"), str)
            else None,
            "error": None,
        }

    def search(self, query: str, k: int) -> dict[str, Any]:
        clean_query = query.strip()
        if not clean_query:
            raise RagError("Поисковый запрос пуст.", status_code=400)
        payload = self._request(
            "GET",
            "/search",
            params={"q": clean_query, "k": k},
            timeout_s=SEARCH_TIMEOUT_S,
        )
        results = payload.get("results")
        if not isinstance(results, list):
            raise RagError("RAG вернул ответ без списка результатов.", status_code=502)
        return {
            "query": clean_query,
            "results": [self._clean_source(item) for item in results if isinstance(item, dict)],
        }

    def ask(self, question: str, k: int) -> dict[str, Any]:
        clean_question = question.strip()
        if not clean_question:
            raise RagError("Вопрос пуст.", status_code=400)
        payload = self._request(
            "POST",
            "/ask",
            json={"question": clean_question, "k": k},
            timeout_s=ASK_TIMEOUT_S,
        )
        answer = payload.get("answer")
        if not isinstance(answer, str):
            raise RagError("RAG вернул ответ без текста.", status_code=502)
        sources = payload.get("sources")
        return {
            "question": clean_question,
            "answer": answer,
            "sources": [
                self._clean_source(item) for item in sources if isinstance(item, dict)
            ]
            if isinstance(sources, list)
            else [],
        }

    @staticmethod
    def _clean_source(item: dict[str, Any]) -> dict[str, Any]:
        """Keep the public fields stable even if ragkb adds new ones."""
        result: dict[str, Any] = {
            "source": str(item.get("source") or ""),
            "breadcrumb": str(item.get("breadcrumb") or ""),
            "text": str(item.get("text") or ""),
        }
        for key in ("id", "position"):
            value = item.get(key)
            if isinstance(value, int):
                result[key] = value
        for key in ("score", "dense", "lexical"):
            value = item.get(key)
            if isinstance(value, (int, float)):
                result[key] = float(value)
        for key in ("exact", "match"):
            value = item.get(key)
            if isinstance(value, bool | str):
                result[key] = value
        return result


def client_from_settings(settings: Settings) -> RagKbClient:
    return RagKbClient(settings.rag_url)


def disabled_status(settings: Settings) -> dict[str, Any]:
    return {
        "enabled": False,
        "configured_url": settings.rag_url,
        "reachable": False,
        "latency_ms": None,
        "chunks": None,
        "indexing": None,
        "gen_model": None,
        "error": "Интеграция с RAG выключена в настройках.",
    }
