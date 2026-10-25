"""LLM providers: local llama-server, ollama, openai-compatible.

All providers implement :class:`LLMProvider`.  The default provider is
``llamacpp-local`` which spawns ``llama-server.exe`` as a child process on
``127.0.0.1`` and talks to its OpenAI-compatible HTTP API.  External providers
(ollama / openai-compatible) are used only when explicitly enabled by the user
(ADR-008).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_LLAMA_SERVER_PORT = 11435  # avoid clash with ollama's 11434
LLAMA_SERVER_STARTUP_TIMEOUT_S = 60.0
LLAMA_SERVER_REQUEST_TIMEOUT_S = 300.0


class LLMError(RuntimeError):
    """LLM provider error."""


class LLMProvider(ABC):
    """LLM provider interface."""

    @abstractmethod
    def summarize(self, text: str, *, language: str = "ru") -> dict[str, Any]:
        """Produce structured summary of *text*."""
        ...

    @abstractmethod
    def ask(self, question: str, *, context: str, language: str = "ru") -> str:
        """Answer *question* using *context* with citations."""
        ...

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Provider health / availability."""
        ...


def _parse_json_strict(text: str) -> dict[str, Any]:
    """Parse JSON from model output, tolerating markdown code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if "```" in cleaned[3:] else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


class LlamaCppProvider(LLMProvider):
    """Local ``llama-server`` provider.

    Spawns ``llama-server.exe`` from ``paths.bin`` (or PATH) as a child process
    on loopback and talks to its OpenAI-compatible ``/v1/chat/completions``
    endpoint.  The process is terminated when the provider is closed.
    """

    def __init__(
        self,
        *,
        model: str | Path,
        ctx_size: int = 4096,
        threads: int = 0,
        port: int = DEFAULT_LLAMA_SERVER_PORT,
        binary: str | Path | None = None,
    ) -> None:
        self.model = str(model)
        self.ctx_size = ctx_size
        self.threads = threads or os.cpu_count() or 4
        self.port = port
        self.binary = Path(binary) if binary else None
        self._process: subprocess.Popen[bytes] | None = None
        self._base_url = f"http://127.0.0.1:{port}"
        self._client = httpx.Client(base_url=self._base_url, timeout=LLAMA_SERVER_REQUEST_TIMEOUT_S)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def _resolve_binary(self) -> str:
        if self.binary is not None and self.binary.is_file():
            return str(self.binary)
        for candidate in ("llama-server.exe", "llama-server"):
            found = self.binary / candidate if self.binary else None
            if found is not None and found.is_file():
                return str(found)
        # Fall back to PATH.
        import shutil

        found = shutil.which("llama-server") or shutil.which("llama-server.exe")
        if found:
            return found
        raise LLMError(
            "llama-server не найден. Положите llama-server.exe в папку bin "
            "рядом с приложением или укажите путь в настройках."
        )

    def _ensure_running(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            binary = self._resolve_binary()
            cmd = [
                binary,
                "--model", self.model,
                "--ctx-size", str(self.ctx_size),
                "--threads", str(self.threads),
                "--host", "127.0.0.1",
                "--port", str(self.port),
            ]
            log.info("starting llama-server: %s", " ".join(cmd))
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=0x08000000 if os.name == "nt" else 0,  # CREATE_NO_WINDOW
            )
            # Wait for /v1/models to become available.
            deadline = time.monotonic() + LLAMA_SERVER_STARTUP_TIMEOUT_S
            last_error = ""
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    _, stderr = self._process.communicate()
                    last_error = stderr.decode("utf-8", errors="replace")[-2000:]
                    raise LLMError(f"llama-server завершился при старте: {last_error}")
                try:
                    resp = self._client.get("/v1/models", timeout=3.0)
                    if resp.status_code == 200:
                        log.info("llama-server ready on port %s", self.port)
                        return
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                time.sleep(0.5)
            raise LLMError(f"llama-server не стал доступен за {LLAMA_SERVER_STARTUP_TIMEOUT_S} с")

    def close(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            self._process = None

    # ------------------------------------------------------------------ #
    # LLMProvider
    # ------------------------------------------------------------------ #
    def health(self) -> dict[str, Any]:
        try:
            self._ensure_running()
            resp = self._client.get("/v1/models", timeout=5.0)
            return {
                "provider": "llamacpp-local",
                "available": resp.status_code == 200,
                "model": self.model,
                "port": self.port,
            }
        except Exception as exc:  # provider boundary
            return {
                "provider": "llamacpp-local",
                "available": False,
                "model": self.model,
                "error": str(exc),
            }

    def _chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        self._ensure_running()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        resp = self._client.post("/v1/chat/completions", json=payload, timeout=LLAMA_SERVER_REQUEST_TIMEOUT_S)
        if resp.status_code != 200:
            raise LLMError(f"llama-server вернул HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def summarize(self, text: str, *, language: str = "ru") -> dict[str, Any]:
        system = (
            "Ты — ассистент по саммари встреч. Пиши на русском языке. "
            "Не выдумывай фактов, которых нет в тексте. "
            "Ответ строго в формате JSON без markdown-блоков."
        )
        user = (
            "Проанализируй транскрибированную речь ниже и составь структурированное "
            "саммари. Каждое решение и каждую задачу привяжи к таймкоду (начало фрагмента), "
            "где оно упомянуто. Если информации недостаточно, оставь поле пустым.\n\n"
            "Текст:\n---\n" + text + "\n---\n\n"
            "Верни JSON со схемой:\n"
            "{\n"
            '  "tl_dr": "строка, 3-5 предложений",\n'
            '  "topics": ["тема 1", "тема 2"],\n'
            '  "decisions": [{"formulation": "строка", "timecode": "MM:SS"}],\n'
            '  "action_items": [{"owner": "строка", "deadline": "строка", "formulation": "строка", "timecode": "MM:SS"}],\n'
            '  "questions_open": ["вопрос 1"],\n'
            '  "risks": ["риск 1"],\n'
            '  "quotes": [{"quote": "цитата", "timecode": "MM:SS", "speaker": "Спикер N"}]\n'
            "}\n"
        )
        content = self._chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return _parse_json_strict(content)

    def ask(self, question: str, *, context: str, language: str = "ru") -> str:
        system = (
            "Ты отвечаешь на вопросы по транскрибированной встрече. "
            "Пиши на русском языке. Используй только информацию из контекста. "
            "Если в контексте нет ответа, скажи, что информации недостаточно. "
            "Ссылайся на таймкоды фрагментов, где найдена информация. "
            "Не выдумывай факты."
        )
        user = (
            "Контекст:\n---\n" + context + "\n---\n\n"
            "Вопрос: " + question + "\n"
        )
        return self._chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )


class ExternalProvider(LLMProvider):
    """External OpenAI-compatible provider (ollama / openai / YandexGPT / ...).

    Used only when explicitly enabled by the user.  No child process is spawned.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: int = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def health(self) -> dict[str, Any]:
        try:
            resp = self._client.get("/v1/models", timeout=5.0)
            return {
                "provider": "external",
                "available": resp.status_code == 200,
                "model": self.model,
                "base_url": self.base_url,
            }
        except Exception as exc:  # provider boundary
            return {"provider": "external", "available": False, "error": str(exc)}

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "stream": False}
        resp = self._client.post("/v1/chat/completions", json=payload, headers=self._headers())
        if resp.status_code != 200:
            raise LLMError(f"внешний LLM вернул HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def summarize(self, text: str, *, language: str = "ru") -> dict[str, Any]:
        system = (
            "Ты — ассистент по саммари встреч. Пиши на русском языке. "
            "Не выдумывай фактов, которых нет в тексте. "
            "Ответ строго в формате JSON без markdown-блоков."
        )
        user = (
            "Проанализируй транскрибированную речь ниже и составь структурированное "
            "саммари. Каждое решение и каждую задачу привяжи к таймкоду.\n\n"
            "Текст:\n---\n" + text + "\n---\n\n"
            "Верни JSON со схемой:\n"
            "{\n"
            '  "tl_dr": "строка",\n'
            '  "topics": [],\n'
            '  "decisions": [{"formulation": "", "timecode": ""}],\n'
            '  "action_items": [{"owner": "", "deadline": "", "formulation": "", "timecode": ""}],\n'
            '  "questions_open": [],\n'
            '  "risks": [],\n'
            '  "quotes": [{"quote": "", "timecode": "", "speaker": ""}]\n'
            "}\n"
        )
        content = self._chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return _parse_json_strict(content)

    def ask(self, question: str, *, context: str, language: str = "ru") -> str:
        system = (
            "Ты отвечаешь на вопросы по транскрибированной встрече. "
            "Пиши на русском языке. Используй только информацию из контекста. "
            "Ссылайся на таймкоды. Не выдумывай факты."
        )
        user = "Контекст:\n---\n" + context + "\n---\n\nВопрос: " + question + "\n"
        return self._chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )


def create_provider(
    kind: str = "llamacpp-local",
    *,
    model: str = "auto",
    base_url: str = "http://127.0.0.1:11434",
    api_key: str | None = None,
    ctx_size: int = 4096,
    threads: int = 0,
    port: int = DEFAULT_LLAMA_SERVER_PORT,
    binary: str | Path | None = None,
) -> LLMProvider:
    """Factory for LLM providers."""
    if kind == "llamacpp-local":
        return LlamaCppProvider(model=model, ctx_size=ctx_size, threads=threads, port=port, binary=binary)
    if kind in {"ollama", "openai-compatible", "external"}:
        return ExternalProvider(base_url=base_url, model=model, api_key=api_key)
    raise LLMError(f"Неизвестный провайдер LLM: {kind}")
