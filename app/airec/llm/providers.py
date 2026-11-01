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

from airec.llm.runtime import find_llama_server

log = logging.getLogger(__name__)

DEFAULT_LLAMA_SERVER_PORT = 11435  # avoid clash with ollama's 11434
LLAMA_SERVER_STARTUP_TIMEOUT_S = 180.0
LLAMA_SERVER_REQUEST_TIMEOUT_S = 600.0

# Canonical JSON schema of the summary result.  It is always appended to the
# prompt by the app and is never under the user's control, so the parsed
# result keeps its shape (the UI and exports depend on these keys).
SUMMARY_SCHEMA = """{
  "tl_dr": "строка, 3-5 предложений",
  "topics": ["тема 1", "тема 2"],
  "decisions": [{"formulation": "строка", "timecode": "MM:SS"}],
  "action_items": [
    {"owner": "строка", "deadline": "строка", "formulation": "строка", "timecode": "MM:SS"}
  ],
  "questions_open": ["вопрос 1"],
  "risks": ["риск 1"],
  "quotes": [{"quote": "цитата", "timecode": "MM:SS", "speaker": "Спикер N"}]
}"""

# Machine-readable form of SUMMARY_SCHEMA.  It is sent to llama-server as a
# ``response_format`` JSON-schema constraint: the model output is then produced
# under a GBNF grammar and cannot be malformed JSON (T-lite/Qwen3 occasionally
# break a key name mid-object, e.g. "time  code", which used to fail the whole
# summary stage with JSONDecodeError).
SUMMARY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tl_dr": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "formulation": {"type": "string"},
                    "timecode": {
                        "type": "string",
                        "pattern": "^[0-9]{1,2}:[0-9]{2}$",
                        "description": "MM:SS от начала записи",
                    },
                },
                "required": ["formulation", "timecode"],
            },
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "deadline": {"type": "string"},
                    "formulation": {"type": "string"},
                    "timecode": {
                        "type": "string",
                        "pattern": "^[0-9]{1,2}:[0-9]{2}$",
                        "description": "MM:SS от начала записи",
                    },
                },
                "required": ["owner", "deadline", "formulation", "timecode"],
            },
        },
        "questions_open": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "timecode": {
                        "type": "string",
                        "pattern": "^[0-9]{1,2}:[0-9]{2}$",
                        "description": "MM:SS от начала записи",
                    },
                    "speaker": {"type": "string"},
                },
                "required": ["quote", "timecode", "speaker"],
            },
        },
    },
    "required": [
        "tl_dr",
        "topics",
        "decisions",
        "action_items",
        "questions_open",
        "risks",
        "quotes",
    ],
}

DEFAULT_SUMMARY_SYSTEM = (
    "Ты — ассистент по саммари встреч. Пиши на русском языке. "
    "Не выдумывай фактов, которых нет в тексте. "
    "Ответ строго в формате JSON без markdown-блоков."
)

# Системный промпт для кастомной инструкции: модель отвечает свободным
# текстом по правилам пользователя, поэтому требование JSON снимается.
DEFAULT_SUMMARY_SYSTEM_FREEFORM = (
    "Ты — ассистент по саммари встреч. Пиши на русском языке. "
    "Не выдумывай фактов, которых нет в тексте. "
    "Следуй инструкции пользователя: оформи ответ читаемым текстом, "
    "можно с разделами, списками и таблицами."
)

DEFAULT_SUMMARY_TASK = (
    "Проанализируй транскрибированную речь ниже и составь структурированное "
    "саммари. Каждое решение и каждую задачу привяжи к таймкоду (начало фрагмента), "
    "где оно упомянуто. Если информации недостаточно, оставь поле пустым."
)


def _summary_response_format() -> dict[str, Any]:
    """OpenAI-style response_format constraining output to SUMMARY_JSON_SCHEMA."""
    return {
        "type": "json_schema",
        "json_schema": {"name": "summary", "schema": SUMMARY_JSON_SCHEMA},
    }


def _summary_messages(
    text: str, *, instructions: str = "", freeform: bool | None = None
) -> list[dict[str, str]]:
    """Build chat messages for a summary request.

    *instructions* — необязательная инструкция пользователя (настройка
    ``summary_prompt``).  Кастомная инструкция заменяет стандартную задачу и
    снимает JSON-ограничения: модель отвечает свободным текстом по правилам
    пользователя (разделы, списки, таблицы), поэтому ``response_format`` и
    «Верни JSON строго со схемой» в этом режиме не применяются.

    *freeform* — явное переключение режима для map/reduce-вызовов, где
    инструкция уже вшита в *text*: ``True`` принудительно снимает JSON-режим,
    ``None`` выводит его из наличия *instructions*.
    """
    use_freeform = bool(instructions.strip()) if freeform is None else freeform
    if use_freeform:
        head = instructions.strip()
        user = (
            f"{head}\n\n" if head else ""
        ) + "Текст:\n---\n" + text + "\n---\n"
        return [
            {"role": "system", "content": DEFAULT_SUMMARY_SYSTEM_FREEFORM},
            {"role": "user", "content": user},
        ]
    user = (
        f"{DEFAULT_SUMMARY_TASK}\n\n"
        "Текст:\n---\n" + text + "\n---\n\n"
        "Верни JSON строго со схемой ниже, без лишних ключей и markdown:\n"
        f"{SUMMARY_SCHEMA}\n"
    )
    return [
        {"role": "system", "content": DEFAULT_SUMMARY_SYSTEM},
        {"role": "user", "content": user},
    ]


class LLMError(RuntimeError):
    """LLM provider error."""


class LLMProvider(ABC):
    """LLM provider interface."""

    @abstractmethod
    def summarize(
        self,
        text: str,
        *,
        language: str = "ru",
        instructions: str = "",
        freeform: bool | None = None,
    ) -> dict[str, Any]:
        """Produce structured summary of *text*.

        *instructions* — необязательная инструкция пользователя (кастомный
        промпт из настроек); она заменяет встроенную задачу и отвечает
        свободным текстом (``custom: True``), без навязанной JSON-схемы.
        *freeform* — явное переключение свободного режима для map/reduce,
        где инструкция уже вшита в *text*.
        """
        ...

    @abstractmethod
    def ask(self, question: str, *, context: str, language: str = "ru") -> str:
        """Answer *question* using *context* with citations."""
        ...

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Provider health / availability."""
        ...


def _normalize_summary(obj: Any) -> dict[str, Any]:
    """Bring a model-returned summary to the canonical key shape.

    Модели иногда искажают имена ключей ("time  code", "actionitems") и
    возвращают строки вместо массивов.  Нормализация чинит известные
    искажения, чтобы UI и экспорт видели ровно поля схемы.
    """
    if not isinstance(obj, dict):
        return {}
    canonical = {
        "tl_dr": "tl_dr",
        "tldr": "tl_dr",
        "tl;dr": "tl_dr",
        "summary": "tl_dr",
        "topics": "topics",
        "theme": "topics",
        "decisions": "decisions",
        "action_items": "action_items",
        "actionitems": "action_items",
        "action items": "action_items",
        "tasks": "action_items",
        "questions_open": "questions_open",
        "open_questions": "questions_open",
        "questions": "questions_open",
        "risks": "risks",
        "quotes": "quotes",
        "quote": "quotes",
    }
    key_fix = {
        "formulation": "formulation",
        "timecode": "timecode",
        "time code": "timecode",
        "time_code": "timecode",
        "owner": "owner",
        "deadline": "deadline",
        "quote": "quote",
        "speaker": "speaker",
    }

    def fix_item(item: Any, fields: tuple[str, ...]) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        fixed: dict[str, Any] = {}
        for key, value in item.items():
            # Схлопываем пробелы: T-lite рвёт "timecode" на "time  code".
            norm = " ".join(key.split()).lower()
            norm = key_fix.get(norm, norm)
            fixed[norm] = value
        return {field: fixed.get(field, "") for field in fields}

    result: dict[str, Any] = {}
    for key, value in obj.items():
        norm = key.strip().lower()
        target = canonical.get(norm)
        if target is None:
            continue
        if target == "tl_dr" and isinstance(value, str):
            result[target] = value
        elif target in {"topics", "questions_open", "risks"}:
            if isinstance(value, str) and value.strip():
                result[target] = [value.strip()]
            elif isinstance(value, list):
                result[target] = [str(item) for item in value]
            else:
                result[target] = []
        elif target == "decisions":
            items = value if isinstance(value, list) else [value]
            result[target] = [
                fixed
                for fixed in (fix_item(item, ("formulation", "timecode")) for item in items)
                if fixed is not None
            ]
        elif target == "action_items":
            items = value if isinstance(value, list) else [value]
            result[target] = [
                fixed
                for fixed in (
                    fix_item(item, ("owner", "deadline", "formulation", "timecode"))
                    for item in items
                )
                if fixed is not None
            ]
        elif target == "quotes":
            items = value if isinstance(value, list) else [value]
            result[target] = [
                fixed
                for fixed in (
                    fix_item(item, ("quote", "timecode", "speaker")) for item in items
                )
                if fixed is not None
            ]
    # Все ключи схемы должны присутствовать (пусть и пустыми).
    for key in ("tl_dr", "topics", "decisions", "action_items", "questions_open", "risks", "quotes"):
        result.setdefault(key, [] if key != "tl_dr" else "")
    return result


def _parse_json_strict(text: str) -> dict[str, Any]:
    """Parse JSON from model output, tolerating fences and short preambles."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if "```" in cleaned[3:] else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Small local models sometimes add "Вот результат:" before the JSON.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


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
        found = find_llama_server(custom=self.binary)
        if found is not None:
            return str(found)
        raise LLMError(
            "llama-server не найден. Скачайте runtime в менеджере моделей "
            "(«Среда llama.cpp») или укажите путь в настройках."
        )

    def _ensure_running(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            model_path = Path(self.model)
            if not model_path.is_file() or model_path.suffix.lower() != ".gguf":
                raise LLMError(
                    "GGUF-модель для саммари не найдена. Скачайте модель в "
                    "менеджере моделей и выберите её в настройках."
                )
            binary = self._resolve_binary()
            cmd = [
                binary,
                "--model", self.model,
                "--ctx-size", str(self.ctx_size),
                "--threads", str(self.threads),
                # Summaries do not need a long chain of thought.  Disabling it
                # keeps CPU-only generation several times faster and avoids
                # models that emit reasoning_content but no final JSON.
                "--reasoning", "off",
                "--host", "127.0.0.1",
                "--port", str(self.port),
            ]
            log.info("starting llama-server: %s", " ".join(cmd))
            # Never pipe llama-server output: it logs continuously and an
            # unread pipe would eventually block the server.  Logs are not
            # needed for normal operation; startup failures are reported by
            # the exit code and the health probe below.
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=0x08000000 if os.name == "nt" else 0,  # CREATE_NO_WINDOW
            )
            # Wait for /v1/models to become available.
            deadline = time.monotonic() + LLAMA_SERVER_STARTUP_TIMEOUT_S
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise LLMError(
                        "llama-server завершился при старте "
                        f"(код {self._process.returncode}). Проверьте, что "
                        "runtime llama.cpp и GGUF-модель совместимы."
                    )
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

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        self._ensure_running()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if response_format is not None:
            # llama-server converts a JSON schema into a GBNF grammar: the
            # output is guaranteed to be valid JSON of this shape.
            payload["response_format"] = response_format
        resp = self._client.post("/v1/chat/completions", json=payload, timeout=LLAMA_SERVER_REQUEST_TIMEOUT_S)
        if resp.status_code != 200:
            raise LLMError(f"llama-server вернул HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def summarize(
        self,
        text: str,
        *,
        language: str = "ru",
        instructions: str = "",
        freeform: bool | None = None,
    ) -> dict[str, Any]:
        messages = _summary_messages(
            text, instructions=instructions, freeform=freeform
        )
        # Кастомный промпт отвечает свободным текстом: грамматика JSON-схемы
        # физически не позволяет модели оформить саммари разделами/таблицами,
        # поэтому ограничение включается только для встроенного промпта.
        custom = bool(instructions.strip()) or bool(freeform)
        content = self._chat(
            messages,
            temperature=0.2,
            response_format=None if custom else _summary_response_format(),
        )
        if custom:
            # Свободный текст — результат как есть, UI рендерит его без схемы.
            return {"tl_dr": content.strip(), "custom": True}
        try:
            parsed = _parse_json_strict(content)
        except json.JSONDecodeError as exc:
            # Грамматика делает это практически невозможным; остаётся на случай
            # несовместимого llama-server (без поддержки json_schema).
            raise LLMError(
                "модель вернула невалидный JSON: "
                f"{exc}; начало ответа: {content[:200]!r}"
            ) from exc
        return _normalize_summary(parsed)

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

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if response_format is not None:
            # Не все OpenAI-совместимые серверы поддерживают json_schema;
            # невалидный JSON у внешних провайдеров ловит retry ниже.
            payload["response_format"] = response_format
        resp = self._client.post("/v1/chat/completions", json=payload, headers=self._headers())
        if resp.status_code != 200:
            # Некоторые серверы (старые ollama) не знают response_format —
            # повторяем без ограничения, парсинг подстрахует нормализация.
            if response_format is not None and resp.status_code in {400, 404, 422}:
                log.warning(
                    "провайдер не принял response_format (HTTP %s); повтор без него",
                    resp.status_code,
                )
                payload.pop("response_format")
                resp = self._client.post(
                    "/v1/chat/completions", json=payload, headers=self._headers()
                )
                if resp.status_code != 200:
                    raise LLMError(f"внешний LLM вернул HTTP {resp.status_code}: {resp.text[:500]}")
            else:
                raise LLMError(f"внешний LLM вернул HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def summarize(
        self,
        text: str,
        *,
        language: str = "ru",
        instructions: str = "",
        freeform: bool | None = None,
    ) -> dict[str, Any]:
        messages = _summary_messages(
            text, instructions=instructions, freeform=freeform
        )
        # Кастомный промпт отвечает свободным текстом: json_schema ломает
        # оформление из промпта (разделы/таблицы), ограничение — только для
        # встроенного промпта.
        custom = bool(instructions.strip()) or bool(freeform)
        content = self._chat(
            messages,
            temperature=0.2,
            response_format=None if custom else _summary_response_format(),
        )
        if custom:
            # Свободный текст без JSON-ретрая: ответ как есть.
            return {"tl_dr": content.strip(), "custom": True}
        try:
            parsed = _parse_json_strict(content)
        except json.JSONDecodeError as exc:
            # Модели без грамматического ограничения иногда ломают JSON
            # (T-lite/Qwen3: "time  code", обрыв на середине).  Одна попытка
            # с явной поправкой почти всегда даёт валидный ответ.
            log.warning("invalid JSON from LLM (%s); retrying with correction", exc)
            retry_messages = [
                *messages,
                {"role": "assistant", "content": content[:2000]},
                {
                    "role": "user",
                    "content": (
                        "Твой предыдущий ответ не является валидным JSON "
                        f"({exc}). Верни ответ заново: только JSON, точно по "
                        "схеме, без markdown и пояснений."
                    ),
                },
            ]
            retry_content = self._chat(retry_messages, temperature=0.2)
            try:
                parsed = _parse_json_strict(retry_content)
            except json.JSONDecodeError as retry_exc:
                raise LLMError(
                    "модель вернула невалидный JSON дважды: "
                    f"{retry_exc}; начало ответа: {retry_content[:200]!r}"
                ) from retry_exc
        return _normalize_summary(parsed)

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
