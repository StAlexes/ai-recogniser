from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from airec.api import llm as llm_api
from airec.core.config import Settings
from airec.llm.providers import (
    DEFAULT_SUMMARY_TASK,
    ExternalProvider,
    LlamaCppProvider,
    _parse_json_strict,
    _summary_messages,
)
from airec.llm.summary import Summarizer


def test_summary_runner_uses_custom_llama_server_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = SimpleNamespace(text="Текст встречи.")
    app = SimpleNamespace(
        state=SimpleNamespace(
            jobs=SimpleNamespace(get=lambda _job_id: job),
            settings=Settings(
                llm_provider="llamacpp-local",
                llm_server_path=r"C:\tools\llama-server.exe",
            ),
            paths=SimpleNamespace(models=tmp_path / "models"),
        )
    )
    custom = tmp_path / "llama-server.exe"
    custom.write_bytes(b"exe")
    seen: list[str | None] = []

    def fake_find(_paths: object, custom: str | None = None) -> Path | None:
        seen.append(custom)
        return Path(custom) if custom else None

    monkeypatch.setattr(llm_api, "find_llama_server", fake_find)
    monkeypatch.setattr(
        llm_api,
        "ensure_runtime",
        lambda *_args, **_kwargs: pytest.fail("runtime download must be skipped"),
    )
    monkeypatch.setattr(
        llm_api,
        "_summarize_text",
        lambda _app, text, _progress: {"tl_dr": text},
    )

    result = llm_api.make_summary_runner(app)("job-1", lambda _value, _phase: None)

    assert result == {"tl_dr": "Текст встречи."}
    assert seen == [r"C:\tools\llama-server.exe"]


# -------------------------------------------------------------------------- #
# Кастомный промпт саммаризации
# -------------------------------------------------------------------------- #
class _FakeProvider:
    """Запоминает messages и возвращает валидный JSON-саммари."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def summarize(
        self, text: str, *, language: str = "ru", instructions: str = ""
    ) -> dict[str, object]:
        self.calls.append(
            _summary_messages(text, instructions=instructions)
        )
        return {
            "tl_dr": text[:50],
            "topics": [],
            "decisions": [],
            "action_items": [],
            "questions_open": [],
            "risks": [],
            "quotes": [],
        }

    def ask(self, question: str, *, context: str, language: str = "ru") -> str:
        return "ответ"

    def health(self) -> dict[str, object]:
        return {"available": True}


class TestSummaryPromptMessages:
    def test_default_instructions_when_empty(self) -> None:
        messages = _summary_messages("Текст", instructions="")
        system = messages[0]["content"]
        user = messages[1]["content"]
        assert system == (
            "Ты — ассистент по саммари встреч. Пиши на русском языке. "
            "Не выдумывай фактов, которых нет в тексте. "
            "Ответ строго в формате JSON без markdown-блоков."
        )
        assert DEFAULT_SUMMARY_TASK in user
        assert '"tl_dr"' in user
        assert "Верни JSON строго со схемой" in user

    def test_custom_instructions_replace_task_keep_schema(self) -> None:
        messages = _summary_messages(
            "Текст", instructions="Делай саммари в стиле телеграм-канала."
        )
        user = messages[1]["content"]
        # Кастомная инструкция заменяет стандартную задачу...
        assert "Делай саммари в стиле телеграм-канала." in user
        assert DEFAULT_SUMMARY_TASK not in user
        # ...но схема и требование JSON всегда добавляются приложением.
        assert '"tl_dr"' in user
        assert '"action_items"' in user
        assert "Верни JSON строго со схемой" in user

    def test_schema_is_never_user_controlled(self) -> None:
        # Даже если пользователь попросит вернуть другой формат, схема
        # приписывается приложением и сохраняет ключи результата.
        messages = _summary_messages("Текст", instructions="Верни CSV, не JSON.")
        user = messages[1]["content"]
        assert '"tl_dr"' in user and "quotes" in user


class TestSummarizerCustomPrompt:
    def test_instructions_passed_to_single_block(self) -> None:
        provider = _FakeProvider()
        summarizer = Summarizer(provider, instructions="Подчёркивай сроки.")
        summarizer.summarize("Короткий текст встречи.")
        assert len(provider.calls) == 1
        user = provider.calls[0][1]["content"]
        assert "Подчёркивай сроки." in user
        assert '"tl_dr"' in user

    def test_instructions_passed_to_map_and_reduce(self) -> None:
        provider = _FakeProvider()
        summarizer = Summarizer(provider, instructions="Подчёркивай сроки.")
        long_text = ". ".join(f"Предложение номер {i} про проект" for i in range(400))
        summarizer.summarize(long_text)
        # map-вызовы + reduce-вызов — все с кастомной инструкцией
        assert len(provider.calls) >= 3
        for messages in provider.calls:
            assert "Подчёркивай сроки." in messages[1]["content"]

    def test_blank_instructions_use_default(self) -> None:
        provider = _FakeProvider()
        summarizer = Summarizer(provider, instructions="   \n  ")
        summarizer.summarize("Короткий текст встречи.")
        user = provider.calls[0][1]["content"]
        assert DEFAULT_SUMMARY_TASK in user

    def test_settings_prompt_reaches_summarizer(self, tmp_path: Path) -> None:
        # _summarize_text прокидывает settings.summary_prompt в Summarizer.
        captured: dict[str, object] = {}

        class _CaptureProvider(_FakeProvider):
            def summarize(
                self, text: str, *, language: str = "ru", instructions: str = ""
            ) -> dict[str, object]:
                captured["instructions"] = instructions
                return super().summarize(text, language=language, instructions=instructions)

        app = SimpleNamespace(
            state=SimpleNamespace(
                settings=Settings(
                    llm_provider="external",
                    summary_prompt="Кастомный промпт из настроек.",
                ),
                llm_glossary=None,
                paths=SimpleNamespace(models=tmp_path / "models"),
            )
        )
        monkey_target = SimpleNamespace(state=app.state)
        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(llm_api, "_provider", lambda _app, _s: _CaptureProvider())
            llm_api._summarize_text(monkey_target, "Текст встречи.", lambda _v, _p: None)
        finally:
            monkeypatch.undo()
        assert captured["instructions"] == "Кастомный промпт из настроек."


class TestProviderSummarizeSignatures:
    def test_llamacpp_summarize_accepts_instructions(self, monkeypatch) -> None:
        provider = LlamaCppProvider(model="whatever.gguf")
        seen: dict[str, object] = {}

        def fake_chat(messages, *, temperature: float = 0.2) -> str:
            seen["messages"] = messages
            return json.dumps({"tl_dr": "ok", "topics": []}, ensure_ascii=False)

        monkeypatch.setattr(provider, "_chat", fake_chat)
        provider._ensure_running = lambda: None  # type: ignore[method-assign]
        result = provider.summarize("Текст", instructions="Мой промпт.")
        assert result["tl_dr"] == "ok"
        user = seen["messages"][1]["content"]
        assert "Мой промпт." in user and '"tl_dr"' in user

    def test_external_summarize_accepts_instructions(self, monkeypatch) -> None:
        provider = ExternalProvider(
            base_url="http://127.0.0.1:11434", model="gemma3:4b"
        )
        seen: dict[str, object] = {}

        def fake_post(url, *, json=None, headers=None, **_kwargs):
            seen["messages"] = json["messages"]

            class _Resp:
                status_code = 200

                def json(self) -> dict[str, object]:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": json_str({"tl_dr": "ok", "topics": []})
                                }
                            }
                        ]
                    }

            return _Resp()

        def json_str(payload: dict[str, object]) -> str:
            import json as _json

            return _json.dumps(payload, ensure_ascii=False)

        monkeypatch.setattr(provider._client, "post", fake_post)
        result = provider.summarize("Текст", instructions="Мой промпт.")
        assert result["tl_dr"] == "ok"
        user = seen["messages"][1]["content"]
        assert "Мой промпт." in user and '"tl_dr"' in user


class TestParseJsonStrict:
    def test_tolerates_fences_and_preamble(self) -> None:
        raw = 'Вот результат:\n```json\n{"tl_dr": "ok"}\n```'
        assert _parse_json_strict(raw) == {"tl_dr": "ok"}

    def test_raises_on_garbage(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_json_strict("модель вернула вовсе не JSON")
