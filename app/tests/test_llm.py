from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from airec.api import llm as llm_api
from airec.core.config import Settings
from airec.llm.providers import (
    DEFAULT_SUMMARY_TASK,
    SUMMARY_JSON_SCHEMA,
    ExternalProvider,
    LlamaCppProvider,
    LLMError,
    _normalize_summary,
    _parse_json_strict,
    _server_cmd,
    _summary_messages,
    _summary_response_format,
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

    def fake_find(
        _paths: object, custom: str | None = None, *, device: str = "cpu"
    ) -> Path | None:
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
    """Запоминает messages и возвращает ответ как реальный провайдер."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def summarize(
        self,
        text: str,
        *,
        language: str = "ru",
        instructions: str = "",
        freeform: bool | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            _summary_messages(text, instructions=instructions, freeform=freeform)
        )
        if instructions.strip() or freeform:
            # Кастомный промпт / свободный режим: текст, как реальные провайдеры.
            return {"tl_dr": f"custom:{text[:50]}", "custom": True}
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

    def test_custom_instructions_replace_task_drop_schema(self) -> None:
        messages = _summary_messages(
            "Текст", instructions="Делай саммари в стиле телеграм-канала."
        )
        system = messages[0]["content"]
        user = messages[1]["content"]
        # Кастомная инструкция заменяет стандартную задачу и снимает
        # JSON-ограничения: модель отвечает свободным текстом по правилам
        # пользователя (разделы, списки, таблицы).
        assert "Делай саммари в стиле телеграм-канала." in user
        assert DEFAULT_SUMMARY_TASK not in user
        assert "Верни JSON строго со схемой" not in user
        assert '"tl_dr"' not in user
        assert "строго в формате JSON" not in system
        assert "оформи ответ читаемым текстом" in system

    def test_schema_is_never_user_controlled(self) -> None:
        # Без кастомного промпта схема приписывается приложением и сохраняет
        # ключи результата.
        messages = _summary_messages("Текст", instructions="")
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
        assert "Короткий текст встречи." in user

    def test_instructions_passed_to_map_and_reduce(self) -> None:
        provider = _FakeProvider()
        summarizer = Summarizer(provider, instructions="Подчёркивай сроки.")
        long_text = ". ".join(f"Предложение номер {i} про проект" for i in range(400))
        summarizer.summarize(long_text)
        # map-вызовы (инструкция внутри map-промпта) + reduce-вызов
        assert len(provider.calls) >= 3
        for messages in provider.calls:
            assert "Подчёркивай сроки." in messages[1]["content"]

    def test_map_and_reduce_use_freeform_mode(self) -> None:
        # Длинный транскрипт с кастомным промптом: map- и reduce-вызовы идут
        # в свободном режиме (freeform=True) — провайдер не навязывает схему.
        seen_freeform: list[bool | None] = []

        class _FreeformProvider(_FakeProvider):
            def summarize(
                self,
                text: str,
                *,
                language: str = "ru",
                instructions: str = "",
                freeform: bool | None = None,
            ) -> dict[str, object]:
                seen_freeform.append(freeform)
                return super().summarize(
                    text, language=language, instructions=instructions, freeform=freeform
                )

        provider = _FreeformProvider()
        summarizer = Summarizer(provider, instructions="Подчёркивай сроки.")
        long_text = ". ".join(f"Предложение номер {i} про проект" for i in range(400))
        result = summarizer.summarize(long_text)
        assert len(seen_freeform) >= 3
        assert all(flag is True for flag in seen_freeform)
        assert result.get("custom") is True

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
                self,
                text: str,
                *,
                language: str = "ru",
                instructions: str = "",
                freeform: bool | None = None,
            ) -> dict[str, object]:
                captured["instructions"] = instructions
                return super().summarize(
                    text, language=language, instructions=instructions, freeform=freeform
                )

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

        def fake_chat(messages, *, temperature: float = 0.2, response_format=None) -> str:
            seen["messages"] = messages
            seen["response_format"] = response_format
            return "Саммари свободным текстом по промпту."

        monkeypatch.setattr(provider, "_chat", fake_chat)
        provider._ensure_running = lambda: None  # type: ignore[method-assign]
        result = provider.summarize("Текст", instructions="Мой промпт.")
        # Кастомный промпт: свободный текст, без навязанной схемы.
        assert result["tl_dr"] == "Саммари свободным текстом по промпту."
        assert result["custom"] is True
        assert seen["response_format"] is None
        user = seen["messages"][1]["content"]
        assert "Мой промпт." in user and "Текст" in user
        assert '"tl_dr"' not in user

    def test_external_summarize_accepts_instructions(self, monkeypatch) -> None:
        provider = ExternalProvider(
            base_url="http://127.0.0.1:11434", model="gemma3:4b"
        )
        seen: dict[str, object] = {}

        def fake_post(url, *, json=None, headers=None, **_kwargs):
            seen["messages"] = json["messages"]
            seen["response_format"] = json.get("response_format")

            class _Resp:
                status_code = 200

                def json(self) -> dict[str, object]:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "Свободный текст саммари."
                                }
                            }
                        ]
                    }

            return _Resp()

        monkeypatch.setattr(provider._client, "post", fake_post)
        result = provider.summarize("Текст", instructions="Мой промпт.")
        assert result["tl_dr"] == "Свободный текст саммари."
        assert result["custom"] is True
        assert seen["response_format"] is None
        user = seen["messages"][1]["content"]
        assert "Мой промпт." in user and "Текст" in user


class TestParseJsonStrict:
    def test_tolerates_fences_and_preamble(self) -> None:
        raw = 'Вот результат:\n```json\n{"tl_dr": "ok"}\n```'
        assert _parse_json_strict(raw) == {"tl_dr": "ok"}

    def test_raises_on_garbage(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_json_strict("модель вернула вовсе не JSON")


# -------------------------------------------------------------------------- #
# Защита от невалидного JSON модели (T-lite рвёт ключи: "time  code")
# -------------------------------------------------------------------------- #
class TestNormalizeSummary:
    def test_broken_time_code_key_is_repaired(self) -> None:
        # Реальный сбой T-lite: ключ "time  code" (двойной пробел) валиден для
        # json, но терялся в UI.  Нормализация возвращает каноническое имя.
        raw = {
            "tl_dr": "ок",
            "decisions": [{"formulation": "решение", "time  code": "05:10"}],
            "action_items": [
                {"owner": "Иван", "deadline": "пт", "formulation": "задача", "time_code": "07:00"}
            ],
            "quotes": [{"quote": "цитата", "time  code": "09:00", "speaker": "Спикер 1"}],
        }
        result = _normalize_summary(raw)
        assert result["decisions"][0]["timecode"] == "05:10"
        assert result["action_items"][0]["timecode"] == "07:00"
        assert result["quotes"][0]["timecode"] == "09:00"

    def test_aliases_map_to_canonical_keys(self) -> None:
        raw = {
            "summary": "кратко",
            "open_questions": ["вопрос"],
            "tasks": [{"formulation": "задача", "timecode": "01:00"}],
        }
        result = _normalize_summary(raw)
        assert result["tl_dr"] == "кратко"
        assert result["questions_open"] == ["вопрос"]
        assert result["action_items"][0]["formulation"] == "задача"

    def test_string_instead_of_array(self) -> None:
        result = _normalize_summary({"risks": "один риск"})
        assert result["risks"] == ["один риск"]

    def test_always_has_all_schema_keys(self) -> None:
        result = _normalize_summary({"tl_dr": "только текст"})
        for key in (
            "tl_dr", "topics", "decisions", "action_items",
            "questions_open", "risks", "quotes",
        ):
            assert key in result

    def test_unknown_keys_dropped(self) -> None:
        result = _normalize_summary({"tl_dr": "текст", "hallucination": "лишнее"})
        assert "hallucination" not in result


class TestResponseFormat:
    def test_response_format_carries_schema(self) -> None:
        rf = _summary_response_format()
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["schema"] is SUMMARY_JSON_SCHEMA
        # Все ключи схемы обязательны — грамматика не даст их пропустить.
        assert set(SUMMARY_JSON_SCHEMA["required"]) == {
            "tl_dr", "topics", "decisions", "action_items",
            "questions_open", "risks", "quotes",
        }

    def test_llamacpp_sends_response_format(self, monkeypatch) -> None:
        provider = LlamaCppProvider(model="whatever.gguf")
        captured: dict[str, object] = {}

        def fake_post(url, *, json=None, **_kwargs):
            captured["payload"] = json

            class _Resp:
                status_code = 200

                def json(self) -> dict[str, object]:
                    import json as _json

                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": _json.dumps(
                                        {
                                            "tl_dr": "ок",
                                            "topics": [],
                                            "decisions": [],
                                            "action_items": [],
                                            "questions_open": [],
                                            "risks": [],
                                            "quotes": [],
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    }

            return _Resp()

        monkeypatch.setattr(provider._client, "post", fake_post)
        provider._ensure_running = lambda: None  # type: ignore[method-assign]
        result = provider.summarize("Текст")
        assert result["tl_dr"] == "ок"
        payload = captured["payload"]
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["schema"]["type"] == "object"

    def test_llamacpp_invalid_json_raises_llmerror_with_excerpt(
        self, monkeypatch
    ) -> None:
        provider = LlamaCppProvider(model="whatever.gguf")

        def fake_post(url, *, json=None, **_kwargs):
            class _Resp:
                status_code = 200

                def json(self) -> dict[str, object]:
                    return {
                        "choices": [{"message": {"content": "совсем не JSON ответ"}}]
                    }

            return _Resp()

        monkeypatch.setattr(provider._client, "post", fake_post)
        provider._ensure_running = lambda: None  # type: ignore[method-assign]
        with pytest.raises(LLMError, match="невалидный JSON"):
            provider.summarize("Текст")

    def test_external_retries_without_response_format_on_400(self, monkeypatch) -> None:
        provider = ExternalProvider(
            base_url="http://127.0.0.1:11434", model="old-ollama"
        )
        calls: list[dict[str, object]] = []

        def fake_post(url, *, json=None, headers=None, **_kwargs):
            # Копия: _chat мутирует payload (pop response_format) между вызовами.
            calls.append(dict(json))
            if "response_format" in json:
                # Старый сервер не знает json_schema.
                class _Bad:
                    status_code = 400

                    text = "response_format not supported"

                    def json(self):  # pragma: no cover - not reached
                        return {}

                return _Bad()

            class _Resp:
                status_code = 200

                def json(self) -> dict[str, object]:
                    import json as _json

                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": _json.dumps(
                                        {
                                            "tl_dr": "ок",
                                            "topics": [],
                                            "decisions": [],
                                            "action_items": [],
                                            "questions_open": [],
                                            "risks": [],
                                            "quotes": [],
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    }

            return _Resp()

        monkeypatch.setattr(provider._client, "post", fake_post)
        result = provider.summarize("Текст")
        assert result["tl_dr"] == "ок"
        assert len(calls) == 2
        assert "response_format" in calls[0]
        assert "response_format" not in calls[1]

    def test_external_retries_once_on_invalid_json(self, monkeypatch) -> None:
        provider = ExternalProvider(
            base_url="http://127.0.0.1:11434", model="gemma3:4b"
        )
        attempts: list[list[dict[str, str]]] = []

        def fake_post(url, *, json=None, headers=None, **_kwargs):
            import json as _json

            attempts.append(json["messages"])
            if len(attempts) == 1:

                class _Broken:
                    status_code = 200

                    def json(self) -> dict[str, object]:
                        return {
                            "choices": [
                                {"message": {"content": "Список решений: да; нет."}}
                            ]
                        }

                return _Broken()

            class _Fixed:
                status_code = 200

                def json(self) -> dict[str, object]:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": _json.dumps(
                                        {
                                            "tl_dr": "исправлено",
                                            "topics": [],
                                            "decisions": [],
                                            "action_items": [],
                                            "questions_open": [],
                                            "risks": [],
                                            "quotes": [],
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    }

            return _Fixed()

        monkeypatch.setattr(provider._client, "post", fake_post)
        result = provider.summarize("Текст")
        assert result["tl_dr"] == "исправлено"
        assert len(attempts) == 2
        # Повтор содержит поправку и черновик предыдущего ответа.
        assert attempts[1][-1]["role"] == "user"
        assert "не является валидным JSON" in attempts[1][-1]["content"]


class TestMapPhaseSalvage:
    def test_map_block_failure_does_not_kill_stage(self) -> None:
        # Фрагмент с ошибкой превращается в сырой текст, reduce всё равно
        # собирает итоговое саммари.
        calls: list[str] = []

        class _FlakyProvider(_FakeProvider):
            def summarize(
                self,
                text: str,
                *,
                language: str = "ru",
                instructions: str = "",
                freeform: bool | None = None,
            ) -> dict[str, object]:
                calls.append(text)
                if len(calls) == 2:
                    raise LLMError("модель вернула невалидный JSON: Expecting ':'")
                return super().summarize(
                    text, language=language, instructions=instructions, freeform=freeform
                )

        summarizer = Summarizer(_FlakyProvider())
        long_text = ". ".join(f"Предложение номер {i} про проект" for i in range(400))
        result = summarizer.summarize(long_text)
        assert "tl_dr" in result
        # Текст упавшего фрагмента дошёл до reduce сырым текстом (fallback).
        assert any("Предложение номер" in call for call in calls)


# -------------------------------------------------------------------------- #
# GPU (Vulkan): команда llama-server и откат на CPU
# -------------------------------------------------------------------------- #


class TestServerCmdGpuFlags:
    def test_cpu_device_has_no_gpu_flags(self) -> None:
        cmd = _server_cmd(
            binary="llama-server.exe",
            model="model.gguf",
            ctx_size=4096,
            threads=8,
            port=8765,
            device="cpu",
            gpu_device="Vulkan0",
            gpu_layers=99,
        )
        joined = " ".join(cmd)
        assert "--n-gpu-layers" not in joined
        assert "--device" not in joined
        assert cmd[-1] == "8765"

    def test_vulkan_device_adds_gpu_layers_and_adapter(self) -> None:
        cmd = _server_cmd(
            binary="llama-server.exe",
            model="model.gguf",
            ctx_size=4096,
            threads=8,
            port=8765,
            device="vulkan",
            gpu_device="Vulkan0",
            gpu_layers=33,
        )
        joined = " ".join(cmd)
        assert "--n-gpu-layers 33" in joined
        assert "--device Vulkan0" in joined

    def test_vulkan_without_adapter_omits_device_flag(self) -> None:
        cmd = _server_cmd(
            binary="llama-server.exe",
            model="model.gguf",
            ctx_size=4096,
            threads=8,
            port=8765,
            device="vulkan",
            gpu_device="  ",
            gpu_layers=99,
        )
        joined = " ".join(cmd)
        assert "--n-gpu-layers 99" in joined
        assert "--device" not in joined

    def test_gpu_layers_clamped_to_bounds(self) -> None:
        cmd = _server_cmd(
            binary="b",
            model="m",
            ctx_size=4096,
            threads=1,
            port=1,
            device="vulkan",
            gpu_layers=1000,
        )
        assert "--n-gpu-layers" in cmd
        assert str(cmd[cmd.index("--n-gpu-layers") + 1]) == "99"


class TestVulkanFallbackToCpu:
    def test_failed_vulkan_start_falls_back_to_cpu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = tmp_path / "model.gguf"
        model.write_bytes(b"gguf")
        provider = LlamaCppProvider(
            model=model, device="vulkan", gpu_device="Vulkan0"
        )
        started_with: list[dict[str, object]] = []

        def fake_start() -> None:
            started_with.append({"device": provider.device, "gpu": provider.gpu_device})
            if provider.device == "vulkan":
                raise LLMError("Vulkan runtime startup failed")

        monkeypatch.setattr(provider, "_start_server", fake_start)
        monkeypatch.setattr(
            provider, "_resolve_binary", lambda: "llama-server.exe"
        )
        provider._ensure_running()

        assert [s["device"] for s in started_with] == ["vulkan", "cpu"]
        # После отката адаптер сброшен: CPU-сборка его не понимает.
        assert provider.device == "cpu"
        assert provider.gpu_device == ""

    def test_cpu_failure_is_not_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = tmp_path / "model.gguf"
        model.write_bytes(b"gguf")
        provider = LlamaCppProvider(model=model, device="cpu")
        attempts: list[str] = []

        def fake_start() -> None:
            attempts.append(provider.device)
            raise LLMError("llama-server не стал доступен")

        monkeypatch.setattr(provider, "_start_server", fake_start)
        with pytest.raises(LLMError):
            provider._ensure_running()
        assert attempts == ["cpu"]

    def test_vulkan_fallback_restores_device_after_total_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = tmp_path / "model.gguf"
        model.write_bytes(b"gguf")
        provider = LlamaCppProvider(
            model=model, device="vulkan", gpu_device="Vulkan0"
        )

        def fake_start() -> None:
            raise LLMError("no luck")

        monkeypatch.setattr(provider, "_start_server", fake_start)
        with pytest.raises(LLMError, match=r"no luck.+CPU"):
            provider._ensure_running()
        # Настройки восстановлены: следующая задача снова попробует Vulkan
        # с выбранным пользователем адаптером.
        assert provider.device == "vulkan"
        assert provider.gpu_device == "Vulkan0"
