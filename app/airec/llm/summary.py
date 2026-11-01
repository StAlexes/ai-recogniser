"""Map-reduce summarisation and Q&A over transcripts.

The transcript (possibly long) is split into blocks of ~N tokens; each block is
summarised independently (map), then the block summaries are merged into one
final structured summary (reduce).  This keeps context within the model's
context window even for hour-long recordings.

The user-defined summary prompt (``Settings.summary_prompt``) replaces the
built-in task and unlocks free-form output: the model may use sections, lists
and tables as the user's instructions demand.  The built-in prompt keeps the
structured JSON schema (``custom`` flag marks free-form results).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from airec.llm.providers import SUMMARY_SCHEMA, LLMError, LLMProvider

log = logging.getLogger(__name__)

# Kept for backwards compatibility: the canonical schema now lives in
# providers.py next to the prompt builder that uses it.
__all__ = ["MAX_BLOCK_CHARS", "SUMMARY_SCHEMA", "Summarizer"]

MAX_BLOCK_CHARS = 6000  # rough proxy for token budget per block

# Русский текст в среднем ~2.0–2.5 символа на токен; берём консервативные
# 2.0, чтобы промпт гарантированно влезал в контекст вместе с ответом.
CHARS_PER_TOKEN = 2.0
OUTPUT_RESERVE_TOKENS = 700
MIN_BLOCK_CHARS = 1200


def _chars_budget(ctx_size: int) -> int:
    """Character budget for a whole prompt under *ctx_size* tokens."""
    return max(MIN_BLOCK_CHARS, int((ctx_size - OUTPUT_RESERVE_TOKENS) * CHARS_PER_TOKEN))


def _map_scaffold_len(instructions: str) -> int:
    """Length of the fixed part of a map prompt (everything but the block)."""
    if instructions.strip():
        sample = _custom_block_prompt(instructions, "", 0, 1)
    else:
        sample = _block_summary_prompt("", 0, 1)
    return len(sample)


def _split_into_blocks(text: str, max_chars: int = MAX_BLOCK_CHARS) -> list[str]:
    """Split *text* into blocks of at most *max_chars* characters on sentence boundaries."""
    if not text or len(text) <= max_chars:
        return [text] if text else []

    blocks: list[str] = []
    current = ""
    # Naive sentence split on Russian sentence-final punctuation.
    for part in text.replace("?", ".").replace("!", ".").split("."):
        part = part.strip()
        if not part:
            continue
        candidate = (current + ". " + part).strip() if current else part
        if len(candidate) > max_chars and current:
            blocks.append(current)
            current = part
        else:
            current = candidate
    if current:
        blocks.append(current)
    return blocks


def _block_summary_prompt(block: str, index: int, total: int) -> str:
    return (
        f"Фрагмент {index + 1} из {total} транскрибированной встречи.\n"
        "Составь краткое содержание фрагмента: ключевые факты, решения, задачи, "
        "вопросы и риски. Пиши на русском языке, не выдумывай фактов.\n\n"
        f"Текст фрагмента:\n---\n{block}\n---\n\nВерни краткое резюме фрагмента (абзац)."
    )


def _custom_block_prompt(
    instructions: str, block: str, index: int, total: int
) -> str:
    """Map-prompt for a custom summary prompt: единый стиль всех фрагментов."""
    return (
        f"Фрагмент {index + 1} из {total} транскрибированной встречи.\n\n"
        f"Инструкция пользователя:\n---\n{instructions}\n---\n\n"
        f"Текст фрагмента:\n---\n{block}\n---\n\n"
        "Обработай фрагмент по инструкции. Ответ — только результат, без "
        "преамбулы."
    )


def _reduce_prompt(
    block_summaries: list[str], *, instructions: str = ""
) -> str:
    joined = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(block_summaries))
    if instructions.strip():
        return (
            "Ниже результаты обработки фрагментов одной встречи "
            "(порядковый номер фрагмента указан в квадратных скобках).\n\n"
            f"Инструкция пользователя:\n---\n{instructions.strip()}\n---\n\n"
            f"Результаты фрагментов:\n---\n{joined}\n---\n\n"
            "Объедини их в единый итоговый ответ по инструкции пользователя. "
            "Таймкоды фрагментов при необходимости замени на их примерное "
            "время от начала встречи. Ответ — только итоговый результат."
        )
    return (
        "Ниже несколько кратких резюме фрагментов одной и той же встречи. "
        "Сложи их в одно структурированное саммари всей встречи. "
        "Пиши на русском языке. Не выдумывай фактов.\n"
        "Каждое решение и каждую задачу привяжи к примерному таймкоду "
        "(порядковый номер фрагмента, где оно упомянуто).\n\n"
        f"Резюме фрагментов:\n---\n{joined}\n---\n\n"
        f"Верни строго JSON со схемой:\n{SUMMARY_SCHEMA}"
    )


class Summarizer:
    """Map-reduce summarisation over a transcript using an LLM provider."""

    def __init__(self, provider: LLMProvider, *, instructions: str = "") -> None:
        self.provider = provider
        # Кастомный промпт читается из настроек при каждом запуске саммари
        # (api/llm.py создаёт Summarizer заново), поэтому правки применяются
        # без перезапуска приложения.
        self.instructions = (instructions or "").strip()

    def summarize(
        self,
        text: str,
        *,
        language: str = "ru",
        progress: Callable[[float, str], None] | None = None,
        ctx_size: int = 4096,
    ) -> dict[str, Any]:
        """Return structured summary dict for *text*.

        *ctx_size* — контекст модели (настройка ``llm_ctx_size``): размер
        блока и иерархия reduce подстраиваются под него.
        """
        notify = progress or (lambda _value, _phase: None)
        custom = bool(self.instructions)
        # Бюджет блока: общий бюджет минус «каркас» map-промпта (инструкция
        # пользователя вшивается в каждый фрагмент).
        block_chars = min(
            MAX_BLOCK_CHARS,
            max(MIN_BLOCK_CHARS, _chars_budget(ctx_size) - _map_scaffold_len(self.instructions)),
        )
        blocks = _split_into_blocks(text, max_chars=block_chars)
        log.info(
            "summarising %d chars in %d block(s) (block budget %d chars, ctx %d)",
            len(text), len(blocks), block_chars, ctx_size,
        )
        notify(0.05, "Подготовка текста")

        if len(blocks) == 1:
            notify(0.25, "Генерация саммари")
            result = self.provider.summarize(
                text, language=language, instructions=self.instructions
            )
            notify(1.0, "Готово")
            return result

        # Map: обработка фрагментов.  С кастомным промптом каждый фрагмент
        # обрабатывается по инструкции пользователя с её же оформлением;
        # встроенный путь сохраняет прежний фиксированный промпт.
        block_summaries: list[str] = []
        total = len(blocks)
        for index, block in enumerate(blocks):
            notify(0.1 + 0.7 * (index / max(total, 1)), f"Фрагмент {index + 1} из {total}")
            map_text = (
                _custom_block_prompt(self.instructions, block, index, total)
                if custom
                else _block_summary_prompt(block, index, total)
            )
            try:
                summary = self.provider.summarize(
                    map_text,
                    language=language,
                    # Инструкция уже вшита в map-промпт; свободный режим
                    # включается явно, чтобы провайдер не навязывал JSON-схему
                    # (она сломала бы оформление из промпта пользователя).
                    instructions="",
                    freeform=True if custom else None,
                )
            except LLMError as exc:
                # Сбой одного фрагмента не должен ронять всю стадию: на map-
                # шаге провайдеру нужен лишь связный текст резюме.  Грамматика
                # llama-server практически исключает такой сценарий; запасной
                # путь спасает внешних провайдеров без response_format.
                log.warning("map block %d failed (%s); using raw text", index, exc)
                summary = {"tl_dr": block[:1200]}
            # LlamaCppProvider.summarize returns a dict; flatten to text.
            block_text = summary.get("tl_dr", "") if isinstance(summary, dict) else str(summary)
            block_summaries.append(block_text)

        # Reduce: объединение в итоговый ответ.
        notify(0.85, "Объединение фрагментов")
        result = self._reduce(
            block_summaries,
            language=language,
            custom=custom,
            ctx_size=ctx_size,
        )
        notify(1.0, "Готово")
        return result

    def _reduce(
        self,
        block_summaries: list[str],
        *,
        language: str,
        custom: bool,
        ctx_size: int,
    ) -> dict[str, Any]:
        """Merge block summaries into the final result.

        При переполнении контекста слияние идёт иерархически: группы
        промежуточных резюме сливаются в черновики, затем черновики — в
        итог.  Сбой самого последнего вызова не роняет стадию: после
        минут работы лучше выдать черновик, чем ошибку.
        """
        budget = _chars_budget(ctx_size)
        # Reduce-каркас (с инструкциями) на примере одного пустого резюме;
        # содержимое промежуточных групп не должно выталкивать промпт за
        # бюджет: content_cap — предел суммы длин сводок в одной группе.
        scaffold = len(_reduce_prompt([""], instructions=self.instructions))
        content_cap = max(400, budget - scaffold)

        summaries = list(block_summaries)
        for _round in range(6):  # защитный предел иерархии слияний
            joined_len = sum(len(s) + 4 for s in summaries) + scaffold
            if len(summaries) <= 1 or joined_len <= budget:
                break
            max_len = max(len(s) for s in summaries)
            if content_cap < 2 * max_len + 8:
                # Даже пара сводок не влезает: ужимаем их для промежуточного
                # слияния — слитый ответ модели короче входа.
                summaries = [
                    s[: max(200, content_cap // 2 - 4)] for s in summaries
                ]
                continue
            k = max(2, min(len(summaries), content_cap // (max_len + 4)))
            groups = [summaries[i : i + k] for i in range(0, len(summaries), k)]
            log.info(
                "reduce: %d summaries -> %d intermediate merges (k=%d)",
                len(summaries),
                len(groups),
                k,
            )
            merged = [
                self._merge_group(
                    group, language=language, custom=custom
                ).get("tl_dr", "")
                for group in groups
            ]
            # Гарантия прогресса: результат раунда обязан быть короче входа,
            # иначе иерархия не сойдётся (сбой merge вернул сырой текст).
            if sum(len(m) for m in merged) >= sum(len(s) for s in summaries):
                limit = max(200, content_cap // len(groups) - 4)
                merged = [m[:limit] for m in merged]
            summaries = merged

        try:
            return self.provider.summarize(
                _reduce_prompt(summaries, instructions=self.instructions),
                language=language,
                # Промпт уже содержит инструкцию и формат ответа; freeform=True
                # удерживает свободный режим и для встроенного reduce-текста.
                instructions="",
                freeform=True if custom else None,
            )
        except LLMError as exc:
            # Финальный сбой после всех map-работ: отдаём лучший черновик.
            if len(block_summaries) <= 1:
                raise
            log.warning("final reduce failed (%s); salvaging drafts", exc)
            joined = "\n\n".join(
                f"[{i + 1}] {s}" for i, s in enumerate(block_summaries)
            )
            return {
                "tl_dr": (
                    "Саммари собрано из фрагментов (итоговое объединение не "
                    f"удалось: {exc}):\n\n{joined}"
                ),
                "custom": custom,
            }

    def _merge_group(
        self,
        group: list[str],
        *,
        language: str,
        custom: bool,
    ) -> dict[str, Any]:
        try:
            return self.provider.summarize(
                _reduce_prompt(group, instructions=self.instructions),
                language=language,
                instructions="",
                freeform=True if custom else None,
            )
        except LLMError as exc:
            log.warning("intermediate merge failed (%s); keeping raw", exc)
            return {"tl_dr": "\n\n".join(group)}

    def ask(self, question: str, *, context: str, language: str = "ru") -> str:
        """Answer *question* using *context* (the full transcript)."""
        return self.provider.ask(question, context=context, language=language)
