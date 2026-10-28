"""Map-reduce summarisation and Q&A over transcripts.

The transcript (possibly long) is split into blocks of ~N tokens; each block is
summarised independently (map), then the block summaries are merged into one
final structured summary (reduce).  This keeps context within the model's
context window even for hour-long recordings.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from airec.llm.providers import LLMProvider

log = logging.getLogger(__name__)

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

MAX_BLOCK_CHARS = 6000  # rough proxy for token budget per block


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


def _reduce_prompt(block_summaries: list[str]) -> str:
    joined = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(block_summaries))
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

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def summarize(
        self,
        text: str,
        *,
        language: str = "ru",
        progress: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        """Return structured summary dict for *text*."""
        notify = progress or (lambda _value, _phase: None)
        blocks = _split_into_blocks(text)
        log.info("summarising %d chars in %d block(s)", len(text), len(blocks))
        notify(0.05, "Подготовка текста")

        if len(blocks) == 1:
            notify(0.25, "Генерация саммари")
            result = self.provider.summarize(text, language=language)
            notify(1.0, "Готово")
            return result

        # Map: summarise each block.
        block_summaries: list[str] = []
        total = len(blocks)
        for index, block in enumerate(blocks):
            notify(0.1 + 0.7 * (index / max(total, 1)), f"Фрагмент {index + 1} из {total}")
            summary = self.provider.summarize(block, language=language)
            # LlamaCppProvider.summarize returns a dict; flatten to text.
            block_text = summary.get("tl_dr", "") if isinstance(summary, dict) else str(summary)
            block_summaries.append(block_text)

        # Reduce: merge into one structured summary.
        notify(0.85, "Объединение фрагментов")
        result = self.provider.summarize(_reduce_prompt(block_summaries), language=language)
        notify(1.0, "Готово")
        return result

    def ask(self, question: str, *, context: str, language: str = "ru") -> str:
        """Answer *question* using *context* (the full transcript)."""
        return self.provider.ask(question, context=context, language=language)
