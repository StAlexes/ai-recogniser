"""Glossary: deterministic term replacements applied to ASR text.

Pairs ``«as heard» → «correct»`` are applied to the transcript before
summarisation/Q&A so that terms like «битрикс» are normalised to «Битрикс24``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class GlossaryEntry:
    heard: str
    correct: str
    note: str = ""


class Glossary:
    """Case-insensitive deterministic term replacement."""

    def __init__(self, entries: list[GlossaryEntry] | None = None) -> None:
        self._entries: list[GlossaryEntry] = list(entries or [])
        # Sort by length descending so longer terms are replaced first.
        self._entries.sort(key=lambda e: len(e.heard), reverse=True)
        self._patterns: list[tuple[re.Pattern[str], str]] = [
            (re.compile(re.escape(e.heard), flags=re.IGNORECASE), e.correct)
            for e in self._entries
        ]

    @classmethod
    def from_dicts(cls, entries: list[dict[str, str]]) -> Glossary:
        return cls([GlossaryEntry(heard=e["heard"], correct=e["correct"], note=e.get("note", "")) for e in entries])

    def apply(self, text: str) -> str:
        """Replace all glossary terms in *text*.  Case-insensitive."""
        result = text
        for pattern, correct in self._patterns:
            result = pattern.sub(correct, result)
        return result

    def as_dicts(self) -> list[dict[str, str]]:
        return [
            {"heard": e.heard, "correct": e.correct, "note": e.note} for e in self._entries
        ]

    def add(self, heard: str, correct: str, note: str = "") -> None:
        self._entries.append(GlossaryEntry(heard=heard, correct=correct, note=note))
        self._entries.sort(key=lambda e: len(e.heard), reverse=True)
        self._patterns = [
            (re.compile(re.escape(e.heard), flags=re.IGNORECASE), e.correct)
            for e in self._entries
        ]

    def __len__(self) -> int:
        return len(self._entries)
