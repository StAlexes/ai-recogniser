"""Export recognition results: TXT, SRT, VTT, Markdown, JSON.

All formats are built from VAD segments (start_s / end_s / text).  If
diarization is enabled, each segment additionally contains ``speaker`` —
the participant identifier ("Спикер 1", "Спикер 2", etc.).  # noqa: RUF002
"""

from __future__ import annotations

import json
from typing import Any


def _format_timestamp_srt(seconds: float) -> str:
    """Format as SRT timestamp: HH:MM:SS,mmm"""
    total_ms = round(max(seconds, 0.0) * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format as VTT timestamp: HH:MM:SS.mmm"""
    total_ms = round(max(seconds, 0.0) * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _speaker_label(segment: dict[str, Any]) -> str:
    speaker = segment.get("speaker")
    if speaker is None:
        return ""
    return f" [{speaker}]"


def _format_clock(seconds: float) -> str:
    total = max(round(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# Соседние реплики одного спикера с зазором не больше этого склеиваются в одну
# строку с общим интервалом: пауза внутри монолога (VAD режет речь по коротким
# паузам, стадия диаризации склеивает куски только при зазоре до 0,5 с) должна
# читаться как одна реплика.  Длинная пауза (на реальных записях — 6–11 с)
# остаётся видимой: строки не склеиваются, и скачок между интервалами показывает
# молчание.  На 29-минутной записи порог 5 с склеивает 98% соседних пар.
SPEAKER_LINE_MERGE_GAP_S = 5.0


def _speaker_interval(start: float, end: float) -> str:
    begin = _format_clock(start)
    finish = _format_clock(end)
    if finish == begin:
        return begin
    return f"{begin}–{finish}"


def export_txt(segments: list[dict[str, Any]]) -> str:
    """Plain text.

    Without diarization the segments are joined into one paragraph.  When
    speakers are present, each line carries a time interval and the speaker
    label, and consecutive lines of the same speaker separated by a short
    pause are merged into one line with a shared interval — otherwise a
    dialogue transcript falls apart into dozens of fragment lines.
    """
    parts = [
        (segment, segment.get("text", "").strip())
        for segment in segments
        if segment.get("text", "").strip()
    ]
    if not any(segment.get("speaker") for segment, _ in parts):
        return " ".join(text for _, text in parts).strip()

    lines: list[str] = []
    group: dict[str, Any] | None = None  # {speaker, start, end, texts}

    def flush() -> None:
        if group is None:
            return
        interval = _speaker_interval(group["start"], group["end"])
        speaker = group["speaker"]
        prefix = f"[{interval}] {speaker}: " if speaker else f"[{interval}] "
        lines.append(prefix + " ".join(group["texts"]))

    for segment, text in parts:
        speaker = segment.get("speaker")
        start = float(segment.get("start_s", 0.0))
        end = max(float(segment.get("end_s", start)), start)
        if (
            group is not None
            and speaker
            and group["speaker"] == speaker
            and start - group["end"] < SPEAKER_LINE_MERGE_GAP_S
        ):
            group["texts"].append(text)
            group["end"] = end
            continue
        flush()
        group = {"speaker": speaker, "start": start, "end": end, "texts": [text]}
    flush()
    return "\n".join(lines).strip()


def export_srt(segments: list[dict[str, Any]]) -> str:
    """SubRip (.srt) с тайм-кодами и optional спикерами."""
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = _format_timestamp_srt(segment.get("start_s", 0.0))
        end = _format_timestamp_srt(segment.get("end_s", segment.get("start_s", 0.0)))
        text = segment.get("text", "").strip()
        if not text:
            continue
        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(text + _speaker_label(segment))
        lines.append("")
    return "\n".join(lines).strip()


def export_vtt(segments: list[dict[str, Any]]) -> str:
    """WebVTT (.vtt) с тайм-кодами и optional спикерами."""
    lines: list[str] = ["WEBVTT", ""]
    for segment in segments:
        start = _format_timestamp_vtt(segment.get("start_s", 0.0))
        end = _format_timestamp_vtt(segment.get("end_s", segment.get("start_s", 0.0)))
        text = segment.get("text", "").strip()
        if not text:
            continue
        lines.append(f"{start} --> {end}")
        lines.append(text + _speaker_label(segment))
        lines.append("")
    return "\n".join(lines).strip()


def export_markdown(segments: list[dict[str, Any]]) -> str:
    """Markdown — каждый сегмент как блок со временем и optional спикером."""
    lines: list[str] = ["# Расшифровка", ""]
    for segment in segments:
        start = _format_timestamp_vtt(segment.get("start_s", 0.0))
        end = _format_timestamp_vtt(segment.get("end_s", segment.get("start_s", 0.0)))
        text = segment.get("text", "").strip()
        if not text:
            continue
        speaker = _speaker_label(segment)
        lines.append(f"### [{start} → {end}]{speaker}")
        lines.append("")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def export_json(segments: list[dict[str, Any]], *, metadata: dict[str, Any] | None = None) -> str:
    """JSON — полная структура с сегментами и опциональными метаданными."""
    payload: dict[str, Any] = {
        "segments": segments,
        "segment_count": len(segments),
    }
    if metadata is not None:
        payload["metadata"] = metadata
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _summary_items(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _summary_item_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("formulation") or item.get("quote") or "").strip()
    return str(item).strip()


def format_summary(summary: Any, *, title: str | None = None) -> str:
    """Human-readable text of a structured summary (summary.md and the UI).

    Mirrors the structure produced by the LLM (tl_dr, topics, decisions,
    action items, ...); tolerates missing keys and non-dict payloads.
    Свободный текст кастомного промпта (``custom: True``) возвращается как
    есть — пользователь сам задаёт оформление.
    """
    if not isinstance(summary, dict):
        return str(summary).strip()
    if summary.get("custom") and isinstance(summary.get("tl_dr"), str):
        text = summary["tl_dr"].strip()
        if title:
            return f"# Саммари — {title}\n\n{text}"
        return text
    lines: list[str] = []
    if title:
        lines += [f"# Саммари — {title}", ""]
    tldr = str(summary.get("tl_dr") or "").strip()
    if tldr:
        lines += ["## Кратко", tldr, ""]
    topics = [str(t).strip() for t in _summary_items(summary.get("topics")) if str(t).strip()]
    if topics:
        lines += ["## Темы", *[f"- {topic}" for topic in topics], ""]
    decisions: list[str] = []
    for item in _summary_items(summary.get("decisions")):
        text = _summary_item_text(item)
        if not text:
            continue
        timecode = item.get("timecode") if isinstance(item, dict) else None
        decisions.append(f"- {text}" + (f" [{timecode}]" if timecode else ""))
    if decisions:
        lines += ["## Решения", *decisions, ""]
    actions: list[str] = []
    for item in _summary_items(summary.get("action_items")):
        text = _summary_item_text(item)
        if not text:
            continue
        if not isinstance(item, dict):
            actions.append(f"- {text}")
            continue
        parts = [text]
        for key in ("owner", "deadline"):
            detail = str(item.get(key) or "").strip()
            if detail:
                parts.append(detail)
        line = "- " + " — ".join(parts)
        timecode = str(item.get("timecode") or "").strip()
        if timecode:
            line += f" [{timecode}]"
        actions.append(line)
    if actions:
        lines += ["## Задачи", *actions, ""]
    questions = [
        str(q).strip() for q in _summary_items(summary.get("questions_open")) if str(q).strip()
    ]
    if questions:
        lines += ["## Открытые вопросы", *[f"- {q}" for q in questions], ""]
    risks = [str(r).strip() for r in _summary_items(summary.get("risks")) if str(r).strip()]
    if risks:
        lines += ["## Риски", *[f"- {risk}" for risk in risks], ""]
    quotes: list[str] = []
    for item in _summary_items(summary.get("quotes")):
        text = _summary_item_text(item)
        if not text:
            continue
        line = f"- «{text}»"
        if isinstance(item, dict):
            speaker = str(item.get("speaker") or "").strip()
            timecode = str(item.get("timecode") or "").strip()
            if speaker:
                line += f" — {speaker}"
            if timecode:
                line += f" [{timecode}]"
        quotes.append(line)
    if quotes:
        lines += ["## Цитаты", *quotes, ""]
    return "\n".join(lines).strip()


FORMATS: dict[str, callable] = {
    "txt": export_txt,
    "srt": export_srt,
    "vtt": export_vtt,
    "md": export_markdown,
    "json": export_json,
}

EXPORT_EXTENSIONS = {
    "txt": ".txt",
    "srt": ".srt",
    "vtt": ".vtt",
    "md": ".md",
    "json": ".json",
}

EXPORT_CONTENT_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
    "json": "application/json; charset=utf-8",
}

EXPORT_DOWNLOAD_NAMES = {
    "txt": "transcript.txt",
    "srt": "transcript.srt",
    "vtt": "transcript.vtt",
    "md": "transcript.md",
    "json": "transcript.json",
}


def export(segments: list[dict[str, Any]], format: str, *, metadata: dict[str, Any] | None = None) -> str:
    """Export segments in *format*.  Raises ValueError for unknown format."""
    if format not in FORMATS:
        raise ValueError(f"Неизвестный формат экспорта: {format}. Доступно: {', '.join(FORMATS)}")
    return FORMATS[format](segments, metadata=metadata) if format == "json" else FORMATS[format](segments)
