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


def export_txt(segments: list[dict[str, Any]]) -> str:
    """Plain text — сегменты склеиваются через пробел."""
    parts = [segment.get("text", "").strip() for segment in segments if segment.get("text", "").strip()]
    return " ".join(parts).strip()


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
