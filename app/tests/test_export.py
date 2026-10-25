from __future__ import annotations

from airec.asr.export import (
    export,
    export_json,
    export_markdown,
    export_srt,
    export_txt,
    export_vtt,
)

SEGMENTS = [
    {"start_s": 0.0, "end_s": 8.708, "text": "Привет, как дела?", "speaker": None},
    {"start_s": 9.18, "end_s": 20.036, "text": "Все хорошо, спасибо.", "speaker": "Спикер 1"},
    {"start_s": 20.508, "end_s": 31.364, "text": "А у тебя как?", "speaker": "Спикер 2"},
]


def test_export_txt() -> None:
    result = export_txt(SEGMENTS)
    assert "Привет, как дела?" in result
    assert "Все хорошо, спасибо." in result
    assert "А у тебя как?" in result


def test_export_srt() -> None:
    result = export_srt(SEGMENTS)
    assert "00:00:00,000 --> 00:00:08,708" in result
    assert "Привет, как дела?" in result
    assert "1" in result


def test_export_srt_with_speaker() -> None:
    result = export_srt(SEGMENTS)
    assert "[Спикер 1]" in result
    assert "[Спикер 2]" in result


def test_export_vtt() -> None:
    result = export_vtt(SEGMENTS)
    assert result.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:08,708".replace(",708", ".708") in result


def test_export_markdown() -> None:
    result = export_markdown(SEGMENTS)
    assert "# Расшифровка" in result
    assert "[Спикер 1]" in result


def test_export_json() -> None:
    result = export_json(SEGMENTS, metadata={"model": "gigaam-v3-e2e-ctc"})
    import json

    payload = json.loads(result)
    assert payload["segment_count"] == 3
    assert payload["metadata"]["model"] == "gigaam-v3-e2e-ctc"
    assert payload["segments"][0]["speaker"] is None


def test_export_unknown_format_raises() -> None:
    try:
        export(SEGMENTS, "docx")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "Неизвестный формат экспорта" in str(exc)
    else:
        raise AssertionError("expected ValueError")
