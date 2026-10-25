from __future__ import annotations

from airec.asr.vad import SpeechSegment, join_segments


def test_join_segments_skips_empty_parts() -> None:
    segments = [
        SpeechSegment(0.0, 1.0, "  Привет,  "),
        SpeechSegment(1.0, 2.0, "   "),
        SpeechSegment(2.0, 3.0, "мир."),
    ]

    assert join_segments(segments) == "Привет, мир."


def test_speech_segment_rounding() -> None:
    segment = SpeechSegment(1.23456, 7.89012, "текст")

    assert segment.as_dict() == {
        "start_s": 1.235,
        "end_s": 7.89,
        "text": "текст",
    }
