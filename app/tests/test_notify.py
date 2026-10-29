from __future__ import annotations

from airec.notify.windows_toast import _queued_toast_xml


def test_queued_toast_uses_cyrillic_and_count() -> None:
    xml = _queued_toast_xml(["встреча.wav", "second.wav"], 2)

    assert "Взят в работу" in xml
    assert "встреча.wav" in xml
    assert "и ещё 1" in xml
    assert "<actions>" not in xml


def test_queued_toast_escapes_filename() -> None:
    xml = _queued_toast_xml(["a&b.wav"], 1)

    assert "a&amp;b.wav" in xml
