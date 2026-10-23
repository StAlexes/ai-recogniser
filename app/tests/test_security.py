from __future__ import annotations

from pathlib import Path

from airec.core.security import (
    Session,
    new_token,
    read_session,
    remove_session,
    write_session,
)


def test_session_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    session = Session.create(port=54321, token="secret", version="0.1.0")
    write_session(path, session)

    loaded = read_session(path)

    assert loaded == session
    assert loaded is not None
    assert loaded.port == 54321


def test_corrupt_session_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    path.write_text("not-json", encoding="utf-8")
    assert read_session(path) is None


def test_remove_session_respects_pid(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    write_session(path, Session.create(port=1, token="t", version="0.1.0"))

    remove_session(path, pid=999999)
    assert path.exists()

    remove_session(path, pid=Session.create(port=1, token="t", version="0.1.0").pid)
    assert not path.exists()


def test_token_is_long_and_unique() -> None:
    a, b = new_token(), new_token()
    assert len(a) >= 40
    assert a != b
