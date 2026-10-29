from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from airec.api import llm as llm_api
from airec.core.config import Settings


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

    def fake_find(_paths: object, custom: str | None = None) -> Path | None:
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
