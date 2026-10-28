from __future__ import annotations

from pathlib import Path

from shell.main import handle_protocol_uri, take_navigation

from airec.core.paths import AppPaths
from airec.integrations.windows import _command


def paths_for(tmp_path: Path) -> AppPaths:
    paths = AppPaths(
        root=tmp_path,
        data=tmp_path,
        models=tmp_path / "models",
        cache=tmp_path / "cache",
        logs=tmp_path / "logs",
        results=tmp_path / "results",
        temp=tmp_path / "temp",
        bin=tmp_path / "bin",
        ui=tmp_path / "ui-dist",
    )
    paths.ensure()
    return paths


def test_context_menu_command_quotes_file_placeholder() -> None:
    assert _command(file_arg=True).endswith(' "%1"')
    assert _command(protocol=True).endswith(' --protocol "%1"')


def test_result_protocol_writes_pending_navigation(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)

    job_id, handled = handle_protocol_uri(paths, "airec://result/abc-123")

    assert job_id == "abc-123"
    assert handled is False
    assert take_navigation(paths) == "abc-123"
    assert take_navigation(paths) is None


def test_folder_protocol_opens_existing_folder(
    tmp_path: Path, monkeypatch
) -> None:
    paths = paths_for(tmp_path)
    (paths.results / "abc-123").mkdir(parents=True)
    opened: list[Path] = []
    monkeypatch.setattr(
        "shell.main.open_result_folder",
        lambda _paths, job_id: opened.append(_paths.results / job_id) or True,
    )

    job_id, handled = handle_protocol_uri(paths, "airec://folder/abc-123")

    assert job_id == "abc-123"
    assert handled is True
    assert opened == [paths.results / "abc-123"]
    assert take_navigation(paths) is None
