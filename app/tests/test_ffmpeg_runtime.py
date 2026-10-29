from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import httpx
import pytest

from airec.asr import ffmpeg_runtime
from airec.asr.manager import JobManager
from airec.core.config import Settings
from airec.core.paths import AppPaths


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


def ffmpeg_zip() -> bytes:
    """A zip laid out like BtbN's FFmpeg-Builds archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        root = "ffmpeg-n8.1.3-25-g611cec792a-win64-lgpl-shared-8.1"
        archive.writestr(f"{root}/bin/ffmpeg.exe", b"exe")
        archive.writestr(f"{root}/bin/avcodec-62.dll", b"dll")
        archive.writestr(f"{root}/bin/avutil-60.dll", b"dll")
        archive.writestr(f"{root}/include/libavutil/avutil.h", b"header")
        archive.writestr(f"{root}/README.txt", b"readme")
    return buffer.getvalue()


def _no_path_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg_runtime.shutil, "which", lambda _name: None)


def test_install_ffmpeg_zip_extracts_only_bin(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)
    archive = tmp_path / "ffmpeg.zip"
    archive.write_bytes(ffmpeg_zip())

    found = ffmpeg_runtime.install_ffmpeg_zip(archive, paths)

    assert found.name == "ffmpeg.exe"
    assert found.parent == ffmpeg_runtime.runtime_dir(paths)
    assert (found.parent / "avcodec-62.dll").is_file()
    # Headers and docs from the archive are not extracted.
    assert not (found.parent / "avutil.h").exists()
    assert not (found.parent / "README.txt").exists()
    assert ffmpeg_runtime.find_ffmpeg(paths) == found


def test_find_ffmpeg_custom_file(tmp_path: Path) -> None:
    custom = tmp_path / "custom" / "ffmpeg.exe"
    custom.parent.mkdir()
    custom.write_bytes(b"exe")
    assert ffmpeg_runtime.find_ffmpeg(custom=custom) == custom


def test_ensure_ffmpeg_downloads_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_path_ffmpeg(monkeypatch)
    paths = paths_for(tmp_path)
    payload = ffmpeg_zip()
    entry = {
        "id": ffmpeg_runtime.FFMPEG_ID,
        "url": "https://example.test/ffmpeg.zip",
        "filename": "ffmpeg.zip",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
    }
    monkeypatch.setattr(ffmpeg_runtime, "_runtime_entry", lambda: entry)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("range") is None
        return httpx.Response(
            200,
            content=payload,
            headers={"content-length": str(len(payload))},
        )

    progress: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    found = ffmpeg_runtime.ensure_ffmpeg(
        paths, progress=lambda value, _phase: progress.append(value), client=client
    )

    assert found.name == "ffmpeg.exe"
    assert progress[-1] == 1.0
    assert not (paths.temp / "ffmpeg.zip.partial").exists()


def test_ensure_ffmpeg_rejects_bad_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_path_ffmpeg(monkeypatch)
    paths = paths_for(tmp_path)
    payload = ffmpeg_zip()
    entry = {
        "id": ffmpeg_runtime.FFMPEG_ID,
        "url": "https://example.test/ffmpeg.zip",
        "filename": "ffmpeg.zip",
        "size_bytes": len(payload),
        "sha256": "00" * 32,
    }
    monkeypatch.setattr(ffmpeg_runtime, "_runtime_entry", lambda: entry)

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=payload,
                headers={"content-length": str(len(payload))},
            )
        )
    )
    with pytest.raises(ffmpeg_runtime.FfmpegInstallError):
        ffmpeg_runtime.ensure_ffmpeg(paths, client=client)
    assert not (paths.temp / "ffmpeg.zip.partial").exists()


def test_ensure_ffmpeg_reuses_complete_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_path_ffmpeg(monkeypatch)
    paths = paths_for(tmp_path)
    payload = ffmpeg_zip()
    entry = {
        "id": ffmpeg_runtime.FFMPEG_ID,
        "url": "https://example.test/ffmpeg.zip",
        "filename": "ffmpeg.zip",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
    }
    monkeypatch.setattr(ffmpeg_runtime, "_runtime_entry", lambda: entry)
    (paths.temp / "ffmpeg.zip.partial").write_bytes(payload)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected download request: {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    found = ffmpeg_runtime.ensure_ffmpeg(paths, client=client)

    assert found.name == "ffmpeg.exe"


def test_resolve_ffmpeg_skips_download_for_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_path_ffmpeg(monkeypatch)
    paths = paths_for(tmp_path)
    manager = JobManager(paths, lambda: Settings())
    called = False

    def fail(*_args, **_kwargs):  # pragma: no cover - must not run
        nonlocal called
        called = True
        raise AssertionError("ensure_ffmpeg must not run for WAV input")

    monkeypatch.setattr(ffmpeg_runtime, "ensure_ffmpeg", fail)
    monkeypatch.setattr(
        ffmpeg_runtime, "find_ffmpeg", lambda *_args, **_kwargs: None
    )

    assert manager._resolve_ffmpeg(Path("voice.wav")) is None
    assert called is False


def test_resolve_ffmpeg_downloads_for_compressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_path_ffmpeg(monkeypatch)
    paths = paths_for(tmp_path)
    manager = JobManager(paths, lambda: Settings())
    installed = paths.data / "runtime" / "ffmpeg" / "ffmpeg.exe"

    def fake_ensure(_paths, **_kwargs):
        installed.parent.mkdir(parents=True, exist_ok=True)
        installed.write_bytes(b"exe")
        return installed

    monkeypatch.setattr(ffmpeg_runtime, "find_ffmpeg", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ffmpeg_runtime, "ensure_ffmpeg", fake_ensure)

    resolved = manager._resolve_ffmpeg(Path("song.mp3"))

    assert resolved == str(installed)
    assert manager._ffmpeg == str(installed)


def test_download_retries_then_falls_back_to_direct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hanging proxy must not look like "no network": retry, then go direct."""
    _no_path_ffmpeg(monkeypatch)
    paths = paths_for(tmp_path)
    payload = ffmpeg_zip()
    entry = {
        "id": ffmpeg_runtime.FFMPEG_ID,
        "url": "https://example.test/ffmpeg.zip",
        "filename": "ffmpeg.zip",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
    }
    monkeypatch.setattr(ffmpeg_runtime, "_runtime_entry", lambda: entry)
    monkeypatch.setattr(ffmpeg_runtime.time, "sleep", lambda _s: None)

    attempts: list[bool] = []

    def fake_single_attempt(
        url: str,
        target: Path,
        *,
        trust_env: bool,
        progress: object,
        cancel: object,
    ) -> None:
        attempts.append(trust_env)
        if len(attempts) < ffmpeg_runtime.DOWNLOAD_ATTEMPTS:
            raise ffmpeg_runtime.FfmpegInstallError(
                "Таймаут соединения: handshake", retryable=True
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    monkeypatch.setattr(ffmpeg_runtime, "_single_attempt", fake_single_attempt)

    found = ffmpeg_runtime.ensure_ffmpeg(paths)

    assert found.name == "ffmpeg.exe"
    # Two attempts through the system proxy, the last one bypassing it.
    assert attempts == [True, True, False]


def test_download_does_not_retry_http_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deterministic failures (HTTP 404) fail on the first attempt."""
    _no_path_ffmpeg(monkeypatch)
    paths = paths_for(tmp_path)
    entry = {
        "id": ffmpeg_runtime.FFMPEG_ID,
        "url": "https://example.test/ffmpeg.zip",
        "filename": "ffmpeg.zip",
    }
    monkeypatch.setattr(ffmpeg_runtime, "_runtime_entry", lambda: entry)
    monkeypatch.setattr(ffmpeg_runtime.time, "sleep", lambda _s: None)

    calls = 0

    def fake_single_attempt(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise ffmpeg_runtime.FfmpegInstallError("Не удалось скачать FFmpeg: HTTP 404")

    monkeypatch.setattr(ffmpeg_runtime, "_single_attempt", fake_single_attempt)

    with pytest.raises(ffmpeg_runtime.FfmpegInstallError) as exc_info:
        ffmpeg_runtime.ensure_ffmpeg(paths)

    assert calls == 1
    assert "HTTP 404" in str(exc_info.value)
