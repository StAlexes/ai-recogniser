from __future__ import annotations

import hashlib
import io
import time
import zipfile
from pathlib import Path

import httpx
import pytest

from airec.core.paths import AppPaths
from airec.llm import runtime
from airec.models.manager import ModelManager


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


def runtime_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("llama-b11946/llama-server.exe", b"exe")
        archive.writestr("llama-b11946/llama.dll", b"dll")
        archive.writestr("llama-b11946/LICENSE", b"MIT")
    return buffer.getvalue()


def test_default_client_follows_huggingface_redirects(tmp_path: Path) -> None:
    manager = ModelManager(tmp_path)
    assert manager._client.follow_redirects is True


def test_download_fetches_extra_files(tmp_path: Path) -> None:
    payloads = {
        "/v3_e2e_ctc.int8.onnx": b"onnx",
        "/config.json": b"{}",
        "/v3_e2e_ctc_vocab.txt": b"<blk> 0\n",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        payload = payloads[request.url.path]
        return httpx.Response(
            200,
            content=payload,
            headers={"content-length": str(len(payload))},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    manager = ModelManager(tmp_path, client=client)
    manager.download(
        "gigaam",
        "https://example.test/v3_e2e_ctc.int8.onnx",
        filename="v3_e2e_ctc.int8.onnx",
        extra_files=[
            {
                "url": "https://example.test/config.json",
                "filename": "config.json",
            },
            {
                "url": "https://example.test/v3_e2e_ctc_vocab.txt",
                "filename": "v3_e2e_ctc_vocab.txt",
            },
        ],
    )

    target_dir = tmp_path / "gigaam"
    for _ in range(300):
        if (target_dir / "v3_e2e_ctc.int8.onnx").is_file():
            break
        time.sleep(0.01)

    assert (target_dir / "v3_e2e_ctc.int8.onnx").read_bytes() == b"onnx"
    assert (target_dir / "config.json").read_bytes() == b"{}"
    assert (target_dir / "v3_e2e_ctc_vocab.txt").is_file()


def test_config_only_directory_is_not_installed(tmp_path: Path) -> None:
    directory = tmp_path / "incomplete"
    directory.mkdir()
    (directory / "config.json").write_bytes(b"{}")

    manager = ModelManager(tmp_path)

    assert manager.installed_models() == []


def test_download_keeps_manifest_filename(tmp_path: Path) -> None:
    payload = b"onnx-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=payload,
            headers={"content-length": str(len(payload))},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    manager = ModelManager(tmp_path, client=client)
    manager.download(
        "wespeaker-voxceleb-resnet34",
        "https://example.test/voxceleb_resnet34.onnx",
        filename="voxceleb_resnet34.onnx",
    )

    target = tmp_path / "wespeaker-voxceleb-resnet34" / "voxceleb_resnet34.onnx"
    for _ in range(300):
        if target.is_file():
            break
        time.sleep(0.01)

    assert target.read_bytes() == payload
    assert manager.is_installed("wespeaker-voxceleb-resnet34")


def test_download_rejects_wrong_size(tmp_path: Path) -> None:
    payload = b"truncated"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=payload,
            headers={"content-length": str(len(payload))},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    manager = ModelManager(tmp_path, client=client)
    manager.download(
        "qwen",
        "https://example.test/model.gguf",
        filename="model.gguf",
        size_bytes=len(payload) + 1,
    )

    status = None
    for _ in range(300):
        status = manager.download_status("qwen")
        if status is not None and status.get("status") in {"done", "error"}:
            break
        time.sleep(0.01)

    assert status is not None
    assert status["status"] == "error"
    assert "размер" in str(status.get("phase", ""))
    assert not (tmp_path / "qwen" / "model.gguf").exists()


def test_partial_download_is_not_installed(tmp_path: Path) -> None:
    directory = tmp_path / "half"
    directory.mkdir()
    (directory / "model.onnx.partial").write_bytes(b"partial")

    manager = ModelManager(tmp_path)

    assert manager.is_installed("half") is False
    assert manager.installed_models() == []


def test_expected_filename_must_exist(tmp_path: Path) -> None:
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "other.onnx").write_bytes(b"x")
    manager = ModelManager(tmp_path)

    assert manager.is_installed("model") is True
    assert manager.is_installed("model", filename="expected.onnx") is False


def test_download_resumes_partial_file(tmp_path: Path) -> None:
    seen_range: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_range.append(request.headers.get("range"))
        payload = b"def"
        return httpx.Response(
            206,
            content=payload,
            headers={
                "content-length": str(len(payload)),
                "content-range": "bytes 3-5/6",
            },
        )

    model_dir = tmp_path / "resume"
    model_dir.mkdir()
    (model_dir / "model.onnx.partial").write_bytes(b"abc")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    manager = ModelManager(tmp_path, client=client)
    manager.download("resume", "https://example.test/model.onnx", filename="model.onnx")

    target = model_dir / "model.onnx"
    status = None
    for _ in range(300):
        status = manager.download_status("resume")
        if status is not None and status.get("status") in {"done", "error"}:
            break
        time.sleep(0.01)

    assert target.read_bytes() == b"abcdef"
    assert seen_range and seen_range[0] == "bytes=3-"
    assert status is not None
    assert status["status"] == "done"


def test_install_runtime_zip_and_find(tmp_path: Path) -> None:
    paths = paths_for(tmp_path)
    archive = tmp_path / "runtime.zip"
    archive.write_bytes(runtime_zip())

    found = runtime.install_runtime_zip(archive, paths)

    assert found.name == "llama-server.exe"
    assert found.parent == paths.data / "runtime" / "llama-server"
    assert (found.parent / "llama.dll").is_file()
    assert runtime.find_llama_server(paths) == found


def test_find_llama_server_custom_file(tmp_path: Path) -> None:
    custom = tmp_path / "custom" / "llama-server.exe"
    custom.parent.mkdir()
    custom.write_bytes(b"exe")
    assert runtime.find_llama_server(custom=custom) == custom


def test_ensure_runtime_downloads_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = paths_for(tmp_path)
    payload = runtime_zip()
    entry = {
        "id": runtime.RUNTIME_ID,
        "kind": "runtime",
        "url": "https://example.test/runtime.zip",
        "filename": "runtime.zip",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
    }
    monkeypatch.setattr(runtime, "_runtime_entry", lambda: entry)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("range") is None
        return httpx.Response(
            200,
            content=payload,
            headers={"content-length": str(len(payload))},
        )

    progress: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    found = runtime.ensure_runtime(
        paths, progress=lambda value, _phase: progress.append(value), client=client
    )

    assert found.name == "llama-server.exe"
    assert progress[-1] == 1.0
    assert not (paths.temp / "runtime.zip.partial").exists()


def test_ensure_runtime_rejects_bad_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = paths_for(tmp_path)
    payload = runtime_zip()
    entry = {
        "id": runtime.RUNTIME_ID,
        "kind": "runtime",
        "url": "https://example.test/runtime.zip",
        "filename": "runtime.zip",
        "size_bytes": len(payload),
        "sha256": "00" * 32,
    }
    monkeypatch.setattr(runtime, "_runtime_entry", lambda: entry)

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=payload,
                headers={"content-length": str(len(payload))},
            )
        )
    )
    with pytest.raises(runtime.RuntimeInstallError):
        runtime.ensure_runtime(paths, client=client)
    assert not (paths.temp / "runtime.zip.partial").exists()


def test_ensure_runtime_reuses_complete_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = paths_for(tmp_path)
    payload = runtime_zip()
    entry = {
        "id": runtime.RUNTIME_ID,
        "kind": "runtime",
        "url": "https://example.test/runtime.zip",
        "filename": "runtime.zip",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
    }
    monkeypatch.setattr(runtime, "_runtime_entry", lambda: entry)
    partial = paths.temp / "runtime.zip.partial"
    partial.write_bytes(payload)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected download request: {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    found = runtime.ensure_runtime(paths, client=client)

    assert found.name == "llama-server.exe"
    assert not partial.exists()
