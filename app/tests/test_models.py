from __future__ import annotations

import time
from pathlib import Path

import httpx

from airec.models.manager import ModelManager


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
