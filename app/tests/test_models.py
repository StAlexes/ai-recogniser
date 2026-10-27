from __future__ import annotations

import time
from pathlib import Path

import httpx

from airec.models.manager import ModelManager


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
