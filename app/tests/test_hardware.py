from __future__ import annotations

from airec.bench.hardware import _is_virtual_gpu, _normalize_gpus


def test_detects_virtual_adapters() -> None:
    assert _is_virtual_gpu("Parsec Virtual Display Adapter")
    assert _is_virtual_gpu("VMware SVGA 3D")
    assert _is_virtual_gpu("Microsoft Basic Display Adapter")
    assert not _is_virtual_gpu("AMD Radeon 780M")
    assert not _is_virtual_gpu("NVIDIA GeForce RTX 4060 Laptop GPU")


def test_physical_gpu_goes_first() -> None:
    gpus = _normalize_gpus(
        [
            {"Name": "Parsec Virtual Display Adapter", "AdapterRAM": 0, "DriverVersion": "1"},
            {"Name": "AMD Radeon 780M", "AdapterRAM": 2 * 1024**3, "DriverVersion": "2"},
        ]
    )

    assert [gpu["name"] for gpu in gpus] == ["AMD Radeon 780M", "Parsec Virtual Display Adapter"]
    assert gpus[0]["vram_gb"] == 2.0
    assert gpus[1]["virtual"] is True
