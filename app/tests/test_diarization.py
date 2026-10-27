from __future__ import annotations

from pathlib import Path

import numpy as np

from airec.asr import diarization
from airec.asr.service import _resolve_diarization
from airec.models.manifest import find_model, load_manifest, manifest_dir


class FakeSpeakerEmbedding:
    """Return [1, +1] for positive clips and [1, -1] for negative ones."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def embedding(self, waveform: object) -> np.ndarray:
        self.calls.append(waveform)
        clips = waveform if isinstance(waveform, list) else [waveform]
        vectors = []
        for clip in clips:
            sign = 1.0 if float(np.mean(clip)) >= 0.0 else -1.0
            vectors.append([1.0, sign])
        return np.asarray(vectors, dtype=np.float32)


def test_diarize_segments_groups_by_embedding(monkeypatch) -> None:
    fake = FakeSpeakerEmbedding()
    monkeypatch.setattr(diarization, "load_se", lambda *args, **kwargs: fake)

    sample_rate = 16_000
    audio = np.concatenate(
        [
            np.full(sample_rate, 0.5, dtype=np.float32),
            np.full(sample_rate, -0.5, dtype=np.float32),
            np.full(sample_rate, 0.5, dtype=np.float32),
        ]
    )
    segments = [
        {"start_s": 0.0, "end_s": 1.0, "text": "one"},
        {"start_s": 1.0, "end_s": 2.0, "text": "two"},
        {"start_s": 2.0, "end_s": 3.0, "text": "three"},
    ]

    result = diarization.diarize_segments(segments, audio, sample_rate, threshold=0.7)

    assert [segment["speaker"] for segment in result] == [
        "Спикер 1",
        "Спикер 2",
        "Спикер 1",
    ]
    # The adapter takes a waveform (or a list of waveforms), not (waveform, lengths).
    assert len(fake.calls) == 1
    assert isinstance(fake.calls[0], list)


def test_manifest_has_diarization_models() -> None:
    manifest = load_manifest(manifest_dir())
    entries = [model for model in manifest["models"] if model.get("kind") == "diarization"]

    assert len(entries) >= 3
    for entry in entries:
        assert entry["repo"]
        assert entry["filename"].endswith(".onnx")
        assert entry["purpose"] == "диаризация"
        assert find_model(manifest, entry["id"]) is entry


def test_resolve_diarization_uses_manifest_repo() -> None:
    repo, local = _resolve_diarization("wespeaker-voxceleb-campplus", "", None)

    assert repo == "Wespeaker/wespeaker-voxceleb-campplus"
    assert local is None


def test_resolve_diarization_custom_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    repo, local = _resolve_diarization("custom", str(model_dir), None)

    assert repo is None
    assert local == model_dir


def test_resolve_diarization_prefers_catalog_download(tmp_path: Path) -> None:
    model_dir = tmp_path / "wespeaker-voxceleb-resnet34"
    model_dir.mkdir()
    (model_dir / "voxceleb_resnet34.onnx").write_bytes(b"onnx")

    repo, local = _resolve_diarization("wespeaker-voxceleb-resnet34", "", tmp_path)

    assert repo == "Wespeaker/wespeaker-voxceleb-resnet34"
    assert local == model_dir
