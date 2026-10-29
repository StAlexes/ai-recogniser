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


class FakePyannoteSession:
    """Return spk1 activity in the first half, spk2 in the second.

    The ONNX graph outputs log-softmax values, so an active class uses 0
    (probability 1) and inactive classes a large negative log-probability.
    """

    def run(self, _outputs: object, feeds: object) -> list[np.ndarray]:
        # 20 frames per 10 s window → 0.5 s of audio per frame.  The raw
        # classes deliberately start at slot 2 (like the real model, which
        # often ignores slot 1) — labels must still begin at «Спикер 1».
        frames = 20
        logits = np.full((1, frames, 7), -10.0, dtype=np.float32)
        logits[:, :6, 2] = 0.0  # speaker A: 0-3 s
        logits[:, 6:, 3] = 0.0  # speaker B: 3-10 s
        return [logits]


def test_diarize_with_pyannote(monkeypatch) -> None:
    monkeypatch.setattr(
        diarization, "load_pyannote", lambda *_args, **_kwargs: FakePyannoteSession()
    )

    sample_rate = 16_000
    audio = np.zeros(6 * sample_rate, dtype=np.float32)
    segments = [
        {"start_s": 0.5, "end_s": 2.0, "text": "one"},
        {"start_s": 3.0, "end_s": 5.0, "text": "two"},
    ]

    result = diarization.diarize_segments(
        segments, audio, sample_rate, repo_id="onnx-community/pyannote-segmentation-3.0"
    )

    assert [segment["speaker"] for segment in result] == ["Спикер 1", "Спикер 2"]


def test_manifest_includes_pyannote() -> None:
    manifest = load_manifest(manifest_dir())
    entry = find_model(manifest, "pyannote-segmentation-3.0")

    assert entry is not None
    assert entry["repo"] == diarization.PYANNOTE_REPO
    assert entry["filename"] == "model.onnx"
    assert entry["size_bytes"] > 0
