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


def test_pyannote_splits_segment_at_speaker_turn(monkeypatch) -> None:
    """A VAD segment spanning two speakers is cut, with its text split too."""
    monkeypatch.setattr(
        diarization, "load_pyannote", lambda *_args, **_kwargs: FakePyannoteSession()
    )

    sample_rate = 16_000
    audio = np.zeros(6 * sample_rate, dtype=np.float32)
    # Speaker A speaks 0-3 s (region «Спикер 1»), B after 3 s («Спикер 2»);
    # VAD returned one segment over the whole file.  Tokens mimic the real
    # GigaAM vocabulary: spaces are standalone " " or leading-space tokens
    # like " как" — there are no bare word-space tokens.
    tokens = ["привет", " ", "друг", " как", " дела"]
    times = [0.5, 1.0, 1.5, 3.5, 4.2]
    segments = [
        {
            "start_s": 0.5,
            "end_s": 5.0,
            "text": "привет друг как дела",
            "token_times": times,
            "tokens": tokens,
        }
    ]

    result = diarization.diarize_segments(
        segments, audio, sample_rate, repo_id="onnx-community/pyannote-segmentation-3.0"
    )

    assert len(result) == 2
    first, second = result
    assert first["speaker"] == "Спикер 1"
    assert first["start_s"] == 0.5
    assert first["text"] == "привет друг"
    assert second["speaker"] == "Спикер 2"
    assert second["start_s"] == 3.5
    assert second["end_s"] == 5.0
    assert second["text"] == "как дела"
    # Internal token fields must not leak into the result.
    assert "token_times" not in first and "tokens" not in first
    assert "token_times" not in second and "tokens" not in second


def test_pyannote_keeps_segment_without_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        diarization, "load_pyannote", lambda *_args, **_kwargs: FakePyannoteSession()
    )

    sample_rate = 16_000
    audio = np.zeros(6 * sample_rate, dtype=np.float32)
    segments = [{"start_s": 0.5, "end_s": 5.0, "text": "целиком"}]

    result = diarization.diarize_segments(
        segments, audio, sample_rate, repo_id="onnx-community/pyannote-segmentation-3.0"
    )

    assert len(result) == 1
    assert result[0]["speaker"] == "Спикер 1"
    assert result[0]["text"] == "целиком"


class SlotPyannoteSession:
    """Fake pyannote session with per-slot frame ranges (0.5 s per frame).

    The ONNX graph outputs log-softmax values, so an active class uses 0
    (probability 1) and inactive classes a large negative log-probability.
    """

    def __init__(self, slots: dict[int, tuple[int, int]]) -> None:
        self._slots = slots

    def run(self, _outputs: object, _feeds: object) -> list[np.ndarray]:
        frames = 20
        logits = np.full((1, frames, 7), -10.0, dtype=np.float32)
        for slot, (first, last) in self._slots.items():
            logits[:, first:last, slot] = 0.0
        return [logits]


def test_pyannote_drops_noise_slot(monkeypatch) -> None:
    """A hallucinated short blip must not steal a speaker number."""
    monkeypatch.setattr(
        diarization,
        "load_pyannote",
        lambda *_args, **_kwargs: SlotPyannoteSession({1: (0, 2), 2: (0, 10), 3: (12, 20)}),
    )

    sample_rate = 16_000
    audio = np.zeros(10 * sample_rate, dtype=np.float32)
    segments = [
        {"start_s": 0.5, "end_s": 2.0, "text": "one"},
        {"start_s": 6.5, "end_s": 8.0, "text": "two"},
    ]

    result = diarization.diarize_segments(
        segments, audio, sample_rate, repo_id="onnx-community/pyannote-segmentation-3.0"
    )

    # Slot 1 holds only 1 s of speech (< MIN_SLOT_SPEECH_S) and is dropped;
    # slot 2 is the first real voice and slot 3 the second one.
    assert [segment["speaker"] for segment in result] == ["Спикер 1", "Спикер 2"]
    assert "Спикер 3" not in [segment["speaker"] for segment in result]


def test_pyannote_numbers_by_first_appearance(monkeypatch) -> None:
    """The first voice to speak becomes «Спикер 1», whichever slot it uses."""
    monkeypatch.setattr(
        diarization,
        "load_pyannote",
        lambda *_args, **_kwargs: SlotPyannoteSession({3: (0, 10), 2: (12, 20)}),
    )

    sample_rate = 16_000
    audio = np.zeros(10 * sample_rate, dtype=np.float32)
    segments = [
        {"start_s": 0.5, "end_s": 2.0, "text": "one"},
        {"start_s": 6.5, "end_s": 8.0, "text": "two"},
    ]

    result = diarization.diarize_segments(
        segments, audio, sample_rate, repo_id="onnx-community/pyannote-segmentation-3.0"
    )

    # Slot 3 speaks first, so it must be «Спикер 1» despite its slot number.
    assert [segment["speaker"] for segment in result] == ["Спикер 1", "Спикер 2"]


def test_pyannote_fallback_continues_previous_speaker(monkeypatch) -> None:
    """A segment outside every region keeps the previous segment's speaker."""
    monkeypatch.setattr(
        diarization,
        "load_pyannote",
        lambda *_args, **_kwargs: SlotPyannoteSession({2: (0, 6), 3: (12, 16)}),
    )

    sample_rate = 16_000
    audio = np.zeros(10 * sample_rate, dtype=np.float32)
    segments = [
        {"start_s": 0.5, "end_s": 2.0, "text": "one"},
        {"start_s": 3.5, "end_s": 4.5, "text": "gap before"},
        {"start_s": 6.5, "end_s": 7.5, "text": "two"},
        {"start_s": 8.5, "end_s": 9.5, "text": "gap after"},
    ]

    result = diarization.diarize_segments(
        segments, audio, sample_rate, repo_id="onnx-community/pyannote-segmentation-3.0"
    )

    assert [segment["speaker"] for segment in result] == [
        "Спикер 1",
        "Спикер 1",
        "Спикер 2",
        "Спикер 2",
    ]


def test_merge_same_speaker_segments() -> None:
    """Fragments of one speaker separated by a short pause are joined."""
    merge = diarization._merge_same_speaker_segments
    segments = [
        {"start_s": 0.0, "end_s": 10.0, "speaker": "Спикер 1", "text": "раз"},
        {"start_s": 10.3, "end_s": 12.0, "speaker": "Спикер 1", "text": "два"},
        {"start_s": 12.4, "end_s": 14.0, "speaker": "Спикер 1", "text": "три"},
        {"start_s": 14.9, "end_s": 16.0, "speaker": "Спикер 2", "text": "четыре"},
        {"start_s": 16.1, "end_s": 18.0, "speaker": "Спикер 2", "text": "пять"},
        {"start_s": 18.7, "end_s": 20.0, "speaker": "Спикер 2", "text": "шесть"},
    ]

    result = merge(segments)

    # 10.3 s and 12.4 s gaps are < MERGE_GAP_S: the three «Спикер 1» pieces
    # join, as do the two «Спикер 2» pieces; the 18.7 s gap (0.7 s) stays.
    assert len(result) == 3
    assert (result[0]["start_s"], result[0]["end_s"]) == (0.0, 14.0)
    assert result[0]["text"] == "раз два три"
    assert (result[1]["start_s"], result[1]["end_s"]) == (14.9, 18.0)
    assert result[1]["text"] == "четыре пять"
    assert (result[2]["start_s"], result[2]["end_s"]) == (18.7, 20.0)
    assert result[2]["text"] == "шесть"
    # The input list must not be modified in place.
    assert len(segments) == 6
    assert segments[1]["text"] == "два"


def test_merge_same_speaker_segments_respects_cap() -> None:
    """Merging stops once the joined segment reaches MAX_MERGED_SEGMENT_S."""
    merge = diarization._merge_same_speaker_segments
    segments = [
        {"start_s": 0.0, "end_s": 30.0, "speaker": "Спикер 1", "text": "a"},
        {"start_s": 30.1, "end_s": 40.0, "speaker": "Спикер 1", "text": "b"},
    ]

    result = merge(segments)

    assert len(result) == 2
    assert [s["text"] for s in result] == ["a", "b"]


def test_merge_same_speaker_segments_requires_label() -> None:
    """Segments without a speaker label are never merged."""
    merge = diarization._merge_same_speaker_segments
    segments = [
        {"start_s": 0.0, "end_s": 1.0, "speaker": None, "text": "a"},
        {"start_s": 1.1, "end_s": 2.0, "speaker": None, "text": "b"},
    ]

    result = merge(segments)

    assert len(result) == 2


def test_manifest_includes_pyannote() -> None:
    manifest = load_manifest(manifest_dir())
    entry = find_model(manifest, "pyannote-segmentation-3.0")

    assert entry is not None
    assert entry["repo"] == diarization.PYANNOTE_REPO
    assert entry["filename"] == "model.onnx"
    assert entry["size_bytes"] > 0
