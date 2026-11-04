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


def _disable_voice_clustering(monkeypatch) -> None:
    """Откат гибридной кластеризации на цепочки слотов (тесты off-line).

    Движок PyAnnote по умолчанию кластеризует регионы по эмбеддингам
    WeSpeaker; в юнит-тестах модель недоступна — load_se должен падать, чтобы
    сработал штатный откат без сети.
    """

    def _no_se(*_args, **_kwargs):
        raise diarization.DiarizationError("speaker embeddings disabled in test")

    monkeypatch.setattr(diarization, "load_se", _no_se)


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
    _disable_voice_clustering(monkeypatch)

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
    _disable_voice_clustering(monkeypatch)

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
    _disable_voice_clustering(monkeypatch)

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
    _disable_voice_clustering(monkeypatch)

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
    _disable_voice_clustering(monkeypatch)

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
    _disable_voice_clustering(monkeypatch)

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


# -------------------------------------------------------------------------- #
# PIT-выравнивание каналов pyannote между окнами
# -------------------------------------------------------------------------- #
class PermutedPyannoteSession:
    """Fake pyannote: real voice A stays in «its» class across windows.

    Сеть с PIT-обучением возвращает анонимные каналы: в чётных окнах голос A
    попадает в класс 1, в нечётных — в класс 2 (классы локальны для окна).
    Наивная аккумуляция суммировала бы голос A то в слот 1, то в слот 2 и
    после усреднения давала бы «винегрет»; выравнивание обязано собрать
    голос A целиком в один слот.
    """

    def __init__(self, window_samples: int, advance_samples: int) -> None:
        self._window_samples = window_samples
        self._advance_samples = advance_samples
        self._calls = 0

    def run(self, _outputs: object, feeds: object) -> list[np.ndarray]:
        window = np.asarray(feeds["input_values"], dtype=np.float32)
        assert window.ndim == 3
        # 10 frames per window; frame length covers the whole window.
        frames = 10
        logits = np.full((1, frames, 7), -10.0, dtype=np.float32)
        # Голос A звучит во всём окне; его канал чередуется 1 -> 2 -> 1 ...
        channel = 1 if self._calls % 2 == 0 else 2
        logits[:, :, channel] = 0.0
        self._calls += 1
        return [logits]


def test_pyannote_permutation_keeps_single_speaker_stable(monkeypatch) -> None:
    """Каналы, меняющиеся между окнами, собираются в один слот (PIT-фикс)."""
    sample_rate = 16_000
    session = PermutedPyannoteSession(
        round(diarization.PYANNOTE_WINDOW_DURATION_S * sample_rate),
        round(diarization.PYANNOTE_WINDOW_STEP_S * sample_rate),
    )
    monkeypatch.setattr(
        diarization, "load_pyannote", lambda *_args, **_kwargs: session
    )

    # 30 s: пять окон с шагом 5 s — четыре сопоставления каналов.
    audio = np.zeros(int(30.0 * sample_rate), dtype=np.float32)

    regions = diarization._pyannote_speaker_regions(session, audio, sample_rate)

    # Один реальный голос — одна метка во всех регионах.  До фикса голос
    # чередовал слоты 1/2/1/2..., каждый слот набирал > MIN_SLOT_SPEECH_S
    # речи и появлялись две метки.
    labels = {label for label, _start, _end in regions}
    assert labels == {"Спикер 1"}


def test_find_best_speaker_permutation_identity_for_equal_channels() -> None:
    """Идентичные каналы не переставляются (нулевой случай)."""
    prev = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    curr = prev.copy()
    assert diarization._find_best_speaker_permutation(prev, curr) == (0, 1, 2)


def test_find_best_speaker_permutation_swaps_channels() -> None:
    """Каналы, поменяные местами, восстанавливаются перестановкой."""
    prev = np.array(
        [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    curr = prev[[1, 0, 2], :]  # голоса поменяны местами
    perm = diarization._find_best_speaker_permutation(prev, curr)
    # perm[i] — канал curr, соответствующий каналу i prev: голос 0 -> канал 1.
    assert perm == (1, 0, 2)


# -------------------------------------------------------------------------- #
# Задержка CTC: знак компенсации
# -------------------------------------------------------------------------- #
def test_split_at_speaker_turns_subtracts_ctc_delay() -> None:
    """Точка голосования сдвигается НАЗАД (t_audio = t_token − Δt).

    Слово «да» стоит у границы спикеров (2.0 с): середина токена 2.025 с,
    после вычитания задержки 0.1 — 1.925 с, регион Спикера 1.  Старый
    (ошибочный) знак давал 2.125 с — слово перепрыгивало Спикеру 2.
    """
    regions = [
        ("Спикер 1", 0.0, 2.0),
        ("Спикер 2", 2.0, 4.0),
    ]
    segment = {
        "start_s": 0.0,
        "end_s": 3.0,
        "text": "да нет",
        "tokens": ["да", " нет"],
        "token_times": [1.9, 2.15],
    }

    result = diarization._split_at_speaker_turns(
        [dict(segment)], regions, vote_delay_s=0.1
    )

    assert len(result) == 2
    assert result[0]["speaker"] == "Спикер 1"
    assert result[0]["text"] == "да"
    assert result[1]["speaker"] == "Спикер 2"


# -------------------------------------------------------------------------- #
# Перекрытие спикеров: перебивка не должна поглощаться монологом
# -------------------------------------------------------------------------- #
def test_split_at_speaker_turns_overlap_keeps_interruption() -> None:
    """Слова в зоне оверлапа двух регионов принадлежат вступившему спикеру.

    Старый label_at отдавал момент первому региону в отсортированном списке,
    поэтому короткое перебивание внутри длинного монолога целиком
    приписывалось говорившему; к тому же однословный всплеск сглаживался
    обратно, а run короче 0.6 с складывался в предыдущий.
    """
    regions = [
        ("Спикер 1", 0.0, 10.75),
        ("Спикер 2", 10.45, 10.8),
        ("Спикер 1", 10.9, 20.0),
    ]
    segment = {
        "start_s": 0.0,
        "end_s": 20.0,
        "text": "раз да два",
        "tokens": ["раз", " да", " два"],
        "token_times": [1.0, 10.5, 10.9],
    }

    result = diarization._split_at_speaker_turns([dict(segment)], regions)

    assert [piece["speaker"] for piece in result] == [
        "Спикер 1",
        "Спикер 2",
        "Спикер 1",
    ]
    assert [piece["text"] for piece in result] == ["раз", "да", "два"]


# -------------------------------------------------------------------------- #
# Голосовая кластеризация регионов (долгосрочная память тембра)
# -------------------------------------------------------------------------- #
def test_agglomerative_speaker_clusters_merge_by_threshold() -> None:
    """Средне-связная кластеризация объединяет близкие тембры по всему файлу."""
    embeddings = np.asarray(
        [
            [1.0, 0.0],  # голос A
            [0.9, 0.1],  # голос A
            [0.0, 1.0],  # голос B
            [0.1, 0.9],  # голос B
        ],
        dtype=np.float32,
    )

    clusters = diarization._agglomerative_speaker_clusters(embeddings, 0.7)

    assert clusters[0] == clusters[1]
    assert clusters[2] == clusters[3]
    assert clusters[0] != clusters[2]


class ChainBreakPyannoteSession:
    """Один и тот же голос до и после паузы попадает в разные слоты сети.

    20 фреймов на 10-с окно (0.5 с на фрейм): 0–3 с — слот 2, 6–9 с —
    слот 3.  Без голосовой кластеризации это две цепочки и два спикера.
    """

    def run(self, _outputs: object, _feeds: object) -> list[np.ndarray]:
        frames = 20
        logits = np.full((1, frames, 7), -10.0, dtype=np.float32)
        logits[:, 0:6, 2] = 0.0
        logits[:, 12:18, 3] = 0.0
        return [logits]


def _chain_break_setup(monkeypatch) -> tuple[int, np.ndarray, list[dict[str, object]]]:
    monkeypatch.setattr(
        diarization,
        "load_pyannote",
        lambda *_args, **_kwargs: ChainBreakPyannoteSession(),
    )
    sample_rate = 16_000
    audio = np.zeros(10 * sample_rate, dtype=np.float32)
    audio[0 : 3 * sample_rate] = 0.5
    audio[6 * sample_rate : 9 * sample_rate] = 0.5
    segments = [
        {"start_s": 0.5, "end_s": 2.5, "text": "one"},
        {"start_s": 6.5, "end_s": 8.5, "text": "two"},
    ]
    return sample_rate, audio, segments


def test_pyannote_voice_clustering_rejoins_speaker_after_pause(monkeypatch) -> None:
    """Пауза длиннее окна сегментатора не плодит нового спикера.

    Две цепочки слотов с одним и тем же голосом объединяются по тембру
    WeSpeaker в один кластер («Спикер 1» и для первой, и для второй реплики).
    """
    sample_rate, audio, segments = _chain_break_setup(monkeypatch)
    fake = FakeSpeakerEmbedding()
    monkeypatch.setattr(diarization, "load_se", lambda *args, **kwargs: fake)

    result = diarization.diarize_segments(
        segments, audio, sample_rate, repo_id="onnx-community/pyannote-segmentation-3.0"
    )

    assert [segment["speaker"] for segment in result] == ["Спикер 1", "Спикер 1"]


def test_pyannote_voice_clustering_falls_back_without_embeddings(monkeypatch) -> None:
    """Недоступность модели эмбеддингов — откат на цепочки слотов."""
    sample_rate, audio, segments = _chain_break_setup(monkeypatch)
    _disable_voice_clustering(monkeypatch)

    result = diarization.diarize_segments(
        segments, audio, sample_rate, repo_id="onnx-community/pyannote-segmentation-3.0"
    )

    assert [segment["speaker"] for segment in result] == ["Спикер 1", "Спикер 2"]


def test_pyannote_voice_clustering_keeps_different_voices_apart(monkeypatch) -> None:
    """Разные голоса в разных цепочках не должны сливаться по тембру."""
    monkeypatch.setattr(
        diarization,
        "load_pyannote",
        lambda *_args, **_kwargs: ChainBreakPyannoteSession(),
    )
    monkeypatch.setattr(
        diarization, "load_se", lambda *args, **kwargs: FakeSpeakerEmbedding()
    )
    sample_rate = 16_000
    audio = np.zeros(10 * sample_rate, dtype=np.float32)
    audio[0 : 3 * sample_rate] = 0.5  # цепочка 1: «голос А»
    audio[6 * sample_rate : 9 * sample_rate] = -0.5  # цепочка 2: «голос Б»
    segments = [
        {"start_s": 0.5, "end_s": 2.5, "text": "one"},
        {"start_s": 6.5, "end_s": 8.5, "text": "two"},
    ]

    result = diarization.diarize_segments(
        segments, audio, sample_rate, repo_id="onnx-community/pyannote-segmentation-3.0"
    )

    assert [segment["speaker"] for segment in result] == ["Спикер 1", "Спикер 2"]


def test_pyannote_voice_clustering_skipped_for_single_chain(monkeypatch) -> None:
    """Одна цепочка слотов — один спикер, модель эмбеддингов не нужна."""
    monkeypatch.setattr(
        diarization,
        "load_pyannote",
        lambda *_args, **_kwargs: SlotPyannoteSession({2: (0, 10)}),
    )
    calls = []

    def _boom(*args, **kwargs):
        calls.append(1)
        raise AssertionError("load_se must not be called for a single chain")

    monkeypatch.setattr(diarization, "load_se", _boom)
    sample_rate = 16_000
    audio = np.zeros(10 * sample_rate, dtype=np.float32)
    segments = [{"start_s": 0.5, "end_s": 8.0, "text": "solo"}]

    result = diarization.diarize_segments(
        segments, audio, sample_rate, repo_id="onnx-community/pyannote-segmentation-3.0"
    )

    assert calls == []
    assert [segment["speaker"] for segment in result] == ["Спикер 1"]
