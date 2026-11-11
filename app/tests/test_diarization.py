from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
def test_cluster_region_voices_reattaches_small_clusters() -> None:
    """Кластер меньше минимума приклеивается к ближайшему крупному."""
    embeddings = np.asarray(
        [
            [1.0, 0.0],  # голос A
            [0.98, 0.2],  # голос A
            [0.7, 0.71],  # выброс: не дотянул до порога A, одиночный
            [0.0, 1.0],  # голос B
            [0.2, 0.98],  # голос B
            [0.1, 1.0],  # голос B
        ],
        dtype=np.float32,
    )

    clusters = diarization._cluster_region_voices(embeddings, 0.9, min_cluster_size=2)

    assert len(set(clusters)) == 2
    assert clusters[0] == clusters[1]
    assert clusters[3] == clusters[4] == clusters[5]


def test_cluster_region_voices_numbers_by_first_appearance() -> None:
    """Номера кластеров отсчитываются от первого появления в записи."""
    embeddings = np.asarray(
        [[0.0, 1.0], [0.1, 0.9], [1.0, 0.0], [0.9, 0.1]], dtype=np.float32
    )

    clusters = diarization._cluster_region_voices(embeddings, 0.8, min_cluster_size=1)

    assert clusters == [0, 0, 1, 1]


def test_cluster_region_voices_respects_cap() -> None:
    """Взаимно далёкие векторы не плодят кластеры сверх потолка."""
    angles = np.linspace(0.0, np.pi, 16)
    embeddings = np.vstack([np.cos(angles), np.sin(angles)]).T.astype(np.float32)

    clusters = diarization._cluster_region_voices(
        embeddings, 0.99, min_cluster_size=1, cap=4
    )

    assert len(set(clusters)) <= 4


def test_adaptive_voice_threshold_lowers_on_compressed_space() -> None:
    """Сжатое пространство (узкая полоса): порог понижается в долину."""
    # 24 неструктурированных вектора: попарные сходства около нуля,
    # p95 ниже профильного порога — как у узкополосной записи.
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(24, 256)).astype(np.float32)

    threshold, adapted = diarization._adaptive_voice_threshold(embeddings, 0.74)

    assert adapted
    assert 0.30 <= threshold < 0.74


def test_adaptive_voice_threshold_unchanged_on_clean_space() -> None:
    """Чистое пространство: p95 выше порога, адаптация не нужна."""
    rows = []
    for base in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        for i in range(8):
            jitter = [0.01 * ((i + j) % 3 - 1) for j in range(3)]
            rows.append([b + dj for b, dj in zip(base, jitter, strict=True)])
    embeddings = np.asarray(rows * 40, dtype=np.float32)

    threshold, adapted = diarization._adaptive_voice_threshold(embeddings, 0.74)

    assert not adapted
    assert threshold == 0.74


def test_adaptive_voice_threshold_skips_small_inputs() -> None:
    """Мало векторов — перцентили непоказательны, порог не трогаем."""
    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)

    threshold, adapted = diarization._adaptive_voice_threshold(embeddings, 0.74)

    assert not adapted
    assert threshold == 0.74


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


# -------------------------------------------------------------------------- #
# Адаптивный порог: порог остаётся в долине между «свой/чужой»
# -------------------------------------------------------------------------- #
_DIM = 256
_PROFILE_THRESHOLD = 0.74  # порог профиля, от которого понижает адаптация
_MIN_CLUSTER_SIZE = 8  # продовый voice_min_regions по умолчанию


def _multi_voice_embeddings(
    seed: int = 1,
    voices: int = 8,
    per_voice: int = 5,
    region_noise: float = 0.04,
) -> np.ndarray:
    """*voices* «голосов» по *per_voice* регионов в R^256.

    direction_i = normalize(e_i + 1.2·common) — общая для всех компонента
    сближает направления, как реальные тембры в узкополосной записи; регион
    = normalize(direction + region_noise·N(0, I)).  Сходства: свои ~0.71
    (мин 0.66), чужие ~0.39 (макс 0.50) — голоса разделимы с запасом, между
    массами есть пустая долина [0.55, 0.66].
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(size=_DIM)
    bases = rng.normal(size=(voices, _DIM))
    rows = []
    for base in bases:
        direction = base + 1.2 * common
        direction = direction / np.linalg.norm(direction)
        for _ in range(per_voice):
            vector = direction + region_noise * rng.normal(size=_DIM)
            rows.append(vector / np.linalg.norm(vector))
    return np.asarray(rows, dtype=np.float32)


def _two_voice_embeddings(
    seed: int = 0,
    per_voice: int = 30,
    within: float = 0.68,
    cross: float = 0.58,
) -> np.ndarray:
    """Два «голоса» по *per_voice* регионов: normalize(base + noise).

    Полная |noise|² = 1/within − 1 в R^256 даёт ожидаемое попарное сходство
    своих сэмплов ~within; базы разведены на косинус cross/within, поэтому
    чужие сэмплы сходятся в среднем на ~cross.  Распределения «свой/чужой»
    перекрываются — попарной долины нет, голоса разделимы только порогом 0.74.
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt((1.0 / within - 1.0) / _DIM)
    base_a = rng.normal(size=_DIM)
    base_a /= np.linalg.norm(base_a)
    perpendicular = rng.normal(size=_DIM)
    perpendicular -= (perpendicular @ base_a) * base_a
    perpendicular /= np.linalg.norm(perpendicular)
    base_cos = cross / within
    base_b = base_cos * base_a + np.sqrt(1.0 - base_cos * base_cos) * perpendicular
    rows = []
    for base in (base_a, base_b):
        for _ in range(per_voice):
            vector = base + sigma * rng.normal(size=_DIM)
            rows.append(vector / np.linalg.norm(vector))
    return np.asarray(rows, dtype=np.float32)


def test_adaptive_threshold_merges_separable_multivoice_record() -> None:
    """Адаптивный порог остаётся в долине на многоголосой записи.

    В записи 8 разделимых голосов, но «своих» пар лишь ~10% массы: p95 = 0.71
    ниже профильного порога — адаптация срабатывает.  Порог обязан не
    опускаться ниже долины между «своими» (0.66–0.78) и «чужими» (0.26–0.50)
    сходствами: прежняя формула «p85 − 0.05» давала ≈ 0.41 (ниже верха чужой
    массы 0.50) и сливала все голоса в один кластер.  Контроль: порог внутри
    долины (0.60) даёт ровно 8 кластеров — синтетика разделима попарно.
    """
    voices, per_voice = 8, 5
    embeddings = _multi_voice_embeddings(voices=voices, per_voice=per_voice)
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = unit @ unit.T
    own = [
        float(sims[i, j])
        for i in range(voices * per_voice)
        for j in range(i + 1, voices * per_voice)
        if i // per_voice == j // per_voice
    ]
    alien = [
        float(sims[i, j])
        for i in range(voices * per_voice)
        for j in range(i + 1, voices * per_voice)
        if i // per_voice != j // per_voice
    ]
    assert min(own) > max(alien)

    valley_threshold = 0.60
    clusters = diarization._cluster_region_voices(
        embeddings, valley_threshold, _MIN_CLUSTER_SIZE
    )
    assert len(set(clusters)) == voices

    threshold, adapted = diarization._adaptive_voice_threshold(
        embeddings, _PROFILE_THRESHOLD
    )
    assert adapted
    clusters = diarization._cluster_region_voices(
        embeddings, threshold, _MIN_CLUSTER_SIZE
    )
    assert len(set(clusters)) == voices


def test_adaptive_threshold_swallows_second_voice_via_grown_centroid() -> None:
    """Без долины порог не понижается — второй голос не проглатывается.

    Два голоса по 30 регионов: свои сэмплы сходятся на ~0.68, чужие — на ~0.57
    (распределения перекрываются, попарной долины нет).  Центроид тридцати
    сэмплов первого голоса притягивает второй голос (среднее сходство с ним
    ~0.69), тогда как среднее попарное «чужое» сходство (~0.57) заметно ниже
    порога: прежняя адаптация опускала порог до ~0.64 и сливала голоса через
    центроидную геометрию, не нарушив порог ни в одном отдельном решении.
    Теперь запись без долины не понижает порог — на профильном 0.74 голоса
    остаются порознь (2 кластера).
    """
    per_voice = 30
    embeddings = _two_voice_embeddings(per_voice=per_voice)
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cross = [
        float(unit[i] @ unit[j])
        for i in range(per_voice)
        for j in range(per_voice, 2 * per_voice)
    ]

    clusters = diarization._cluster_region_voices(
        embeddings, _PROFILE_THRESHOLD, _MIN_CLUSTER_SIZE
    )
    assert len(set(clusters)) == 2

    threshold, _adapted = diarization._adaptive_voice_threshold(
        embeddings, _PROFILE_THRESHOLD
    )
    # Попарные «чужие» сходства в среднем ниже возвращённого порога: слияние
    # (если бы оно случилось) могло обеспечить только центроидная геометрия.
    assert float(np.mean(cross)) < threshold
    clusters = diarization._cluster_region_voices(
        embeddings, threshold, _MIN_CLUSTER_SIZE
    )
    assert len(set(clusters)) == 2


# -------------------------------------------------------------------------- #
# PIT-выравнивание: неинформативный overlap не раздаёт слоты вслепую
# -------------------------------------------------------------------------- #
class ScenarioPyannoteSession:
    """Фейковая pyannote-сессия с планом активности в абсолютном времени.

    Каждый вызов ``run`` — очередное скользящее окно: старт окна равен
    index·step (файлы в тестах подобраны без хвостового окна, поэтому окна
    идут ровно с шагом от нуля — см. цикл стартов в
    ``_pyannote_speaker_regions``).  Активность голосов задана списком
    ``(канал, t_start, t_end)`` в секундах от начала записи: 20 фреймов на
    10-с окно (0.5 с на фрейм), активный класс = 0 (лог-вероятность 1),
    неактивные = −10 — как у настоящей модели с log-softmax.  Каналы локальны
    для окна: сеть вольна отдать один голос разным классам в разных окнах,
    план это и кодирует.  Одновременные голоса дают активность обоим каналам
    (эквивалент power-set классов {spk_i+spk_j}).
    """

    def __init__(
        self,
        voices: list[tuple[int, float, float]],
        sample_rate: int,
        *,
        window_s: float,
        step_s: float,
    ) -> None:
        self._voices = voices
        self._sample_rate = sample_rate
        self._window_s = window_s
        self._step_s = step_s
        self._calls = 0

    def run(self, _outputs: object, _feeds: object) -> list[np.ndarray]:
        window_start = self._calls * self._step_s
        self._calls += 1
        frames = 20
        frame_s = self._window_s / frames
        logits = np.full((1, frames, 7), -10.0, dtype=np.float32)
        for channel, voice_start, voice_end in self._voices:
            for frame in range(frames):
                frame_start = window_start + frame * frame_s
                frame_end = frame_start + frame_s
                if frame_start < voice_end and frame_end > voice_start:
                    logits[:, frame, channel + 1] = 0.0
        return [logits]


# Параметры профиля meeting (app/airec/asr/diarization_profiles.py):
# шаг 3.5 с — overlap-участок между окнами длиннее (6.5 с), гистерезис
# onset/offset 0.54/0.42.  Пустой словарь — встроенные дефолты функции
# (окно 10 с, шаг 5 с, onset = offset = 0.5).
MEETING_PARAMS: dict[str, float] = {
    "onset": 0.54,
    "offset": 0.42,
    "min_duration_on_s": 0.12,
    "min_duration_off_s": 0.22,
    "pad_onset_s": 0.05,
    "pad_offset_s": 0.05,
    "window_duration_s": 10.0,
    "window_step_s": 3.5,
}


@pytest.mark.parametrize(
    ("params", "audio_s"),
    [
        pytest.param({}, 30.0, id="defaults-step-5s"),
        pytest.param(MEETING_PARAMS, 27.5, id="meeting-profile-step-3.5s"),
    ],
)
def test_silent_overlap_does_not_split_returning_voice(
    params: dict[str, float], audio_s: float
) -> None:
    """Голос, молчавший весь overlap соседних окон, не раздваивается.

    A говорит всю запись каналом 0; B — [4.0, 6.5) каналом 1 и [16.0, 19.0)
    каналом 2: на всём overlap-участке соседних окон (window − step) B молчит,
    и слепая PIT-перестановка при равных счётах отдавала его реплики разным
    слотам («Спикер 2» и «Спикер 3»).  Выравнивание обязано разрешать
    совпадающие счёты по активности вне overlap: активная в новой части окна
    реплика наследует слот, который нёс голос в прошлом окне.  Проверяется на
    уровне ``_pyannote_speaker_regions``: с рабочей моделью эмбеддингов
    тембровая кластеризация склеила бы цепочки сама и замаскировала баг.
    """
    sample_rate = 16_000
    voices = [
        (0, 0.0, audio_s),  # A говорит всю запись
        (1, 4.0, 6.5),  # реплика B до паузы — канал 1
        (2, 16.0, 19.0),  # реплика B после паузы — сеть даёт канал 2
    ]
    session = ScenarioPyannoteSession(
        voices,
        sample_rate,
        window_s=params.get("window_duration_s", diarization.PYANNOTE_WINDOW_DURATION_S),
        step_s=params.get("window_step_s", diarization.PYANNOTE_WINDOW_STEP_S),
    )
    # Длина файла подобрана так, что последнее скользящее окно заканчивается
    # ровно в конце записи: хвостового окна нет, старты окон = index·step.
    audio = np.zeros(round(audio_s * sample_rate), dtype=np.float32)

    regions = diarization._pyannote_speaker_regions(session, audio, sample_rate, **params)

    # Синтетика воспроизвела геометрию сценария: монолог A от нуля и две
    # реплики B по краям паузы (допуск 0.15 с — паддинг профиля meeting).
    timed = sorted(regions, key=lambda item: item[1])
    assert len(timed) == 3, f"ожидались три региона (A + две реплики B): {timed}"
    for position, expected_start in enumerate((0.0, 4.0, 16.0)):
        assert abs(timed[position][1] - expected_start) <= 0.15, timed

    labels = [label for label, _start, _end in timed]
    # Обе реплики B — один голос: одна метка, а не «Спикер 2» и «Спикер 3».
    assert labels[1] == labels[2], (
        f"голос B раздвоился из-за слепой PIT-перестановки на тихом overlap: {timed}"
    )
    assert labels[0] != labels[1], f"реплики B склеились с голосом A: {timed}"
    assert len(set(labels)) == 2, f"ожидались ровно два голоса: {timed}"


def test_silent_overlap_does_not_glue_new_voice_to_previous_speaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Новый голос в полностью тихий overlap не прилипает к цепочке A.

    A говорит [0, 3) каналом 0, затем длинная тишина, B [14, 18) — сеть снова
    отдаёт ему канал 0.  Overlap первых двух окон [5, 10) полностью тихий:
    матрица скалярных произведений нулевая, все перестановки равны, и B
    наследует слот A — оба голоса попадают в одну цепочку слотов.  Тембровая
    кластеризация обязана для такой записи запуститься (load_se вызывается)
    и разделить голоса по тембру — две метки.
    """
    sample_rate = 16_000
    voices = [
        (0, 0.0, 3.0),  # A — канал 0
        (0, 14.0, 18.0),  # B — тот же канал 0 (слоты сети анонимны)
    ]
    session = ScenarioPyannoteSession(voices, sample_rate, window_s=10.0, step_s=5.0)
    # 25 с: окна с шагом 5 с (0, 5, 10, 15), хвостового окна нет; overlap
    # окон 0-1 — [5, 10) — полностью тихий.  Тембры A и B противопоставлены
    # знаком среднего, чтобы FakeSpeakerEmbedding давал разные векторы.
    audio = np.zeros(round(25.0 * sample_rate), dtype=np.float32)
    audio[0 : 3 * sample_rate] = 0.5  # тембр A -> эмбеддинг [1, +1]
    audio[14 * sample_rate : 18 * sample_rate] = -0.5  # тембр B -> [1, -1]
    segments = [
        {"start_s": 0.5, "end_s": 2.5, "text": "one"},
        {"start_s": 14.5, "end_s": 16.5, "text": "two"},
    ]

    se_calls: list[int] = []

    def _recording_load_se(*_args: object, **_kwargs: object) -> FakeSpeakerEmbedding:
        se_calls.append(1)
        return FakeSpeakerEmbedding()

    monkeypatch.setattr(
        diarization, "load_pyannote", lambda *_args, **_kwargs: session
    )
    monkeypatch.setattr(diarization, "load_se", _recording_load_se)

    result = diarization.diarize_segments(
        segments, audio, sample_rate, repo_id="onnx-community/pyannote-segmentation-3.0"
    )

    # Одна цепочка слотов из нескольких регионов — не повод обходить
    # кластеризацию: в одной цепочке могут быть два разных человека.
    assert se_calls, (
        "тембровая кластеризация пропущена (guard одной цепочки слотов), "
        "хотя в одной цепочке два разных голоса"
    )
    labels = [segment["speaker"] for segment in result]
    assert labels[0] != labels[1], (
        f"два разных голоса получили одну метку ({labels}): B прилип к цепочке A"
    )


# -------------------------------------------------------------------------- #
# Тупик дробления и адаптация порога (профиль «Совещание»)
# -------------------------------------------------------------------------- #
# Профиль «Совещание»: threshold 0.74, min_cluster_size 16; pyannote-путь
# берёт max(4, 16 // 2) = 8 (см. PyannoteSegmentationEngine.run_diarization).
THRESHOLD = 0.74
MIN_CLUSTER_SIZE = 8
SAMPLE_RATE = 16_000


def _voice_embeddings(
    per_voice: int,
    *,
    voices: int = 2,
    seed: int = 3,
    dim: int = 256,
    cross: float = 0.30,
    within: float = 0.66,
) -> np.ndarray:
    """*voices* × *per_voice* 256-мерных векторов: normalize(base + noise).

    Базы голосов имеют косинус *cross* (0.30); шум ортогонален базе голоса с
    полной |noise|² = 1/within − 1, поэтому внутри-голосовое сходство сэмплов
    ≈ *within* (0.66 — ниже порога 0.74), меж-голосовое ≈ within × cross ≈ 0.20.
    """
    rng = np.random.default_rng(seed)
    noise_power = 1.0 / within - 1.0
    base_a = np.zeros(dim)
    base_a[0] = 1.0
    base_b = np.zeros(dim)
    base_b[0] = cross
    base_b[1] = float(np.sqrt(1.0 - cross * cross))
    rows: list[np.ndarray] = []
    for base in [base_a, base_b][:voices]:
        for _ in range(per_voice):
            noise = rng.normal(size=dim)
            noise -= float(noise @ base) * base  # шум ортогонален базе
            noise *= float(np.sqrt(noise_power / float(noise @ noise)))
            vector = base + noise
            rows.append(vector / float(np.linalg.norm(vector)))
    return np.vstack(rows).astype(np.float32)


def _near_duplicate_embeddings(
    *,
    dups: int = 5,
    normals: int = 5,
    seed: int = 3,
    dim: int = 256,
    dup_cos: float = 0.99,
) -> np.ndarray:
    """2 голоса × (*dups* почти-дубликатов + *normals* обычных регионов).

    Почти-дубликаты (косинус ~*dup_cos* к якорю) держат p95 попарных сходств
    выше порога, хотя масса пар «своих» (~0.66) ниже порога — ловушка
    триггера адаптации по «лучшим» парам.
    """
    rng = np.random.default_rng(seed)
    noise_power = 1.0 / 0.66 - 1.0
    delta_power = 1.0 / dup_cos**2 - 1.0
    base_a = np.zeros(dim)
    base_a[0] = 1.0
    base_b = np.zeros(dim)
    base_b[0] = 0.30
    base_b[1] = float(np.sqrt(1.0 - 0.30 * 0.30))
    rows: list[np.ndarray] = []
    for base in (base_a, base_b):
        anchor = None
        for _ in range(dups):
            if anchor is None:
                noise = rng.normal(size=dim)
                noise -= float(noise @ base) * base
                noise *= float(np.sqrt(noise_power / float(noise @ noise)))
                vector = base + noise
                anchor = vector / float(np.linalg.norm(vector))
                rows.append(anchor)
                continue
            delta = rng.normal(size=dim)
            delta -= float(delta @ anchor) * anchor
            delta *= float(np.sqrt(delta_power / float(delta @ delta)))
            vector = anchor + delta
            rows.append(vector / float(np.linalg.norm(vector)))
        for _ in range(normals):
            noise = rng.normal(size=dim)
            noise -= float(noise @ base) * base
            noise *= float(np.sqrt(noise_power / float(noise @ noise)))
            vector = base + noise
            rows.append(vector / float(np.linalg.norm(vector)))
    return np.vstack(rows).astype(np.float32)


def _pairwise_stats(embeddings: np.ndarray) -> tuple[float, float, float, float]:
    """(среднее внутри-голосовое, среднее меж-голосовое, p95, p85) попарных косинусов."""
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = unit @ unit.T
    count = len(embeddings)
    upper = sims[np.triu_indices(count, k=1)]
    half = count // 2
    same = np.zeros((count, count), dtype=bool)
    same[:half, :half] = True
    same[half:, half:] = True
    np.fill_diagonal(same, False)
    within = float(sims[same].mean())
    cross = float(sims[~same].mean())
    return within, cross, float(np.percentile(upper, 95)), float(np.percentile(upper, 85))


class QueueSpeakerEmbedding:
    """Фейковый SE: раздаёт заготовленные векторы по порядку клипов.

    Как FakeSpeakerEmbedding, но векторы берутся из очереди: _embed вызывает
    embedding батчами по EMBED_BATCH в порядке следования регионов, так что
    каждый регион получает свой вектор.
    """

    def __init__(self, vectors: np.ndarray) -> None:
        self._queue = [np.asarray(vector, dtype=np.float32) for vector in vectors]

    def embedding(self, waveform: object) -> np.ndarray:
        clips = waveform if isinstance(waveform, list) else [waveform]
        vectors = [self._queue.pop(0) for _ in clips]
        return np.asarray(vectors, dtype=np.float32)


class BurstPyannoteSession:
    """Один голос в виде цепочки коротких реплик, как ChainBreakPyannoteSession.

    Каждое 10-с окно возвращает 100 фреймов (0.1 с на фрейм); активны первые
    20 фреймов — всплеск 2 с в начале окна; слот чередуется 2 -> 3 -> 2 ...
    (реплика после паузы попадает в новый слот сети).  PIT-выравнивание не
    смешивает цепочки: на участке перекрытия окон предыдущее окно молчит.
    Файл длиной 5*N + 5 с даёт ровно N окон и N регионов.
    """

    def __init__(self, frames: int = 100, burst_frames: int = 20) -> None:
        self._frames = frames
        self._burst_frames = burst_frames
        self._calls = 0

    def run(self, _outputs: object, _feeds: object) -> list[np.ndarray]:
        logits = np.full((1, self._frames, 7), -10.0, dtype=np.float32)
        slot = 2 if self._calls % 2 == 0 else 3
        logits[:, : self._burst_frames, slot] = 0.0
        self._calls += 1
        return [logits]


def _burst_setup(
    regions: int, embeddings: np.ndarray, monkeypatch
) -> tuple[int, np.ndarray, list[dict[str, object]]]:
    """Фейковые pyannote + SE: *regions* двухсекундных реплик одного голоса."""
    monkeypatch.setattr(
        diarization, "load_pyannote", lambda *_args, **_kwargs: BurstPyannoteSession()
    )
    monkeypatch.setattr(
        diarization,
        "load_se",
        lambda *_args, **_kwargs: QueueSpeakerEmbedding(embeddings),
    )
    audio = np.zeros((5 * regions + 5) * SAMPLE_RATE, dtype=np.float32)
    segments = [
        {"start_s": float(5 * index), "end_s": float(5 * index + 2), "text": f"реплика {index}"}
        for index in range(regions)
    ]
    return SAMPLE_RATE, audio, segments


def test_fragmented_voices_stay_split_without_large_clusters() -> None:
    """Тупик дробления разрешается даже без «крупных» кластеров.

    Внутри-голосовое сходство 0.66 ниже порога: ни один регион не может
    присоединиться к синглтону-центроиду, все 16 регионов открывают свои
    кластеры, и «крупных» (≥ min_cluster_size=8) нет — приклеивание молчит.
    Жадное склеивание мелких кластеров по центроидам обязано собрать каждому
    голосу ровно одну метку: два голоса — две метки.
    """
    embeddings = _voice_embeddings(8)  # 16 регионов: голос A (0-7), голос B (8-15)
    within, cross, _p95, _p85 = _pairwise_stats(embeddings)
    # Фактура синтетики: «свои» ниже порога, «чужие» — намного ниже.
    assert 0.55 <= within < THRESHOLD
    assert cross + 0.2 < within

    clusters = diarization._cluster_region_voices(
        embeddings, THRESHOLD, MIN_CLUSTER_SIZE
    )
    # Желаемое: два голоса -> две метки, каждому голосу — ровно одна.
    assert len(set(clusters[:8])) == 1
    assert len(set(clusters[8:])) == 1


def test_longer_file_adapts_threshold_and_merges() -> None:
    """Контроль: 30 регионов (2 голоса × 15) — адаптация спасает дробление."""
    embeddings = _voice_embeddings(15)  # 30 регионов
    threshold, adapted = diarization._adaptive_voice_threshold(embeddings, THRESHOLD)
    assert adapted is True
    assert 0.30 <= threshold < THRESHOLD

    clusters = diarization._cluster_region_voices(embeddings, threshold, MIN_CLUSTER_SIZE)
    assert len(set(clusters)) == 2
    assert len(set(clusters[:15])) == 1
    assert len(set(clusters[15:])) == 1


def test_adaptive_threshold_gate_blocks_short_files() -> None:
    """Короткий файл (16 регионов) тоже адаптируется.

    Фактура: p95 попарных сходств 0.69 < 0.74 — запись «сжата», дробление
    уже есть; ворота на адаптацию не должны отсекать файлы короче 20
    регионов.  Порог понижается ниже профильного.
    """
    embeddings = _voice_embeddings(8)  # 16 регионов
    _within, _cross, p95, _p85 = _pairwise_stats(embeddings)
    assert p95 < THRESHOLD

    threshold, adapted = diarization._adaptive_voice_threshold(embeddings, THRESHOLD)
    assert adapted is True
    assert threshold < THRESHOLD


def test_adaptive_threshold_p95_trigger_misses_degraded_mix() -> None:
    """Триггер адаптации смотрит на типичную пару (p85), а не на «лучшую» (p95).

    Несколько почти-дублирующихся пар (косинус ~0.99) держат p95 = 0.885 выше
    порога, тогда как масса пар «своих» ниже (p85 = 0.67 < 0.74) и тупик
    дробления уже наступил.  Адаптация обязана сработать, несмотря на
    «лучшие» пары.
    """
    embeddings = _near_duplicate_embeddings()  # 20 регионов
    _within, _cross, p95, p85 = _pairwise_stats(embeddings)
    assert p95 >= THRESHOLD
    assert p85 < THRESHOLD

    threshold, adapted = diarization._adaptive_voice_threshold(embeddings, THRESHOLD)
    assert adapted is True
    assert threshold < THRESHOLD


def test_relabel_regions_by_voice_merges_fragmented_voice(monkeypatch) -> None:
    """_relabel_regions_by_voice обязан дать каждому голосу одну метку.

    Две цепочки слотов по 8 регионов (голос A — регионы 0-7, голос B — 8-15,
    эмбеддинги раздаются очередью): голос A не должен распасться на 8 меток,
    голос B — на 4; цепочки одного голоса воссоединяются, разные голоса не
    сливаются.
    """
    embeddings = _voice_embeddings(8)  # 16 регионов: голос A (0-7), голос B (8-15)
    monkeypatch.setattr(
        diarization,
        "load_se",
        lambda *_args, **_kwargs: QueueSpeakerEmbedding(embeddings),
    )
    regions = (
        [("Спикер 1", float(index), float(index + 1)) for index in range(8)]
        + [("Спикер 2", float(8 + index), float(9 + index)) for index in range(8)]
    )
    audio = np.zeros(16 * SAMPLE_RATE, dtype=np.float32)

    result, _assigner = diarization._relabel_regions_by_voice(
        regions,
        audio,
        SAMPLE_RATE,
        cpu_threads=0,
        threshold=THRESHOLD,
        min_cluster_size=MIN_CLUSTER_SIZE,
    )
    labels = [label for label, _start, _end in result]
    assert len(set(labels[:8])) == 1
    assert len(set(labels[8:])) == 1
    assert labels[0] != labels[8]  # разные голоса не сливаются


class NoisyShortSpeakerEmbedding:
    """Длинные клипы — чистые векторы голосов, короткие — шум около них.

    Голос A — [1, 0, 0], голос B — [0, 1, 0] (по знаку среднего клипа).
    Короткий клип A даёт [1, 0.6, 0.8] (сходство с A 0,71), короткий B —
    [0.6, 1, 0.8]: ниже порога, но argmax указывает на свой голос — как у
    реальных коротких вставок (попарное сходство 0,375 на записи из проверки).
    """

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate

    def embedding(self, waveform: object) -> np.ndarray:
        clips = waveform if isinstance(waveform, list) else [waveform]
        vectors = []
        for clip in clips:
            sign = 1.0 if float(np.mean(clip)) >= 0.0 else -1.0
            long_clip = clip.size >= int(1.2 * self._sample_rate)
            if long_clip:
                base = [1.0, 0.0, 0.0] if sign > 0 else [0.0, 1.0, 0.0]
            else:
                base = [1.0, 0.6, 0.8] if sign > 0 else [0.6, 1.0, 0.8]
            vectors.append(base)
        return np.asarray(vectors, dtype=np.float32)


def test_short_regions_do_not_found_junk_cluster(monkeypatch) -> None:
    """Короткие вставки не основывают мусорный кластер.

    На реальной записи 30 коротких вставок из четырёх голосов выжили как
    «крупный» кластер (мусорный «Спикер 2»): их шумные эмбеддинги не дотягивали
    до порога, но кластер был велик для приклейки.  Теперь короткий регион
    присоединяется к лучшему кластеру по среднему сходству с членами, и
    сторонних меток не появляется.
    """
    monkeypatch.setattr(
        diarization,
        "load_se",
        lambda *_args, **_kwargs: NoisyShortSpeakerEmbedding(SAMPLE_RATE),
    )
    # 3 длинных региона голоса A (0-2, 4-6, 8-10), 3 голоса B (40-42, 44-46,
    # 48-50) и по 3 коротких вставки каждого голоса (0,8 с).
    regions = (
        [("Спикер 1", float(start), float(start + 2)) for start in (0.0, 4.0, 8.0)]
        + [("Спикер 1", float(start), float(start + 2)) for start in (40.0, 44.0, 48.0)]
        + [("Спикер 1", float(start), float(start + 0.8)) for start in (12.0, 16.0, 20.0)]
        + [("Спикер 1", float(start), float(start + 0.8)) for start in (52.0, 56.0, 60.0)]
    )
    audio = np.zeros(64 * SAMPLE_RATE, dtype=np.float32)
    for start, end, level in [
        (0.0, 2.0, 0.5), (4.0, 6.0, 0.5), (8.0, 10.0, 0.5),
        (12.0, 12.8, 0.5), (16.0, 16.8, 0.5), (20.0, 20.8, 0.5),
        (40.0, 42.0, -0.5), (44.0, 46.0, -0.5), (48.0, 50.0, -0.5),
        (52.0, 52.8, -0.5), (56.0, 56.8, -0.5), (60.0, 60.8, -0.5),
    ]:
        audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] = level

    result, assigner = diarization._relabel_regions_by_voice(
        regions,
        audio,
        SAMPLE_RATE,
        cpu_threads=0,
        threshold=0.75,
        min_cluster_size=3,
    )
    labels = [label for label, _start, _end in result]
    voice_a = labels[0]
    voice_b = labels[3]
    assert voice_a != voice_b
    assert len(set(labels)) == 2  # короткие вставки не завели третий спикер
    # Короткие вставки присоединились к СВОИМ голосам, а не к одному мусорному.
    assert labels[6] == labels[7] == labels[8] == voice_a
    assert labels[9] == labels[10] == labels[11] == voice_b
    # И тембр отдельного клипа вне регионов разрешается в метку голоса.
    assert assigner is not None
    gap_clip = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    assert assigner(gap_clip) == voice_a


def test_gap_segment_gets_speaker_by_voice(monkeypatch) -> None:
    """Фрагмент вне регионов pyannote получает спикера по тембру, а не
    «продолжает предыдущего».

    Голос B вставляет реплику в паузе между регионами голоса A: pyannote
    региона под неё не дал, и прежний откат записывал её за предыдущим
    спикером (A).  Эмбеддинг самого фрагмента указывает на голос B.
    """

    class GapPyannoteSession:
        """Слоты активны 0-2.5 с и 4.5-7 с; 2.5-4.5 с — пауза без региона."""

        def run(self, _outputs: object, _feeds: object) -> list[np.ndarray]:
            frames = 20  # 10-с окно -> 0.5 с на фрейм
            logits = np.full((1, frames, 7), -10.0, dtype=np.float32)
            logits[:, 0:5, 2] = 0.0
            logits[:, 9:14, 3] = 0.0
            return [logits]

    monkeypatch.setattr(
        diarization,
        "load_pyannote",
        lambda *_args, **_kwargs: GapPyannoteSession(),
    )
    monkeypatch.setattr(
        diarization, "load_se", lambda *_args, **_kwargs: FakeSpeakerEmbedding()
    )
    sample_rate = SAMPLE_RATE
    audio = np.zeros(10 * sample_rate, dtype=np.float32)
    audio[0 : int(2.5 * sample_rate)] = 0.5  # голос A
    audio[int(2.5 * sample_rate) : int(4.5 * sample_rate)] = -0.5  # вставка B
    audio[int(4.5 * sample_rate) : int(7.0 * sample_rate)] = -0.5  # голос B
    segments = [{"start_s": 2.8, "end_s": 4.2, "text": "вставка"}]

    result = diarization.diarize_segments(
        segments,
        audio,
        sample_rate,
        repo_id="onnx-community/pyannote-segmentation-3.0",
    )

    # Регионы покрывают 0-2.5 и 4.5-7.0; фрагмент 2.8-4.2 — дыра.
    assert [segment["speaker"] for segment in result] == ["Спикер 2"]


def test_end_to_end_chain_break_fragments_after_pause(monkeypatch) -> None:
    """16 реплик одного голоса двумя чередующимися цепочками — метка одна.

    Сквозной прогон (fake-pyannote на манер ChainBreakPyannoteSession + фейковый
    SE с внутри-голосовым косинусом 0.66): без разрешения тупика дробления
    один голос получал 12 меток («Спикер 1»..«Спикер 12»).
    """
    embeddings = _voice_embeddings(16, voices=1)  # 16 регионов одного голоса
    sample_rate, audio, segments = _burst_setup(16, embeddings, monkeypatch)

    result = diarization.diarize_segments(
        segments,
        audio,
        sample_rate,
        repo_id="onnx-community/pyannote-segmentation-3.0",
        threshold=THRESHOLD,
    )
    speakers = [segment["speaker"] for segment in result]
    assert set(speakers) == {"Спикер 1"}


def test_end_to_end_longer_file_rejoins_after_pause(monkeypatch) -> None:
    """Контроль: 30 реплик того же голоса — адаптация спасает, метка одна."""
    embeddings = _voice_embeddings(30, voices=1)  # 30 регионов одного голоса
    sample_rate, audio, segments = _burst_setup(30, embeddings, monkeypatch)

    result = diarization.diarize_segments(
        segments,
        audio,
        sample_rate,
        repo_id="onnx-community/pyannote-segmentation-3.0",
        threshold=THRESHOLD,
    )
    speakers = [segment["speaker"] for segment in result]
    assert set(speakers) == {"Спикер 1"}


# -------------------------------------------------------------------------- #
# Двойное отсечение малоречевых участников
# -------------------------------------------------------------------------- #
def _voice_vector(index: int, dim: int = 64, noise: float = 0.045) -> np.ndarray:
    """Почти ортогональный «тембр»: единичный базис + шум 0.045 на координату.

    Сходство независимых зашумлённых векторов одного голоса ~0.89, разных —
    около нуля (проверено: sim(A, B) ≈ -0.09 при фиксированных сиднах).
    """
    rng = np.random.default_rng(101 + index)
    base = np.zeros(dim, dtype=np.float64)
    base[index] = 1.0
    return base + rng.normal(0.0, noise, dim)


class TimelinePyannoteSession:
    """Fake pyannote: активность слотов задана на абсолютной шкале времени.

    Окно не знает своего смещения, поэтому аудио — линейный ramp: первый
    отсчёт каждого окна равен его смещению в секундах (5-с шаг окон даёт
    точные значения 0/5/10/15).  20 фреймов на 10-с окно (0.5 с на фрейм, как
    у реальной сети): фрейм активен для слота, если его середина попадает в
    интервал речи.  Логиты — log-softmax: активный класс 0.0, остальные
    -10.0 (как в соседних фейках).
    """

    def __init__(self, speech: dict[int, list[tuple[float, float]]]) -> None:
        # Ключ — номер сырого класса сети (1..3), значение — интервалы речи (с).
        self._speech = speech

    def run(self, _outputs: object, feeds: object) -> list[np.ndarray]:
        window = np.asarray(feeds["input_values"], dtype=np.float32)
        start_s = float(window[0, 0, 0])
        frames = 20
        frame_s = diarization.PYANNOTE_WINDOW_DURATION_S / frames
        logits = np.full((1, frames, 7), -10.0, dtype=np.float32)
        for frame in range(frames):
            middle = start_s + (frame + 0.5) * frame_s
            for slot, spans in self._speech.items():
                if any(begin <= middle < end for begin, end in spans):
                    logits[:, frame, slot] = 0.0
        return [logits]


class FramePyannoteSession:
    """Fake pyannote: перечисленные фреймы (по 0.5 с) активны для слота.

    Аудио 10 с — одно окно из 20 фреймов; паузы между активными фреймами
    (≥ min_duration_off = 0 без профиля) дают отдельные регионы: 10 регионов
    голоса A и 5 регионов голоса B.
    """

    def __init__(self, active_frames: dict[int, list[int]]) -> None:
        self._active = active_frames

    def run(self, _outputs: object, _feeds: object) -> list[np.ndarray]:
        frames = 20
        logits = np.full((1, frames, 7), -10.0, dtype=np.float32)
        for slot, frame_ids in self._active.items():
            for frame in frame_ids:
                logits[:, frame, slot] = 0.0
        return [logits]


def test_small_voice_cluster_glued_into_large_at_default_min_size() -> None:
    """Кластер из 6 регионов при минимуме 8 не приклеивается к чужому голосу.

    Без профиля движок считает минимум 8 регионов, и приклеивание забирало
    весь малый кластер B целиком.  Теперь приклеиваются только кластеры не
    больше половины минимума: кластер, близкий к минимуму, — реальный
    малоречивый участник.  Контроль разделимости: при заниженном минимуме 4
    голоса и так расходятся — векторы разделимы, виноват только размерный
    порог.
    """
    voice_a = _voice_vector(0)
    voice_b = _voice_vector(1)
    similarity = float(
        voice_a @ voice_b / (np.linalg.norm(voice_a) * np.linalg.norm(voice_b))
    )
    assert similarity < 0.5  # «чужой» голос заведомо ниже порога 0.74

    embeddings = np.vstack([voice_a] * 20 + [voice_b] * 6).astype(np.float32)

    # Контроль разделимости: при заниженном минимуме голоса расходятся.
    loose = diarization._cluster_region_voices(embeddings, 0.74, min_cluster_size=4)
    assert loose == [0] * 20 + [1] * 6

    # Дефолтный минимум 8: кластер B из 6 регионов остаётся самостоятельным.
    strict = diarization._cluster_region_voices(embeddings, 0.74, min_cluster_size=8)
    assert strict == [0] * 20 + [1] * 6


def test_short_slot_participant_inherits_neighbour_label(monkeypatch) -> None:
    """Слот с 1.5 с речи проходит фильтр — реплика C получает свою метку.

    Слоты: A [0, 6) с (6 с), B [11, 16) с (5 с), C [21, 22.5) с (1.5 с).
    Фильтр MIN_SLOT_SPEECH_S отсекает только блики короче 1.5 с: раньше порог
    в 2 с отбрасывал и реального участника — его реплика не перекрывалась ни
    с одним регионом и наследовала метку предыдущего сегмента («Спикер 2»).
    C обязан получить свою метку «Спикер 3».
    """
    monkeypatch.setattr(
        diarization,
        "load_pyannote",
        lambda *_args, **_kwargs: TimelinePyannoteSession(
            {1: [(0.0, 6.0)], 2: [(11.0, 16.0)], 3: [(21.0, 22.5)]}
        ),
    )
    _disable_voice_clustering(monkeypatch)

    sample_rate = 16_000
    # Ramp: первый отсчёт каждого окна кодирует его смещение в секундах.
    audio = np.arange(int(22.5 * sample_rate), dtype=np.float32) / sample_rate
    segments = [
        {"start_s": 1.0, "end_s": 5.0, "text": "one"},
        {"start_s": 12.0, "end_s": 15.0, "text": "two"},
        {"start_s": 21.2, "end_s": 22.3, "text": "three"},
    ]

    result = diarization.diarize_segments(
        segments, audio, sample_rate, repo_id="onnx-community/pyannote-segmentation-3.0"
    )

    assert [segment["speaker"] for segment in result] == [
        "Спикер 1",
        "Спикер 2",
        "Спикер 3",
    ]


def test_five_region_voice_loses_label_next_to_ten_region_voice(monkeypatch) -> None:
    """Голос из 5 регионов не приклеивается к голосу из 10 регионов.

    A: 10 однофреймовых регионов (5 с), B: 5 регионов (2.5 с) — оба слота
    проходят фильтр MIN_SLOT_SPEECH_S, тембры идеально разделимы ([1, ±1]).
    Кластер B из 5 регионов больше половины минимума 8 и остаётся
    самостоятельным: все 15 реплик получают две метки, а не одну.
    """
    a_frames = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]  # голос A: 10 регионов
    b_frames = [1, 5, 9, 13, 17]  # голос B: 5 регионов
    monkeypatch.setattr(
        diarization,
        "load_pyannote",
        lambda *_args, **_kwargs: FramePyannoteSession({1: a_frames, 2: b_frames}),
    )
    monkeypatch.setattr(
        diarization, "load_se", lambda *args, **kwargs: FakeSpeakerEmbedding()
    )

    sample_rate = 16_000
    audio = np.zeros(10 * sample_rate, dtype=np.float32)
    a_spans = [(0.5 * frame, 0.5 * frame + 0.5) for frame in a_frames]
    b_spans = [(0.5 * frame, 0.5 * frame + 0.5) for frame in b_frames]
    for start, end in a_spans:
        audio[int(start * sample_rate) : int(end * sample_rate)] = 0.5
    for start, end in b_spans:
        audio[int(start * sample_rate) : int(end * sample_rate)] = -0.5

    # По реплике на каждый регион; каждая реплика перекрывает только свой.
    turns = sorted(a_spans + b_spans)
    segments = [
        {"start_s": start + 0.02, "end_s": start + 0.42, "text": f"реплика {index}"}
        for index, (start, _end) in enumerate(turns)
    ]

    result = diarization.diarize_segments(
        segments, audio, sample_rate, repo_id="onnx-community/pyannote-segmentation-3.0"
    )

    assert {segment["speaker"] for segment in result} == {"Спикер 1", "Спикер 2"}


# -------------------------------------------------------------------------- #
# Нестабильность назначений: приклейка целиком, дрейф центроида, cap
# -------------------------------------------------------------------------- #
def test_glue_scatters_one_small_cluster_between_two_large() -> None:
    """Мелкий кластер приклеивается к крупному ЦЕЛИКОМ, а не по-регионно.

    20 регионов A и 20 регионов B (внутри кластера косинус 1.0, между A и B —
    0.30) и малоречивый участник C: 4 региона одним блоком, каждый ниже порога
    0.74 до чужих центроидов, но взаимно связный (0.79).  Приклейка по-регионно
    разъезжала его между крупными ([B, B, B, A]); теперь весь кластер C идёт
    к ближайшему крупному по сходству центроидов — одна метка на все четыре
    региона.  Контроль посылки: без приклеивания (min_cluster_size=1) регионы
    C и так остаются одним кластером.
    """
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.3, np.sqrt(1.0 - 0.3**2), 0.0])  # cos(a, b) = 0.30
    z = np.array([0.0, 0.0, 1.0])
    # Голос C: наклонён к B либо к A (косинус 0.55), до чужих центроидов ниже
    # порога 0.74, взаимное сходство 0.79 — векторы одного человека.
    c_near_b = 0.55 * b + np.sqrt(1.0 - 0.55**2) * z
    c_near_a = 0.55 * a + np.sqrt(1.0 - 0.55**2) * z
    embeddings = np.asarray([a] * 20 + [b] * 20 + [c_near_b] * 3 + [c_near_a], dtype=np.float32)

    premise = diarization._cluster_region_voices(embeddings, 0.74, min_cluster_size=1)
    assert len(set(premise[40:44])) == 1, premise[40:44]

    clusters = diarization._cluster_region_voices(embeddings, 0.74, min_cluster_size=8)
    assert len(set(clusters[40:44])) == 1, clusters[40:44]


def test_centroid_drift_captures_second_voice_without_threshold_violation() -> None:
    """Дрейф центроида не захватывает второй голос: кластер обязан разрезаться.

    Точные 2D-векторы: a=(1,0), b=(0.6,0.8) — косинус 0.60 ниже порога 0.74;
    blend=normalize(a+b) — бленд оверлапа.  Последовательность 8×a + 6×blend +
    6×b: центроид кластера A тянется блендами, и первый чистый регион B входит
    при cos 0.746 >= 0.74 — порог не нарушен ни в одном отдельном решении.
    Перепроверка обязана разрезать кластер: среднее сходство между кусками
    {8a+6blend}/{6b} равно 0.726 < 0.74, внутри кусков — не ниже (0.944, 1.0).
    Чистые A (регионы 0-7) и чистые B (14-19) — разные метки.  Контроль
    границы захвата: при 4 блендах дрейф не дотягивает (первый чистый B видит
    центроид под cos 0.715 < 0.74) и голос B остаётся отдельным кластером.
    """
    a = np.array([1.0, 0.0])
    b = np.array([0.6, 0.8])
    blend = (a + b) / np.linalg.norm(a + b)

    six_blends = np.asarray([a] * 8 + [blend] * 6 + [b] * 6, dtype=np.float32)
    clusters = diarization._cluster_region_voices(six_blends, 0.74, min_cluster_size=1)
    assert set(clusters[:8]).isdisjoint(set(clusters[14:])), clusters

    # Контроль границы захвата: на 4 блендах дрейф ещё не захватывает B.
    four_blends = np.asarray([a] * 8 + [blend] * 4 + [b] * 6, dtype=np.float32)
    clusters4 = diarization._cluster_region_voices(four_blends, 0.74, min_cluster_size=1)
    assert set(clusters4[:8]).isdisjoint(set(clusters4[12:])), clusters4


def test_cap_exhaustion_absorbs_thirteenth_voice_into_foreign_cluster() -> None:
    """13-й голос не поглощается после исчерпания cap.

    13 ортогональных голосов (ортонормальный базис 256-мерного пространства)
    по 3 региона: порог 0.90, все межголосовые сходства 0.0.  Контроль с
    cap=13 подтверждает, что данным хватает потолка.  Дефолт cap=16 тоже даёт
    13 меток, и 13-й голос не попадает в чужой кластер (раньше дефолт cap=12
    исчерпывался на двенадцатом голосе, и тринадцатый приклеивался к
    ближайшему кластеру с cos 0.0).
    """
    basis = np.eye(256, dtype=np.float32)
    embeddings = np.vstack([np.vstack([basis[i]] * 3) for i in range(13)])

    control = diarization._cluster_region_voices(
        embeddings, 0.90, min_cluster_size=1, cap=13
    )
    assert len(set(control)) == 13

    clusters = diarization._cluster_region_voices(embeddings, 0.90, min_cluster_size=1)
    # 13 реальных голосов должны получить 13 меток...
    assert len(set(clusters)) == 13, sorted(set(clusters))
    # ...и 13-й голос не должен оказаться в чужом кластере.
    assert set(clusters[36:39]).isdisjoint(set(clusters[:36])), clusters[36:39]
