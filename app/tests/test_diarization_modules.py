"""Тесты модульной системы диаризации: реестр движков, группы, YAML-конфиг."""

from __future__ import annotations

import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from airec.asr import diarization, diarization_profiles
from airec.asr.diarization_profiles import (
    PROFILES_FILE_NAME,
    load_diarization_profiles,
    profiles_file_exists,
)
from airec.models.manifest import load_manifest, manifest_dir


# --------------------------------------------------------------------------
# Гистерезис и постобработка регионов
# --------------------------------------------------------------------------
class TestHysteresisMask:
    def test_enters_on_onset_exits_on_offset(self) -> None:
        # Вероятность поднимается до 0.6 (>= onset 0.54) и держится на 0.45:
        # это ниже onset, но выше offset 0.42 — активность должна сохраниться.
        probs = np.array([0.1, 0.6, 0.45, 0.45, 0.41, 0.1], dtype=np.float32)
        mask = diarization._hysteresis_mask(probs, onset=0.54, offset=0.42)
        assert list(mask) == [False, True, True, True, False, False]

    def test_does_not_enter_without_onset(self) -> None:
        # Пик между offset и onset не открывает активность (нижняя база).
        probs = np.array([0.1, 0.45, 0.5, 0.45, 0.1], dtype=np.float32)
        mask = diarization._hysteresis_mask(probs, onset=0.54, offset=0.42)
        assert not mask.any()

    def test_chunked_processing_matches_plain(self) -> None:
        rng = np.random.default_rng(42)
        probs = rng.random(20_000).astype(np.float32)
        plain = diarization._hysteresis_mask(probs, 0.54, 0.42, chunk=1_000_000)
        chunked = diarization._hysteresis_mask(probs, 0.54, 0.42, chunk=997)
        assert np.array_equal(plain, chunked)


class TestRegionsFromMask:
    def test_fill_short_pauses(self) -> None:
        mask = np.zeros(100, dtype=bool)
        mask[10:30] = True
        mask[35:50] = True  # пауза 5 < min_duration_off 6 — склеится
        regions = diarization._regions_from_mask(
            mask, 100, min_duration_off=6, pad_onset=0, pad_offset=0, min_duration_on=1
        )
        assert regions == [(10, 50)]

    def test_pad_extends_both_sides(self) -> None:
        mask = np.zeros(100, dtype=bool)
        mask[20:40] = True
        regions = diarization._regions_from_mask(
            mask, 100, min_duration_off=0, pad_onset=5, pad_offset=3, min_duration_on=1
        )
        assert regions == [(15, 43)]

    def test_pad_does_not_cross_neighbours(self) -> None:
        mask = np.zeros(100, dtype=bool)
        mask[20:40] = True
        mask[44:60] = True
        regions = diarization._regions_from_mask(
            mask, 100, min_duration_off=0, pad_onset=5, pad_offset=5, min_duration_on=1
        )
        # Регионы соприкасаются (pad до края соседа), но не пересекаются.
        assert regions == [(15, 45), (45, 65)]
        for first, second in pairwise(regions):
            assert first[0] < first[1] <= second[0]

    def test_drop_short_regions_last(self) -> None:
        # Регион 3 сэмпла появится после паддинга, но короче min_duration_on 4.
        mask = np.zeros(100, dtype=bool)
        mask[50:53] = True
        regions = diarization._regions_from_mask(
            mask, 100, min_duration_off=0, pad_onset=0, pad_offset=0, min_duration_on=4
        )
        assert regions == []


# --------------------------------------------------------------------------
# Реестр движков (Strategy/Registry)
# --------------------------------------------------------------------------
class TestRegistry:
    def test_pyannote_selected_by_substring(self) -> None:
        cls = diarization.REGISTRY.engine_class_for("onnx-community/pyannote-segmentation-3.0")
        assert cls is diarization.PyannoteSegmentationEngine

    def test_wespeaker_selected_by_substring(self) -> None:
        cls = diarization.REGISTRY.engine_class_for("Wespeaker/wespeaker-voxceleb-resnet34")
        assert cls is diarization.WeSpeakerResNetEngine

    def test_local_path_selects_engine(self) -> None:
        cls = diarization.REGISTRY.engine_class_for(r"D:\models\pyannote-segmentation")
        assert cls is diarization.PyannoteSegmentationEngine
        cls = diarization.REGISTRY.engine_class_for(r"D:\models\wespeaker-resnet103")
        assert cls is diarization.WeSpeakerResNetEngine

    def test_unknown_key_falls_back_to_wespeaker(self) -> None:
        cls = diarization.REGISTRY.engine_class_for(r"D:\models\some-custom-onnx")
        assert cls is diarization.WeSpeakerResNetEngine

    def test_known_engines(self) -> None:
        ids = [engine_id for engine_id, _label in diarization.REGISTRY.known_engines()]
        assert "pyannote-segmentation-3.0" in ids
        assert "wespeaker" in ids


class TestEngineProfiles:
    def _profiles(self) -> diarization_profiles.DiarizationProfiles:
        return diarization_profiles._profiles_from_payload(None)

    def test_missing_profile_falls_back_to_first(self, caplog: pytest.LogCaptureFixture) -> None:
        engine = diarization.PyannoteSegmentationEngine(
            profiles=self._profiles(), profile_id="no-such-group"
        )
        with caplog.at_level("WARNING"):
            group = engine._resolve_group()
        assert group is not None and group.id == "meeting"
        assert "no-such-group" in caplog.text

    def test_group_is_cached(self) -> None:
        profiles = self._profiles()
        engine = diarization.WeSpeakerResNetEngine(profiles=profiles, profile_id="interview")
        first = engine._resolve_group()
        assert first is not None and first.method == "average"
        assert engine._resolve_group() is first

    def test_none_profiles_means_builtin_defaults(self) -> None:
        # profiles=None — прежнее поведение (встроенные дефолты, не файл):
        # на этот контракт опираются тесты и офлайн-харнесс.
        engine = diarization.PyannoteSegmentationEngine(profiles=None)
        assert engine._resolve_group() is None
        engine = diarization.WeSpeakerResNetEngine(profiles=None)
        assert engine._resolve_group() is None


# --------------------------------------------------------------------------
# WeSpeaker: average-linkage, min_cluster_size, перенумерация
# --------------------------------------------------------------------------
class TestWeSpeakerGroup:
    def _engine_run(self, monkeypatch, group_id: str, cluster_size: int):
        fake = FakeEmbedding(cluster_size=cluster_size)
        monkeypatch.setattr(diarization, "load_se", lambda *args, **kwargs: fake)
        profiles = diarization_profiles._profiles_from_payload(None)
        engine = diarization.WeSpeakerResNetEngine(profiles=profiles, profile_id=group_id)
        sample_rate = 16_000
        audio = np.zeros(6 * sample_rate, dtype=np.float32)
        segments = [
            {"start_s": float(i), "end_s": float(i + 1), "text": str(i)}
            for i in range(6)
        ]
        return engine.run_diarization(
            segments,
            audio,
            sample_rate,
            cpu_threads=0,
            model_path=None,
            repo_id=None,
            threshold=0.7,
            cancel=None,
            progress=None,
        )

    def test_interview_profile_uses_average_method(self, monkeypatch) -> None:
        # average-метод сравнивает со средним попарного сходства; результат
        # не должен падать и должен разложить сегменты по кластерам.
        result = self._engine_run(monkeypatch, "interview", cluster_size=4)
        speakers = {segment["speaker"] for segment in result}
        assert len(speakers) >= 1
        assert all(speaker is not None for speaker in speakers)

    def test_small_clusters_merge_and_renumber(self, monkeypatch) -> None:
        # 1 из 6 сегментов образует свой кластер (выброс): при
        # min_cluster_size=2 он приклеивается к ближайшему крупному,
        # нумерация остаётся плотной, без дыр.
        profiles = diarization_profiles._profiles_from_payload(None)
        profiles.wespeaker[2].min_cluster_size = 2  # группа conference
        fake = FakeEmbedding(cluster_size=5)
        monkeypatch.setattr(diarization, "load_se", lambda *args, **kwargs: fake)
        engine = diarization.WeSpeakerResNetEngine(profiles=profiles, profile_id="conference")
        sample_rate = 16_000
        audio = np.zeros(6 * sample_rate, dtype=np.float32)
        segments = [
            {"start_s": float(i), "end_s": float(i + 1), "text": str(i)}
            for i in range(6)
        ]
        result = engine.run_diarization(
            segments,
            audio,
            sample_rate,
            cpu_threads=0,
            model_path=None,
            repo_id=None,
            threshold=0.7,
            cancel=None,
            progress=None,
        )
        speakers = [segment["speaker"] for segment in result]
        unique = list(dict.fromkeys(speakers))
        assert len(unique) == 1, speakers


class FakeEmbedding:
    """Кластер из *cluster_size* одинаковых векторов + один выброс."""

    def __init__(self, cluster_size: int) -> None:
        self.cluster_size = cluster_size

    def embedding(self, waveform: object) -> np.ndarray:
        clips = waveform if isinstance(waveform, list) else [waveform]
        vectors = []
        for index in range(len(clips)):
            if index < self.cluster_size:
                vectors.append([1.0, 0.1])
            else:
                vectors.append([0.1, 1.0])
        return np.asarray(vectors, dtype=np.float32)


# --------------------------------------------------------------------------
# Загрузчик внешнего YAML-конфига
# --------------------------------------------------------------------------
class TestProfilesLoader:
    def test_bootstrap_creates_file(self, tmp_path: Path) -> None:
        assert not profiles_file_exists(tmp_path)
        profiles = load_diarization_profiles(tmp_path)
        assert (tmp_path / PROFILES_FILE_NAME).exists()
        assert profiles_file_exists(tmp_path)
        assert profiles.pyannote[0].onset == 0.54

    def test_no_data_dir_returns_defaults(self) -> None:
        profiles = load_diarization_profiles(None, bootstrap=False)
        assert profiles.gigaam.overlap_attribution_threshold == 0.45

    def test_broken_file_falls_back(self, tmp_path: Path) -> None:
        (tmp_path / PROFILES_FILE_NAME).write_text("pyannote: [broken", encoding="utf-8")
        profiles = load_diarization_profiles(tmp_path, bootstrap=False)
        assert profiles.pyannote[0].id == "meeting"

    def test_edits_picked_up_without_restart(self, tmp_path: Path) -> None:
        path = tmp_path / PROFILES_FILE_NAME
        load_diarization_profiles(tmp_path)
        time.sleep(0.02)
        path.write_text(
            "wespeaker:\n"
            "  profiles:\n"
            "    - id: meeting\n"
            "      threshold: 0.81\n",
            encoding="utf-8",
        )
        profiles = load_diarization_profiles(tmp_path)
        group = profiles.wespeaker_group("meeting")
        # Пользовательское значение перекрывает дефолт.
        assert group.threshold == 0.81
        # Нетронутые поля сохраняются из встроенных дефолтов.
        assert group.method == "centroid"

    def test_unknown_vad_keys_ignored(self, tmp_path: Path) -> None:
        (tmp_path / PROFILES_FILE_NAME).write_text(
            "vad:\n  threshold: 0.58\n  future_option: 7\n", encoding="utf-8"
        )
        profiles = load_diarization_profiles(tmp_path, bootstrap=False)
        assert profiles.vad.threshold == 0.58

    def test_gigaam_link_reaches_pyannote_engine(self, tmp_path: Path) -> None:
        profiles = load_diarization_profiles(None, bootstrap=False)
        engine = diarization.PyannoteSegmentationEngine(
            profiles=profiles, profile_id="meeting"
        )
        group = engine._resolve_group()
        assert group is not None
        # GigaamLink доходит до движка (используется при голосовании слов).
        assert engine._profiles.gigaam.ctc_delay_compensation_ms == 70


# --------------------------------------------------------------------------
# Манифест: обе модели на месте
# --------------------------------------------------------------------------
def test_manifest_has_both_engines() -> None:
    manifest = load_manifest(manifest_dir())
    kinds = {
        model.get("id"): model.get("kind")
        for model in manifest["models"]
        if model.get("kind") == "diarization"
    }
    assert "pyannote-segmentation-3.0" in kinds
    assert any("wespeaker" in model_id for model_id in kinds)
