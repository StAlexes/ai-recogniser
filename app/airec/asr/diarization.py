"""Speaker diarization: PyAnnote segmentation or WeSpeaker embeddings + clustering.

Two engines are available:

- ``pyannote`` (default) — `pyannote/segmentation-3.0` weights in ONNX
  (`onnx-community/pyannote-segmentation-3.0`, MIT).  The model is an end-to-end
  speaker-segmentation network: for every 10 s audio window it outputs per-frame
  probabilities of "no speaker", "speaker 1", "speaker 2", "speaker 3" and
  overlapping combinations.  Overlapping windows are fused so each frame gets
  up to three per-speaker activity probabilities; frames are then turned into
  speech regions per speaker.  The result is deterministic and captures
  overlapped speech that embedding clustering misses.
- ``wespeaker`` — WeSpeaker speaker embeddings + cosine clustering (the
  original engine, kept as an option; it does not cap the number of speakers).

For each VAD segment, the dominant speaker of the region it overlaps is
assigned as the segment label.  Models are downloaded from Hugging Face on
first use and cached there; a model downloaded through the model catalogue is
loaded from ``<data>/models/<id>/`` instead, so it works offline.  If the model
is unavailable, diarization is disabled with a warning — segments remain
without a speaker label, but recognition keeps working.
"""

from __future__ import annotations

import logging
import re
import threading
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from airec.asr.diarization_profiles import (
    DiarizationProfiles,
    GigaamLink,
    PyannoteGroup,
    WeSpeakerGroup,
)

log = logging.getLogger(__name__)

# Segments with cosine similarity >= this value are assigned to the same speaker.
SIMILARITY_THRESHOLD = 0.70

# Upstream onnx-asr test model; used when no model is selected (e.g. an old
# config without ``diarize_model``).
DEFAULT_SE_REPO = "Wespeaker/wespeaker-voxceleb-resnet34"

# PyAnnote segmentation (end-to-end speaker activity model, MIT licence).
PYANNOTE_REPO = "onnx-community/pyannote-segmentation-3.0"
# Per-frame probability above which a frame counts as "speaker is talking".
PYANNOTE_ONSET = 0.5
# A speaker region shorter than this is dropped (pyannote default).
PYANNOTE_MIN_DURATION_S = 0.2
# Sliding segmentation window (seconds) and the step between windows — the
# defaults mirror onnx-asr: 10 s window with a 5 s step (50% overlap).
PYANNOTE_WINDOW_DURATION_S = 10.0
PYANNOTE_WINDOW_STEP_S = 5.0
# A speaker slot with less total speech than this is treated as network noise
# (a hallucinated sub-second blip) and dropped before labels are numbered —
# otherwise the blip steals a number from the real voices.
MIN_SLOT_SPEECH_S = 2.0

# How many segments are embedded in one onnx-asr call.  Larger batches are
# faster but pad every clip to the longest one, so keep it modest.
EMBED_BATCH = 8
MIN_CLIP_SAMPLES = 400  # 25 ms at 16 kHz

_se: Any | None = None
_se_key: tuple[str, str, int] | None = None
_se_lock = threading.Lock()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


class DiarizationError(RuntimeError):
    """Диаризация недоступна (модель эмбеддингов не загружена)."""


class DiarizationCancelled(RuntimeError):
    """The user stopped the diarization stage."""


def load_se(
    cpu_threads: int = 0,
    model_path: str | Path | None = None,
    repo_id: str | None = None,
) -> Any:
    """Load and cache the WeSpeaker speaker-embedding model.

    If *model_path* points to a directory with a ``*.onnx`` file, the model is
    loaded from there (offline).  Otherwise *repo_id* (a Hugging Face repository
    id) is resolved/downloaded by onnx-asr.
    """
    global _se, _se_key

    local_dir = Path(model_path) if model_path else None
    repo = repo_id or DEFAULT_SE_REPO
    key = (repo, str(local_dir or ""), cpu_threads)

    with _se_lock:
        if _se is not None and _se_key == key:
            return _se

        try:
            import onnxruntime as ort
            from onnx_asr.loader import Manager
        except ImportError as exc:
            raise DiarizationError(
                "Компоненты диаризации не установлены. Установите extras «asr»."
            ) from exc

        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise DiarizationError("ONNX Runtime не предоставляет CPUExecutionProvider для диаризации.")

        session_options = None
        if cpu_threads > 0:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = cpu_threads
            session_options.inter_op_num_threads = 1

        source = str(local_dir) if local_dir else repo
        log.info(
            "loading speaker-embedding model %s (CPU, threads=%s)",
            source,
            cpu_threads or "auto",
        )
        try:
            manager = Manager(sess_options=session_options, providers=["CPUExecutionProvider"])
            if local_dir is not None:
                se = manager.create_se(repo, local_dir=local_dir)
            else:
                se = manager.create_se(repo)
        except Exception as exc:  # model download/read/init errors
            raise DiarizationError(f"Не удалось загрузить модель диаризации: {exc}") from exc

        _se = se
        _se_key = key
        log.info("speaker-embedding model loaded: %s", source)
        return _se


# --------------------------------------------------------------------------- #
# PyAnnote segmentation engine
# --------------------------------------------------------------------------- #

_pyannote: Any | None = None
_pyannote_key: tuple[str, str, int] | None = None


def load_pyannote(
    cpu_threads: int = 0,
    model_path: str | Path | None = None,
    repo_id: str | None = None,
) -> Any:
    """Load and cache the pyannote segmentation-3.0 ONNX session.

    The session is used directly (see :func:`_pyannote_speaker_regions`) because
    onnx-asr exposes the model only as VAD, not as a speaker diarizer.  When
    *model_path* is given the ``model.onnx`` file is loaded from that directory
    (offline); otherwise *repo_id* (default ``onnx-community/
    pyannote-segmentation-3.0``) is downloaded through huggingface_hub and
    cached in the standard HF cache.
    """
    global _pyannote, _pyannote_key

    local_dir = Path(model_path) if model_path else None
    repo = repo_id or PYANNOTE_REPO
    key = (repo, str(local_dir or ""), cpu_threads)

    with _se_lock:
        if _pyannote is not None and _pyannote_key == key:
            return _pyannote

        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise DiarizationError(
                "Компоненты диаризации не установлены. Установите extras «asr»."
            ) from exc

        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise DiarizationError("ONNX Runtime не предоставляет CPUExecutionProvider для диаризации.")

        session_options = None
        if cpu_threads > 0:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = cpu_threads
            session_options.inter_op_num_threads = 1

        if local_dir is not None:
            model_file = local_dir / "model.onnx"
            if not model_file.is_file():
                found = sorted(local_dir.rglob("model.onnx"))
                if not found:
                    raise DiarizationError(
                        f"В папке {local_dir} нет model.onnx (модель pyannote segmentation)."
                    )
                model_file = found[0]
        else:
            try:
                model_file = Path(hf_hub_download(repo, "onnx/model.onnx"))
            except Exception as exc:
                raise DiarizationError(
                    f"Не удалось загрузить модель pyannote segmentation: {exc}"
                ) from exc

        log.info(
            "loading pyannote segmentation model %s (CPU, threads=%s)",
            model_file,
            cpu_threads or "auto",
        )
        try:
            session = ort.InferenceSession(
                str(model_file),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise DiarizationError(f"Не удалось загрузить модель диаризации: {exc}") from exc

        _pyannote = session
        _pyannote_key = key
        return _pyannote


def _hysteresis_mask(
    averaged: np.ndarray, onset: float, offset: float, *, chunk: int = 2_500_000
) -> np.ndarray:
    """Binarise a per-frame speaker-probability curve with an onset/offset pair.

    Кадр переключается в активное состояние, когда вероятность достигает
    *onset*, и остаётся активным, пока вероятность не опустится до *offset*
    (гистерезис исключает мерцание метки на паузах).  Кривая обрабатывается
    блоками, чтобы не держать дополнительные копии длинного массива: состояние
    (кадр последнего включения/выключения) переносится между блоками.
    """
    total = averaged.shape[0]
    mask = np.empty(total, dtype=bool)
    last_enter = -1
    last_kill = -1
    for begin in range(0, total, chunk):
        end = min(begin + chunk, total)
        slice_probs = averaged[begin:end]
        above_onset = slice_probs >= onset
        above_offset = slice_probs > offset
        indices = np.arange(begin, end, dtype=np.int64)
        enter = np.maximum.accumulate(
            np.where(above_onset, indices, last_enter), axis=0
        )
        kill = np.maximum.accumulate(
            np.where(~above_offset, indices, last_kill), axis=0
        )
        mask[begin:end] = (enter > kill) & above_offset
        last_enter = int(enter[-1])
        last_kill = int(kill[-1])
    return mask


def _regions_from_mask(
    mask: np.ndarray,
    total: int,
    *,
    min_duration_off: int,
    pad_onset: int,
    pad_offset: int,
    min_duration_on: int,
) -> list[tuple[int, int]]:
    """Turn a binarised speaker curve into (start, end) sample regions.

    Порядок постобработки повторяет pyannote-конвейер: сперва заполняются
    паузы короче *min_duration_off* (склейка речевых вздохов), затем регионы
    расширяются на *pad_onset*/*pad_offset* (защита согласных), и только после
    этого отбрасываются регионы короче *min_duration_on*.
    """
    if not mask.any():
        return []
    edges = np.diff(mask.astype(np.int8))
    region_starts = (np.flatnonzero(edges == 1) + 1).tolist()
    region_ends = (np.flatnonzero(edges == -1) + 1).tolist()
    if mask[0]:
        region_starts.insert(0, 0)
    if mask[-1]:
        region_ends.append(total)
    regions = [(int(s), int(e)) for s, e in zip(region_starts, region_ends, strict=False)]

    # Fill pauses shorter than min_duration_off between neighbouring regions
    # of the same speaker slot.
    if min_duration_off > 0:
        filled: list[tuple[int, int]] = []
        for start, end in regions:
            if filled and start - filled[-1][1] < min_duration_off:
                filled[-1] = (filled[-1][0], end)
            else:
                filled.append((start, end))
        regions = filled

    # Padded regions keep their neighbours: a start may only grow leftwards
    # up to the previous region's end, an end only up to the next start.
    if pad_onset > 0 or pad_offset > 0:
        padded: list[tuple[int, int]] = []
        for start, end in regions:
            new_start = max(start - pad_onset, padded[-1][1] if padded else 0)
            new_end = end + pad_offset
            padded.append((new_start, new_end))
        regions = [
            (s, min(e, next_s))
            for (s, e), (_n, next_s) in zip(
                padded, [*padded[1:], (total, total)], strict=True
            )
        ]
    return [(s, e) for s, e in regions if e - s >= min_duration_on]


def _pyannote_speaker_regions(
    session: Any,
    audio: np.ndarray,
    sample_rate: int,
    *,
    onset: float = PYANNOTE_ONSET,
    offset: float | None = None,
    min_duration_on_s: float = PYANNOTE_MIN_DURATION_S,
    min_duration_off_s: float = 0.0,
    pad_onset_s: float = 0.0,
    pad_offset_s: float = 0.0,
    window_duration_s: float = PYANNOTE_WINDOW_DURATION_S,
    window_step_s: float = PYANNOTE_WINDOW_STEP_S,
    cancel: Callable[[], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> list[tuple[str, float, float]]:
    """Run pyannote segmentation and return per-speaker speech regions.

    Implements the same windowing/decoding math as onnx-asr's ``PyAnnoteVad``
    (configurable sliding window with period *window_step_s*, per-speaker
    probability fusion), but keeps all speaker channels instead of collapsing
    them into one speech channel, and yields ``(speaker, start_s, end_s)``
    regions.  Бинаризация использует гистерезис onset/offset (когда *offset*
    задан), заполняет паузы короче *min_duration_off_s*, расширяет регионы на
    *pad_onset_s*/*pad_offset_s* и отбрасывает регионы короче *min_duration_on_s*.

    The ONNX graph outputs 7 frames per class: {no-speech}, {spk1}, {spk2},
    {spk3}, {spk1+spk2}, {spk1+spk3}, {spk2+spk3}; multi-speaker classes add
    their probability to each participating speaker.
    """
    window_size = max(1, round(window_duration_s * sample_rate))
    advance = max(1, round(window_step_s * sample_rate))
    if offset is None:
        offset = onset

    total = len(audio)
    if total == 0:
        return []

    starts: list[int] = []
    offset_index = 0
    while offset_index + window_size <= total:
        starts.append(offset_index)
        offset_index += advance
    if starts:
        if starts[-1] + window_size < total:
            # Pad the tail window like onnx-asr does.
            starts.append(total - window_size)
    elif total <= window_size:
        starts = [0]
    else:
        starts = [total - window_size]

    num_speakers = 3
    probs = [np.zeros(total, dtype=np.float32) for _ in range(num_speakers)]
    counts = [np.zeros(total, dtype=np.float32) for _ in range(num_speakers)]
    frame_stride = 270  # ~589 frames per 10 s window

    def process(window: np.ndarray) -> np.ndarray:
        # ONNX input shape: (batch_size, num_channels, num_samples).
        logits = session.run(["logits"], {"input_values": window[None, None, :]})[0]
        output = np.exp(np.asarray(logits, dtype=np.float32))[0]
        merged = output.copy()
        # {spk1+spk2}, {spk1+spk3}, {spk2+spk3} contribute to both speakers.
        merged[:, 1] += output[:, 4] + output[:, 5]
        merged[:, 2] += output[:, 4] + output[:, 6]
        merged[:, 3] += output[:, 5] + output[:, 6]
        return merged

    for index, start in enumerate(starts):
        if cancel is not None and index % 4 == 0:
            cancel()
        window = audio[start : start + window_size]
        if window.shape[0] < window_size:
            window = np.pad(window, (0, window_size - window.shape[0]))
        merged = process(window)
        frames = merged.shape[0]
        # Frames map over the real window length, not stride * frames: the
        # last frame may be wider, and mocked/other models may return fewer
        # frames than the 270-sample stride implies.
        frame_samples = window_size / frames

        for frame in range(frames):
            begin = start + int(frame * frame_samples)
            end = min(start + int((frame + 1) * frame_samples), total)
            if begin >= total:
                break
            if end <= begin:
                end = min(begin + frame_stride, total)
            for spk in range(num_speakers):
                value = float(merged[frame, spk + 1])
                probs[spk][begin:end] += value
                counts[spk][begin:end] += 1.0
        if progress is not None:
            progress(
                0.55 * min((index + 1) / len(starts), 1.0),
                f"Анализ спикеров: окно {index + 1} из {len(starts)}",
            )

    # Region boundaries per raw speaker slot (sample indices).
    slot_regions: dict[int, list[tuple[int, int]]] = {}
    filter_off = int(min_duration_off_s * sample_rate)
    pad_on = int(pad_onset_s * sample_rate)
    pad_off = int(pad_offset_s * sample_rate)
    filter_on = int(min_duration_on_s * sample_rate)
    for spk in range(num_speakers):
        active = counts[spk] > 0
        averaged = np.divide(
            probs[spk], counts[spk], out=np.zeros_like(probs[spk]), where=active
        )
        if offset != onset:
            mask = _hysteresis_mask(averaged, onset, offset) & active
        else:
            mask = active & (averaged >= onset)
        slot_regions_for_spk = _regions_from_mask(
            mask,
            total,
            min_duration_off=filter_off,
            pad_onset=pad_on,
            pad_offset=pad_off,
            min_duration_on=filter_on,
        )
        if slot_regions_for_spk:
            slot_regions[spk] = slot_regions_for_spk

    if not slot_regions:
        return []

    # Drop slots whose total speech is negligible: the network sometimes
    # hallucinates a sub-second blip in one of its three slots, which would
    # otherwise steal a number from the real voices (the first voice then
    # gets labelled «Спикер 2»).
    slot_regions = {
        spk: regs
        for spk, regs in slot_regions.items()
        if sum(end - start for start, end in regs) / sample_rate >= MIN_SLOT_SPEECH_S
    }

    # Rename raw speaker slots to consecutive labels in order of first
    # temporal appearance: the network may pick any of its three slots (e.g.
    # classes 2 and 3 for two voices), but users expect the first person to
    # speak to be «Спикер 1», the next «Спикер 2», ...
    ordered = sorted(slot_regions.items(), key=lambda item: item[1][0][0])
    regions: list[tuple[str, float, float]] = []
    for number, (_spk, regs) in enumerate(ordered, start=1):
        label = f"Спикер {number}"
        regions.extend(
            (label, region_start / sample_rate, region_end / sample_rate)
            for region_start, region_end in regs
        )
    regions.sort(key=lambda item: item[1])
    return regions


def _embed(
    se: Any,
    clips: list[np.ndarray],
    *,
    cancel: Callable[[], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> np.ndarray:
    """Embed *clips* (1-D float32 arrays) and return an (N, dim) matrix."""
    usable: list[np.ndarray] = []
    for clip in clips:
        if clip.size < MIN_CLIP_SAMPLES:
            clip = np.pad(clip, (0, MIN_CLIP_SAMPLES - clip.size))
        usable.append(clip.astype(np.float32, copy=False))

    embeddings: list[np.ndarray] = []
    total = max(len(usable), 1)
    for offset in range(0, len(usable), EMBED_BATCH):
        if cancel is not None:
            cancel()
        batch = usable[offset : offset + EMBED_BATCH]
        # SeAdapter.embedding accepts a list of waveforms and returns (N, dim).
        result = np.asarray(se.embedding(batch), dtype=np.float32)
        if result.ndim == 1:
            result = result.reshape(1, -1)
        embeddings.append(result)
        if progress is not None:
            progress(
                0.9 * min((offset + len(batch)) / total, 1.0),
                f"Эмбеддинги спикеров: {offset + len(batch)} из {len(usable)}",
            )
    return np.vstack(embeddings)


def diarize_segments(
    segments: list[dict[str, Any]],
    audio: np.ndarray,
    sample_rate: int,
    *,
    cpu_threads: int = 0,
    threshold: float = SIMILARITY_THRESHOLD,
    model_path: str | Path | None = None,
    repo_id: str | None = None,
    profile_id: str | None = None,
    profiles: DiarizationProfiles | None = None,
    cancel: Callable[[], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Assign speaker labels to *segments* in-place and return them.

    *audio* — float32 PCM array at *sample_rate* (16 kHz).

    Движок выбирается реестром по текстовому ID: ключом служит *repo_id*
    (например, ``pyannote-segmentation-3.0``) либо путь к локальной модели.
    Ключи с ``pyannote`` выбирают энд-тю-энд сегментацию (сеть сама решает,
    какой из трёх слотов говорит), остальные — эмбеддинги WeSpeaker с
    косинусной кластеризацией.

    *profiles* — набор групп гиперпараметров (None -> встроенные дефолты);
    *profile_id* — активная группа внутри набора (например, ``interview``).
    Новые группы без перезапуска подхватываются на уровне менеджера, который
    перечитывает YAML-файл при каждом запуске стадии.  *cancel* pollится
    между дорогими шагами, чтобы остановленная стадия не крутила worker.
    """
    if not segments:
        return segments

    engine_class = REGISTRY.engine_class_for(
        repo_id or (str(model_path) if model_path else "")
    )
    engine = engine_class(profiles=profiles, profile_id=profile_id)
    return engine.run_diarization(
        segments,
        audio,
        sample_rate,
        cpu_threads=cpu_threads,
        model_path=model_path,
        repo_id=repo_id,
        threshold=threshold,
        cancel=cancel,
        progress=progress,
    )


def _diarize_with_pyannote(
    segments: list[dict[str, Any]],
    audio: np.ndarray,
    sample_rate: int,
    *,
    cpu_threads: int,
    model_path: str | Path | None,
    repo_id: str,
    group: PyannoteGroup | None = None,
    link: GigaamLink | None = None,
    cancel: Callable[[], None] | None,
    progress: Callable[[float, str], None] | None,
) -> list[dict[str, Any]]:
    """Label segments with the pyannote segmentation engine (up to 3 speakers).

    *group* — активная группа гиперпараметров (None -> встроенные дефолты);
    *link* — настройки связки с GigaAM (компенсация задержки CTC и допуск
    границ слов при голосовании).
    """
    if progress is not None:
        progress(0.0, "Загрузка модели спикеров")
    session = load_pyannote(cpu_threads, model_path=model_path, repo_id=repo_id)
    if cancel is not None:
        cancel()
    try:
        regions = _pyannote_speaker_regions(
            session,
            audio.astype(np.float32, copy=False),
            sample_rate,
            onset=group.onset if group else PYANNOTE_ONSET,
            offset=group.offset if group else None,
            min_duration_on_s=group.min_duration_on if group else PYANNOTE_MIN_DURATION_S,
            min_duration_off_s=group.min_duration_off if group else 0.0,
            pad_onset_s=group.pad_onset if group else 0.0,
            pad_offset_s=group.pad_offset if group else 0.0,
            window_duration_s=(
                group.window_duration if group else PYANNOTE_WINDOW_DURATION_S
            ),
            window_step_s=group.window_step if group else PYANNOTE_WINDOW_STEP_S,
            cancel=cancel,
            progress=progress,
        )
    except DiarizationCancelled:
        raise
    except Exception as exc:
        raise DiarizationError(f"Не удалось разобрать спикеров: {exc}") from exc

    if not regions:
        for segment in segments:
            segment["speaker"] = "Спикер 1"
        return segments

    # A VAD segment may contain several speakers: Silero only splits at real
    # pauses, and a speaker change is often seamless.  Segments with token
    # timestamps are cut at speaker turns; each piece keeps the tokens that
    # fall inside its own time range, so the text is split correctly too.
    vote_delay_s = (link.ctc_delay_compensation_ms / 1000.0) if link else 0.0
    boundary_margin_s = (
        link.word_boundary_margin_ms / 1000.0 if link else TURN_TOLERANCE_S
    )
    segments = _split_at_speaker_turns(
        segments,
        regions,
        vote_delay_s=vote_delay_s,
        boundary_margin_s=boundary_margin_s,
    )
    if len(segments) > 1:
        log.info("speaker turns: %d segments after split", len(segments))

    # Longest overlap wins; ties prefer the earlier region.  Segments that
    # overlap nothing (short interjections in the pauses between regions)
    # continue the previous segment's speaker instead of jumping to the very
    # first region — a real speaker turn needs a new region to appear.
    total = max(len(segments), 1)
    previous_label: str | None = None
    for position, segment in enumerate(segments):
        if cancel is not None and position % 32 == 0:
            cancel()
        start = float(segment.get("start_s", 0.0))
        end = max(float(segment.get("end_s", start)), start)
        best_label: str | None = None
        best_overlap = 0.0
        for label, region_start, region_end in regions:
            overlap = min(end, region_end) - max(start, region_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        segment["speaker"] = (
            segment.get("speaker") or best_label or previous_label or regions[0][0]
        )
        previous_label = segment["speaker"]
        if progress is not None and (position % 32 == 0 or position + 1 == total):
            progress(
                0.55 + 0.45 * ((position + 1) / total),
                f"Разметка сегментов: {position + 1} из {len(segments)}",
            )

    # Fragments of the same speaker separated by short VAD pauses (a lone
    # «— Да.», a split utterance) read as separate turns — join them back.
    before_merge = len(segments)
    segments = _merge_same_speaker_segments(segments)
    if len(segments) < before_merge:
        log.info(
            "speaker merge: %d -> %d segments", before_merge, len(segments)
        )
    return segments


# A turn shorter than this is not split off: too short pieces are useless and
# usually mean a token-time jitter right at the boundary.
MIN_TURN_S = 0.6

# Pyannote region edges jitter by up to a couple of frames (~200 ms), so a
# word whose midpoint lands in a gap between regions snaps to the nearest
# region within this distance instead of losing its speaker.
TURN_TOLERANCE_S = 0.2

# Consecutive segments of the same speaker separated by a pause shorter than
# this are merged: Silero VAD cuts at ~300 ms pauses, so a short «— Да.»
# interjection or a pause-broken utterance would otherwise stay a fragment.
MERGE_GAP_S = 0.5
# Never merge past the recognition stage's segment cap.
MAX_MERGED_SEGMENT_S = 30.0


def _merge_same_speaker_segments(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge consecutive segments of the same speaker separated by a short pause.

    Silero VAD cuts utterances at short pauses (~300 ms) and pads the edges,
    so a speaker's continuous speech arrives as fragments — a lone «— Да.»
    between two parts of one utterance, for instance.  Adjacent segments with
    the same speaker label and a gap below :data:`MERGE_GAP_S` are joined
    (texts are concatenated with a space) until the merged piece reaches
    :data:`MAX_MERGED_SEGMENT_S`.  Runs after :func:`_split_at_speaker_turns`,
    so token fields are already stripped from the pieces.
    """
    if len(segments) < 2:
        return segments

    merged: list[dict[str, Any]] = []
    for segment in segments:
        start = float(segment.get("start_s", 0.0))
        end = max(float(segment.get("end_s", start)), start)
        speaker = segment.get("speaker")
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and speaker
            and previous.get("speaker") == speaker
            and start - float(previous.get("end_s", start)) < MERGE_GAP_S
            and end - float(previous.get("start_s", start)) <= MAX_MERGED_SEGMENT_S
        ):
            previous["text"] = (
                f"{str(previous.get('text') or '').strip()} "
                f"{str(segment.get('text') or '').strip()}".strip()
            )
            previous["end_s"] = end
            continue
        merged.append(segment)
    return merged


# Same decoding rule as onnx-asr's ``DECODE_SPACE_PATTERN``: a token that is a
# lone space is dropped, spaces attached to a word boundary are kept as one.
_DECODE_SPACE_PATTERN = re.compile(r"\A\s|\s\B|(\s)\b")


def _decode_tokens(tokens: list[str]) -> str:
    return _DECODE_SPACE_PATTERN.sub(
        lambda match: " " if match.group(1) else "", "".join(tokens)
    ).strip()


def _token_midpoint(times: list[float], index: int, end_time: float) -> float:
    """Midpoint time of token *index* (tokens carry their start time)."""
    start = times[index]
    end = times[index + 1] if index + 1 < len(times) else end_time
    return (max(start, 0.0) + max(end, start)) / 2.0


def _token_starts_word(index: int, tokens: list[str]) -> bool:
    """True when token *index* begins a new word in the GigaAM stream.

    The vocabulary has no word-space tokens: a space is either a standalone
    ``" "`` token or a leading-space token like ``" как"``.  Letters and
    punctuation continue the current word.
    """
    token = tokens[index]
    if token == " ":
        # A standalone space opens the next word (if there is one).
        return index + 1 < len(tokens) and tokens[index + 1] != " "
    return token.startswith(" ")


def _split_at_speaker_turns(
    segments: list[dict[str, Any]],
    regions: list[tuple[str, float, float]],
    *,
    vote_delay_s: float = 0.0,
    boundary_margin_s: float = TURN_TOLERANCE_S,
) -> list[dict[str, Any]]:
    """Cut segments that span several speakers into per-speaker pieces.

    A segment is only split when it carries token timestamps (from the
    recognition stage).  Tokens are grouped into words, each word votes with
    the speaker of its middle tokens, single-word flips are smoothed away and
    micro-runs under :data:`MIN_TURN_S` fold into the previous piece — so one
    token straddling the boundary can neither tear a word apart nor spawn a
    useless fragment.  Every remaining change of speaker starts a new piece
    with its own decoded text.  Segments without tokens are unchanged.

    *vote_delay_s* — компенсация задержки CTC-пика GigaAM: точка голосования
    слова сдвигается вперёд на это значение (пик регистрируется позже фонемы).
    *boundary_margin_s* — токен, попавший в зазор между регионами спикеров,
    привязывается к ближайшему краю, если тот не дальше допуска.
    """
    if not regions:
        return segments

    result: list[dict[str, Any]] = []
    for segment in segments:
        times = segment.get("token_times")
        tokens = segment.get("tokens")
        if not times or not tokens or len(times) != len(tokens):
            result.append(segment)
            continue

        times = list(times)
        tokens = list(tokens)
        start = float(segment.get("start_s", 0.0))
        end = max(float(segment.get("end_s", start)), start)

        def label_at(moment: float, default: str) -> str:
            # Point-in-region check: a moment inside a region belongs to that
            # speaker; tokens in a gap between regions snap to the nearest
            # region edge within a small tolerance — pyannote boundaries
            # jitter, and words right at a speaker turn must keep their
            # speaker instead of being folded into the previous one.
            for label, region_start, region_end in regions:
                if region_start <= moment < region_end:
                    return label
            best_label = default
            best_distance = boundary_margin_s
            for label, region_start, region_end in regions:
                distance = min(abs(moment - region_start), abs(moment - region_end))
                if distance < best_distance:
                    best_label = label
                    best_distance = distance
            return best_label

        # One vote per token ("" = no region covers it), grouped into words.
        # The vote point is the token midpoint shifted forward by the CTC
        # peak delay (the acoustic model registers a phoneme after it is
        # spoken), clamped so it never leaves the segment.
        labels = [
            label_at(
                min(_token_midpoint(times, index, end) + vote_delay_s, end),
                "",
            )
            for index in range(len(tokens))
        ]
        words: list[list[int]] = []
        for index in range(len(tokens)):
            if index == 0 or _token_starts_word(index, tokens):
                words.append([index])
            else:
                words[-1].append(index)

        word_labels: list[str] = []
        for word in words:
            counted = Counter(
                label for label in (labels[index] for index in word) if label
            )
            best = counted.most_common(1)
            word_labels.append(best[0][0] if best else "")

        if not any(word_labels):
            # No region covers this segment at all.
            result.append(segment)
            continue

        # Words outside every region keep the surrounding speaker.
        first_label = next(label for label in word_labels if label)
        previous = ""
        for position, label in enumerate(word_labels):
            if not label:
                word_labels[position] = previous or first_label
            else:
                previous = label

        # Smooth single-word flips (label jitter right at the boundary).
        for position, label in enumerate(word_labels):
            if (
                0 < position < len(word_labels) - 1
                and word_labels[position - 1] == word_labels[position + 1]
                and word_labels[position - 1] != label
            ):
                word_labels[position] = word_labels[position - 1]

        # Runs of consecutive words with the same speaker; short runs fold
        # into the previous piece and repeated labels merge back together.
        runs: list[list[int]] = []
        for position, label in enumerate(word_labels):
            if runs and word_labels[runs[-1][0]] == label:
                runs[-1].append(position)
            else:
                runs.append([position])
        stable_runs: list[list[int]] = []
        for position, run in enumerate(runs):
            run_start = times[words[run[0]][0]]
            run_end = (
                times[words[runs[position + 1][0]][0]]
                if position + 1 < len(runs)
                else end
            )
            short = run_end - run_start < MIN_TURN_S
            same = (
                bool(stable_runs)
                and word_labels[stable_runs[-1][0]] == word_labels[run[0]]
            )
            if stable_runs and (short or same):
                stable_runs[-1].extend(run)
            else:
                stable_runs.append(list(run))

        pieces: list[dict[str, Any]] = []
        piece_start = start
        for run_position, run in enumerate(stable_runs):
            is_last = run_position + 1 == len(stable_runs)
            if is_last:
                piece_end = end
            else:
                next_first = words[stable_runs[run_position + 1][0]][0]
                piece_end = max(_token_midpoint(times, next_first, end), piece_start)
            piece_tokens = [
                tokens[index] for word_position in run for index in words[word_position]
            ]
            pieces.append(
                {
                    **segment,
                    "start_s": piece_start,
                    "end_s": piece_end,
                    "speaker": word_labels[run[0]],
                    "text": _decode_tokens(piece_tokens),
                }
            )
            if not is_last:
                piece_start = times[next_first]

        if len(pieces) == 1:
            # One speaker in this segment: keep the original bounds and text.
            pieces[0]["start_s"] = start
            pieces[0]["end_s"] = end
            pieces[0]["text"] = (
                str(segment.get("text") or "").strip() or pieces[0]["text"]
            )
        else:
            for piece in pieces:
                piece["text"] = piece["text"].strip() or "…"
        for piece in pieces:
            # Token data has done its job and must not leak further.
            piece.pop("token_times", None)
            piece.pop("tokens", None)
        result.extend(pieces)
    return result


def _diarize_with_embeddings(
    segments: list[dict[str, Any]],
    audio: np.ndarray,
    sample_rate: int,
    *,
    cpu_threads: int,
    threshold: float,
    model_path: str | Path | None,
    repo_id: str | None,
    group: WeSpeakerGroup | None = None,
    cancel: Callable[[], None] | None,
    progress: Callable[[float, str], None] | None,
) -> list[dict[str, Any]]:
    """Label segments with WeSpeaker embeddings + cosine clustering.

    *group* — активная группа гиперпараметров: косинусный порог, способ
    сравнения с кластером (``centroid`` — с усреднённым центроидом,
    ``average`` — со средним попарного сходства с членами) и минимальный
    размер кластера (мелкие кластеры приклеиваются к ближайшему крупному).
    """
    if not segments:
        return segments
    effective_threshold = group.threshold if group else threshold
    method = group.method if group else "centroid"
    min_cluster_size = group.min_cluster_size if group else 1

    clips: list[np.ndarray] = []
    embedded_indices: list[int] = []
    for index, segment in enumerate(segments):
        if cancel is not None and index % 32 == 0:
            cancel()
        start = round(float(segment.get("start_s", 0.0)) * sample_rate)
        end = round(float(segment.get("end_s", segment.get("start_s", 0.0))) * sample_rate)
        end = max(end, start + 1)
        clip = audio[start:end]
        if clip.size == 0:
            segment["speaker"] = "Спикер 1"
            continue
        clips.append(clip)
        embedded_indices.append(index)

    if not clips:
        return segments

    if progress is not None:
        progress(0.0, "Загрузка модели спикеров")
    se = load_se(cpu_threads, model_path=model_path, repo_id=repo_id)
    try:
        embeddings = _embed(se, clips, cancel=cancel, progress=progress)
    except DiarizationCancelled:
        raise
    except Exception as exc:  # onnx runtime/preprocessor failure
        raise DiarizationError(f"Не удалось вычислить эмбеддинги спикеров: {exc}") from exc

    # Unit vectors, one per embedded segment.
    unit_embeddings = [_normalize(embedding.reshape(-1)) for embedding in embeddings]

    # Cluster state: label -> (running centroid, member embeddings/counts).
    cluster_labels: list[str] = []
    cluster_centroids: list[np.ndarray] = []
    cluster_members: list[list[np.ndarray]] = []
    assignment: dict[int, str] = {}

    total_segments = max(len(embedded_indices), 1)
    for position, (segment_index, emb) in enumerate(
        zip(embedded_indices, unit_embeddings, strict=True)
    ):
        if cancel is not None and position % 32 == 0:
            cancel()
        best_idx: int | None = None
        best_sim = -1.0
        for i, centroid in enumerate(cluster_centroids):
            if method == "average" and len(cluster_members[i]) > 0:
                sim = float(
                    np.mean(
                        [ _cosine_similarity(emb, member) for member in cluster_members[i] ]
                    )
                )
            else:
                sim = _cosine_similarity(emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx is not None and best_sim >= effective_threshold:
            label = cluster_labels[best_idx]
            # Update centroid with running average of unit vectors.
            count = len(cluster_members[best_idx])
            cluster_centroids[best_idx] = _normalize(
                (cluster_centroids[best_idx] * count + emb) / (count + 1)
            )
            cluster_members[best_idx].append(emb)
        else:
            label = f"Спикер {len(cluster_labels) + 1}"
            cluster_labels.append(label)
            cluster_centroids.append(emb.copy())
            cluster_members.append([emb])

        assignment[segment_index] = label
        segments[segment_index]["speaker"] = label
        if progress is not None and (position % 32 == 0 or position + 1 == total_segments):
            progress(
                0.9 + 0.1 * ((position + 1) / total_segments),
                f"Группировка спикеров: {position + 1} из {len(embedded_indices)}",
            )

    # Attach clusters that stayed under min_cluster_size to the nearest
    # surviving cluster (singletons and glitches must not become speakers).
    if min_cluster_size > 1:
        sizes = {label: len(cluster_members[index]) for index, label in enumerate(cluster_labels)}
        large = [
            (cluster_labels[index], cluster_centroids[index])
            for index, label in enumerate(cluster_labels)
            if sizes[label] >= min_cluster_size
        ]
        small = [label for label, size in sizes.items() if size < min_cluster_size]
        if small and large:
            for segment_index, label in list(assignment.items()):
                if label in small:
                    emb = unit_embeddings[embedded_indices.index(segment_index)]
                    best_label, best_centroid = large[0]
                    best_sim = _cosine_similarity(emb, best_centroid)
                    for target_label, centroid in large[1:]:
                        sim = _cosine_similarity(emb, centroid)
                        if sim > best_sim:
                            best_sim, best_label = sim, target_label
                    assignment[segment_index] = best_label
                    segments[segment_index]["speaker"] = best_label
            _renumber_speakers(segments)

    return segments


def _renumber_speakers(segments: list[dict[str, Any]]) -> None:
    """Renumber labels in order of first temporal appearance, closing gaps.

    После приклеивания мелких кластеров нумерация может получить дыры
    («Спикер 1», «Спикер 3»); перенумерация делает её плотной и по-прежнему
    отсчитывает спикеров от первого голоса, который появился в записи.
    """
    first_appearance: dict[str, None] = {}
    for segment in segments:
        label = segment.get("speaker")
        if label and label not in first_appearance:
            first_appearance[label] = None
    mapping = {
        old: f"Спикер {number}"
        for number, old in enumerate(first_appearance, start=1)
    }
    for segment in segments:
        old = segment.get("speaker")
        if old in mapping:
            segment["speaker"] = mapping[old]


# ============================================================================
# Движки и реестр: Strategy/Registry поверх реализаций выше.
# ============================================================================


class BaseDiarizationEngine(ABC):
    """Abstract diarization engine: turns VAD segments into speaker labels.

    Конкретные движки (:class:`PyannoteSegmentationEngine`,
    :class:`WeSpeakerResNetEngine`) оборачивают свои реализации и подставляют
    активную группу гиперпараметров из :class:`DiarizationProfiles`.
    """

    #: Текстовый ID движка; его подстроки ищутся в ключе модели (repo_id или
    #: путь к локальной ONNX-папке).
    id: str = ""
    label: str = ""

    def __init__(
        self,
        profiles: DiarizationProfiles | None = None,
        profile_id: str | None = None,
    ) -> None:
        self._profiles = profiles
        self._profile_id = profile_id
        self._profile_cache: Any = None

    @abstractmethod
    def run_diarization(
        self,
        segments: list[dict[str, Any]],
        audio: np.ndarray,
        sample_rate: int,
        *,
        cpu_threads: int,
        model_path: str | Path | None,
        repo_id: str | None,
        threshold: float,
        cancel: Callable[[], None] | None,
        progress: Callable[[float, str], None] | None,
    ) -> list[dict[str, Any]]:
        """Lable *segments* in-place and return them."""

    def _log_missing_profile(self, exc: KeyError) -> None:
        log.warning(
            "diarization profile %r not found; falling back to the first one",
            exc.args[0] if exc.args else self._profile_id,
        )


class PyannoteSegmentationEngine(BaseDiarizationEngine):
    """Энд-тю-энд сегментация PyAnnote (PowerSet-бинаризация, до 3 спикеров)."""

    id = "pyannote-segmentation-3.0"
    label = "PyAnnote Segmentation 3.0"

    def _resolve_group(self) -> PyannoteGroup | None:
        if not self._profiles:
            return None
        if self._profile_cache is None:
            try:
                self._profile_cache: PyannoteGroup | None = self._profiles.pyannote_group(
                    self._profile_id
                )
            except KeyError as exc:
                self._log_missing_profile(exc)
                self._profile_cache = self._profiles.pyannote[0]
        return self._profile_cache

    def run_diarization(
        self,
        segments: list[dict[str, Any]],
        audio: np.ndarray,
        sample_rate: int,
        *,
        cpu_threads: int,
        model_path: str | Path | None,
        repo_id: str | None,
        threshold: float,
        cancel: Callable[[], None] | None,
        progress: Callable[[float, str], None] | None,
    ) -> list[dict[str, Any]]:
        return _diarize_with_pyannote(
            segments,
            audio,
            sample_rate,
            cpu_threads=cpu_threads,
            model_path=model_path,
            repo_id=repo_id or PYANNOTE_REPO,
            group=self._resolve_group(),
            link=self._profiles.gigaam if self._profiles else None,
            cancel=cancel,
            progress=progress,
        )


class WeSpeakerResNetEngine(BaseDiarizationEngine):
    """Эмбеддинги WeSpeaker ResNet34 + косинусная кластеризация."""

    id = "wespeaker"
    label = "WeSpeaker ResNet34"

    def _resolve_group(self) -> WeSpeakerGroup | None:
        if not self._profiles:
            return None
        if self._profile_cache is None:
            try:
                self._profile_cache: WeSpeakerGroup | None = self._profiles.wespeaker_group(
                    self._profile_id
                )
            except KeyError as exc:
                self._log_missing_profile(exc)
                self._profile_cache = self._profiles.wespeaker[0]
        return self._profile_cache

    def run_diarization(
        self,
        segments: list[dict[str, Any]],
        audio: np.ndarray,
        sample_rate: int,
        *,
        cpu_threads: int,
        model_path: str | Path | None,
        repo_id: str | None,
        threshold: float,
        cancel: Callable[[], None] | None,
        progress: Callable[[float, str], None] | None,
    ) -> list[dict[str, Any]]:
        return _diarize_with_embeddings(
            segments,
            audio,
            sample_rate,
            cpu_threads=cpu_threads,
            threshold=threshold,
            model_path=model_path,
            repo_id=repo_id,
            group=self._resolve_group(),
            cancel=cancel,
            progress=progress,
        )


class DiarizationRegistry:
    """Реестр движков: выбор по текстовому ID или ключу модели.

    Подстрока ID движка ищется в ключе модели (repo_id с манифеста либо путь
    к локальной папке ONNX).  Ни один ключ не подошёл -> WeSpeaker (прежнее
    поведение для «custom»-моделей без подсказок в пути).
    """

    def __init__(self) -> None:
        self._engine_classes: dict[str, type[BaseDiarizationEngine]] = {}

    def register(self, engine_class: type[BaseDiarizationEngine]) -> None:
        self._engine_classes[engine_class.id] = engine_class

    def engine_class_for(self, model_key: str) -> type[BaseDiarizationEngine]:
        key = (model_key or "").lower()
        for engine_id, engine_class in self._engine_classes.items():
            if engine_id in key:
                return engine_class
        if "pyannote" in key:
            return self._engine_classes[PyannoteSegmentationEngine.id]
        return self._engine_classes[WeSpeakerResNetEngine.id]

    def known_engines(self) -> list[tuple[str, str]]:
        return [(engine.id, engine.label) for engine in self._engine_classes.values()]


REGISTRY = DiarizationRegistry()
REGISTRY.register(PyannoteSegmentationEngine)
REGISTRY.register(WeSpeakerResNetEngine)
