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
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

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


def _pyannote_speaker_regions(
    session: Any,
    audio: np.ndarray,
    sample_rate: int,
    *,
    onset: float = PYANNOTE_ONSET,
    min_duration_s: float = PYANNOTE_MIN_DURATION_S,
    cancel: Callable[[], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> list[tuple[str, float, float]]:
    """Run pyannote segmentation and return per-speaker speech regions.

    Implements the same windowing/decoding math as onnx-asr's ``PyAnnoteVad``
    (10 s window, 5 s overlap, per-speaker probability fusion), but keeps all
    speaker channels instead of collapsing them into one speech channel, and
    yields ``(speaker, start_s, end_s)`` regions.

    The ONNX graph outputs 7 frames per class: {no-speech}, {spk1}, {spk2},
    {spk3}, {spk1+spk2}, {spk1+spk3}, {spk2+spk3}; multi-speaker classes add
    their probability to each participating speaker.
    """
    window_size = 10 * sample_rate
    overlap = 5 * sample_rate

    total = len(audio)
    if total == 0:
        return []

    starts: list[int] = []
    offset = 0
    while offset + window_size <= total:
        starts.append(offset)
        offset += window_size - overlap
    if not starts or total - starts[-1] > 0:
        if total <= window_size:
            starts = [0]
        else:
            # Pad the tail window like onnx-asr does.
            starts.append(total - window_size)

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

    # Rename raw speaker channels to consecutive labels in order of first
    # appearance: the network may pick any of its three slots (e.g. classes
    # 2 and 3 for two voices), but users expect «Спикер 1», «Спикер 2», ...
    channel_labels: dict[int, str] = {}

    regions: list[tuple[str, float, float]] = []
    min_duration = int(min_duration_s * sample_rate)
    for spk in range(num_speakers):
        active = counts[spk] > 0
        averaged = np.divide(
            probs[spk], counts[spk], out=np.zeros_like(probs[spk]), where=active
        )
        mask = active & (averaged >= onset)
        if not mask.any():
            continue
        label = channel_labels.setdefault(spk, f"Спикер {len(channel_labels) + 1}")
        edges = np.diff(mask.astype(np.int8))
        region_starts = np.flatnonzero(edges == 1) + 1
        region_ends = np.flatnonzero(edges == -1) + 1
        if mask[0]:
            region_starts = np.concatenate(([0], region_starts))
        if mask[-1]:
            region_ends = np.concatenate((region_ends, [total]))
        for region_start, region_end in zip(region_starts, region_ends, strict=False):
            if region_end - region_start >= min_duration:
                regions.append(
                    (label, region_start / sample_rate, region_end / sample_rate)
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
    cancel: Callable[[], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Assign speaker labels to *segments* in-place and return them.

    *audio* — float32 PCM array at *sample_rate* (16 kHz).

    A *repo_id* pointing at the pyannote segmentation model (default in the
    manifest) selects the end-to-end engine: the network decides per frame
    which of up to three speakers is talking and segments get the label of the
    speaker they overlap most.  Any other *repo_id* (WeSpeaker models) selects
    the embedding engine: each segment is embedded and clustered by cosine
    similarity to the running cluster centroids.  *cancel* is polled between
    expensive steps so a stopped stage does not keep the worker busy.
    """
    if not segments:
        return segments

    engine_repo = repo_id or ""
    use_pyannote = "pyannote" in engine_repo.lower()

    if use_pyannote:
        return _diarize_with_pyannote(
            segments,
            audio,
            sample_rate,
            cpu_threads=cpu_threads,
            model_path=model_path,
            repo_id=engine_repo,
            cancel=cancel,
            progress=progress,
        )
    return _diarize_with_embeddings(
        segments,
        audio,
        sample_rate,
        cpu_threads=cpu_threads,
        threshold=threshold,
        model_path=model_path,
        repo_id=engine_repo or None,
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
    cancel: Callable[[], None] | None,
    progress: Callable[[float, str], None] | None,
) -> list[dict[str, Any]]:
    """Label segments with the pyannote segmentation engine (up to 3 speakers)."""
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

    # Longest overlap wins; ties prefer the earlier region.
    total = max(len(segments), 1)
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
        segment["speaker"] = best_label or regions[0][0]
        if progress is not None and (position % 32 == 0 or position + 1 == total):
            progress(
                0.55 + 0.45 * ((position + 1) / total),
                f"Разметка сегментов: {position + 1} из {len(segments)}",
            )
    return segments


def _diarize_with_embeddings(
    segments: list[dict[str, Any]],
    audio: np.ndarray,
    sample_rate: int,
    *,
    cpu_threads: int,
    threshold: float,
    model_path: str | Path | None,
    repo_id: str | None,
    cancel: Callable[[], None] | None,
    progress: Callable[[float, str], None] | None,
) -> list[dict[str, Any]]:
    """Label segments with WeSpeaker embeddings + cosine clustering."""
    if not segments:
        return segments

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

    # Centroids of discovered speaker clusters (unit vectors).
    centroid_labels: list[str] = []
    centroid_embs: list[np.ndarray] = []
    speaker_counts: dict[str, int] = {}

    total_segments = max(len(embedded_indices), 1)
    for position, (segment_index, embedding) in enumerate(
        zip(embedded_indices, embeddings, strict=True)
    ):
        if cancel is not None and position % 32 == 0:
            cancel()
        emb = _normalize(embedding.reshape(-1))

        best_idx: int | None = None
        best_sim = -1.0
        for i, centroid in enumerate(centroid_embs):
            sim = _cosine_similarity(emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx is not None and best_sim >= threshold:
            label = centroid_labels[best_idx]
            # Update centroid with running average of unit vectors.
            count = speaker_counts[label]
            centroid_embs[best_idx] = _normalize(
                (centroid_embs[best_idx] * count + emb) / (count + 1)
            )
            speaker_counts[label] = count + 1
        else:
            label = f"Спикер {len(centroid_labels) + 1}"
            centroid_labels.append(label)
            centroid_embs.append(emb.copy())
            speaker_counts[label] = 1

        segments[segment_index]["speaker"] = label
        if progress is not None and (position % 32 == 0 or position + 1 == total_segments):
            progress(
                0.9 + 0.1 * ((position + 1) / total_segments),
                f"Группировка спикеров: {position + 1} из {len(embedded_indices)}",
            )

    return segments
