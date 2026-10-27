"""Speaker diarization based on WeSpeaker embeddings + cosine clustering.

For each VAD segment, a speaker embedding is computed with a WeSpeaker ONNX
model (onnx-asr ``create_se``).  Segments are grouped by cosine similarity: if
the similarity to an existing cluster centroid is above the threshold, the
segment is assigned to that speaker, otherwise a new cluster is created.

The model can be chosen in settings (manifest entries with ``kind:
"diarization"``).  It is downloaded from Hugging Face on first use and cached
there; a model downloaded through the model catalogue is loaded from
``<data>/models/<id>/`` instead, so it works offline.  If the model is
unavailable, diarization is disabled with a warning — segments remain without a
speaker label, but recognition keeps working.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Segments with cosine similarity >= this value are assigned to the same speaker.
SIMILARITY_THRESHOLD = 0.70

# Upstream onnx-asr test model; used when no model is selected (e.g. an old
# config without ``diarize_model``).
DEFAULT_SE_REPO = "Wespeaker/wespeaker-voxceleb-resnet34"

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


def _embed(se: Any, clips: list[np.ndarray]) -> np.ndarray:
    """Embed *clips* (1-D float32 arrays) and return an (N, dim) matrix."""
    usable: list[np.ndarray] = []
    for clip in clips:
        if clip.size < MIN_CLIP_SAMPLES:
            clip = np.pad(clip, (0, MIN_CLIP_SAMPLES - clip.size))
        usable.append(clip.astype(np.float32, copy=False))

    embeddings: list[np.ndarray] = []
    for offset in range(0, len(usable), EMBED_BATCH):
        batch = usable[offset : offset + EMBED_BATCH]
        # SeAdapter.embedding accepts a list of waveforms and returns (N, dim).
        result = np.asarray(se.embedding(batch), dtype=np.float32)
        if result.ndim == 1:
            result = result.reshape(1, -1)
        embeddings.append(result)
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
) -> list[dict[str, Any]]:
    """Assign speaker labels to *segments* in-place and return them.

    *audio* — float32 PCM array at *sample_rate* (16 kHz).  Each segment is
    sliced and embedded with WeSpeaker; embeddings are clustered by cosine
    similarity to the running cluster centroids.
    """
    if not segments:
        return segments

    clips: list[np.ndarray] = []
    embedded_indices: list[int] = []
    for index, segment in enumerate(segments):
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

    se = load_se(cpu_threads, model_path=model_path, repo_id=repo_id)
    try:
        embeddings = _embed(se, clips)
    except Exception as exc:  # onnx runtime/preprocessor failure
        raise DiarizationError(f"Не удалось вычислить эмбеддинги спикеров: {exc}") from exc

    # Centroids of discovered speaker clusters (unit vectors).
    centroid_labels: list[str] = []
    centroid_embs: list[np.ndarray] = []
    speaker_counts: dict[str, int] = {}

    for segment_index, embedding in zip(embedded_indices, embeddings, strict=True):
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

    return segments
