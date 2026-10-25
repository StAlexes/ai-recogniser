"""Speaker diarization based on Wespeaker embeddings + simple clustering.

For each VAD segment, a speaker embedding is computed (Wespeaker ONNX, MIT).
Segments are grouped by cosine similarity: if similarity to an existing cluster
is above the threshold, the segment is assigned to that speaker, otherwise a
new cluster is created.

The model is downloaded from the Hugging Face cache on first use (~50 MB) and
cached next to GigaAM.  If the model is unavailable, diarization is disabled
with a warning — segments remain without a speaker label, but recognition
keeps working.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Segments with cosine similarity >= this value are assigned to the same speaker.
SIMILARITY_THRESHOLD = 0.70

_se: Any | None = None
_se_lock = threading.Lock()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class DiarizationError(RuntimeError):
    """Диаризация недоступна (модель эмбеддингов не загружена)."""


def load_se(cpu_threads: int = 0, model_path: str | None = None) -> Any:
    """Load and cache the Wespeaker speaker-embedding ONNX session.

    If *model_path* is given, the model is loaded from that directory
    (offline).  Otherwise ``onnx-asr`` tries to resolve/download it from
    Hugging Face.
    """
    global _se

    with _se_lock:
        if _se is not None:
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

        log.info("loading Wespeaker speaker-embedding model (CPU, threads=%s)", cpu_threads or "auto")
        try:
            manager = Manager(sess_options=session_options, providers=["CPUExecutionProvider"])
            if model_path:
                se = manager.create_se("wespeaker", local_dir=model_path)
            else:
                se = manager.create_se("wespeaker")
        except Exception as exc:  # model download/read/init errors
            raise DiarizationError(f"Не удалось загрузить модель диаризации: {exc}") from exc

        _se = se
        log.info("Wespeaker speaker-embedding model loaded")
        return _se


def diarize_segments(
    segments: list[dict[str, Any]],
    audio: np.ndarray,
    sample_rate: int,
    *,
    cpu_threads: int = 0,
    threshold: float = SIMILARITY_THRESHOLD,
    model_path: str | None = None,
) -> list[dict[str, Any]]:
    """Assign speaker labels to *segments* in-place and return them.

    *audio* — float32 PCM array at *sample_rate* (16 kHz).  Each segment is
    sliced, embedded with Wespeaker, and clustered by cosine similarity.
    """
    if not segments:
        return segments

    se = load_se(cpu_threads, model_path=model_path)

    # Centroids of discovered speaker clusters.
    centroid_labels: list[str] = []
    centroid_embs: list[np.ndarray] = []
    speaker_counts: dict[str, int] = {}

    for segment in segments:
        start = round(float(segment.get("start_s", 0.0)) * sample_rate)
        end = round(float(segment.get("end_s", segment.get("start_s", 0.0))) * sample_rate)
        end = max(end, start + 1)
        clip = audio[start:end]

        if clip.size == 0:
            label = "Спикер 1"
            segment["speaker"] = label
            continue

        waveform = clip.reshape(1, -1).astype(np.float32)
        lengths = np.array([waveform.shape[1]], dtype=np.int64)
        emb = se.embedding(waveform, lengths)  # (1, embedding_dim)
        emb = emb.reshape(-1)

        # Find best matching cluster.
        best_idx: int | None = None
        best_sim = -1.0
        for i, centroid in enumerate(centroid_embs):
            sim = _cosine_similarity(emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx is not None and best_sim >= threshold:
            label = centroid_labels[best_idx]
            # Update centroid with running average.
            count = speaker_counts[label]
            centroid_embs[best_idx] = (centroid_embs[best_idx] * count + emb) / (count + 1)
            speaker_counts[label] = count + 1
        else:
            label = f"Спикер {len(centroid_labels) + 1}"
            centroid_labels.append(label)
            centroid_embs.append(emb.copy())
            speaker_counts[label] = 1

        segment["speaker"] = label

    return segments
