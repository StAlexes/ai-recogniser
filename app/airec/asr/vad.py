"""Silero VAD segmentation for long recordings.

Long files cannot be passed to GigaAM in one piece: the exported ONNX graph has
a fixed positional limit (~5000 encoder frames, roughly 200 s of audio).  This
module uses the Silero VAD v5 ONNX model shipped through ``onnx-asr`` to find
speech regions, splits them at natural pauses, caps each segment and feeds the
segments to the ASR model one by one.

The VAD model is downloaded from Hugging Face on first use (about 2 MB) and is
cached afterwards, exactly like the GigaAM weights.  If it cannot be loaded the
caller falls back to fixed-length windows, so long files still work offline.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

#: Files at or below this duration are recognised in one pass, without VAD.
SHORT_AUDIO_SECONDS = 30.0

DEFAULT_MAX_SEGMENT_S = 30.0
DEFAULT_MIN_SILENCE_MS = 300.0
DEFAULT_SPEECH_PAD_MS = 100.0
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 4

_VAD_MODEL = "silero"

_vad: Any | None = None
_vad_key: int | None = None
_vad_lock = threading.Lock()


class VadError(RuntimeError):
    """The Silero VAD model could not be loaded."""


@dataclass(frozen=True)
class SpeechSegment:
    """One recognised speech region with its position in the source audio."""

    start_s: float
    """Segment start time."""
    end_s: float
    """Segment end time."""
    text: str
    """Segment recognized text."""
    speaker: str | None = None
    """Optional speaker label (e.g. «Спикер 1») after diarization."""
    token_times: tuple[float, ...] | None = None
    """Absolute token start times, seconds (recognition only, not persisted)."""
    tokens: tuple[str, ...] | None = None
    """Token texts aligned with :attr:`token_times` (recognition only)."""

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "text": self.text,
            "speaker": self.speaker,
        }


def load_vad(cpu_threads: int = 0) -> Any:
    """Load and cache the Silero VAD ONNX session.

    The model is small and shares the same ONNX Runtime as ASR, so it is kept
    for the lifetime of the process.  ``cpu_threads=0`` lets ONNX Runtime pick
    the thread count.
    """
    global _vad, _vad_key

    key = cpu_threads
    with _vad_lock:
        if _vad is not None and _vad_key == key:
            return _vad

        try:
            import onnx_asr
            import onnxruntime as ort
        except ImportError as exc:
            raise VadError(
                "Компоненты VAD не установлены. Установите extras «asr» "
                "или пересоберите приложение."
            ) from exc

        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise VadError("ONNX Runtime не предоставляет CPUExecutionProvider для VAD.")

        session_options = None
        if cpu_threads > 0:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = cpu_threads
            session_options.inter_op_num_threads = 1

        log.info("loading Silero VAD (CPU, threads=%s)", cpu_threads or "auto")
        try:
            vad = onnx_asr.load_vad(
                _VAD_MODEL,
                providers=["CPUExecutionProvider"],
                sess_options=session_options,
            )
        except Exception as exc:  # download/read/init errors from onnx-asr
            raise VadError(f"Не удалось загрузить Silero VAD: {exc}") from exc

        _vad = vad
        _vad_key = key
        log.info("Silero VAD loaded")
        return vad


def segment_and_recognize(
    model: Any,
    wav_path: Path,
    *,
    cpu_threads: int = 0,
    max_segment_s: float = DEFAULT_MAX_SEGMENT_S,
    min_silence_ms: float = DEFAULT_MIN_SILENCE_MS,
    speech_pad_ms: float = DEFAULT_SPEECH_PAD_MS,
    threshold: float = DEFAULT_THRESHOLD,
) -> Iterator[SpeechSegment]:
    """Yield recognised speech regions of *wav_path* with token timestamps.

    ``onnx-asr`` reads the WAV, runs Silero VAD and recognises each segment in
    small batches.  Segments are capped at ``max_segment_s`` and split at pauses
    of at least ``min_silence_ms``.  Empty segments are skipped.

    Token timestamps are relative to the segment start (onnx-asr behaviour),
    so they are shifted to absolute time here: diarization uses them to cut a
    VAD segment at speaker turns.
    """
    vad = load_vad(cpu_threads)
    adapter = model.with_vad(
        vad,
        threshold=threshold,
        min_speech_duration_ms=250,
        max_speech_duration_s=max_segment_s,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        batch_size=DEFAULT_BATCH_SIZE,
    ).with_timestamps()

    for result in adapter.recognize(str(wav_path), channel="mean"):
        text = (getattr(result, "text", "") or "").strip()
        if not text:
            continue
        start = float(result.start)
        timestamps = getattr(result, "timestamps", None)
        tokens = getattr(result, "tokens", None)
        # onnx-asr returns token times relative to the segment start.
        abs_times = (
            [start + float(value) for value in timestamps]
            if timestamps and tokens and len(timestamps) == len(tokens)
            else None
        )
        if abs_times is None or not abs_times:
            yield SpeechSegment(start_s=start, end_s=float(result.end), text=text)
            continue
        yield SpeechSegment(
            start_s=start,
            end_s=float(result.end),
            text=text,
            token_times=tuple(abs_times),
            tokens=tuple(tokens),
        )


def join_segments(segments: Iterable[SpeechSegment]) -> str:
    """Glue segment texts into one transcript, keeping punctuation as-is."""
    parts = [segment.text.strip() for segment in segments if segment.text.strip()]
    return " ".join(parts).strip()
