"""Lazy GigaAM/onnx-asr service used by the background job worker."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from airec.asr.audio import AudioError, prepare_wav, read_pcm16_mono
from airec.asr.diarization import DiarizationError, diarize_segments
from airec.asr.vad import (
    DEFAULT_MAX_SEGMENT_S,
    DEFAULT_MIN_SILENCE_MS,
    DEFAULT_SPEECH_PAD_MS,
    SAMPLE_RATE,
    SHORT_AUDIO_SECONDS,
    SpeechSegment,
    VadError,
    join_segments,
    segment_and_recognize,
)
from airec.models.manifest import find_model, load_manifest, manifest_dir

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gigaam-v3-e2e-ctc"
QUANTIZATION = "int8"

# Used only when Silero VAD cannot be loaded.  60 s is well below GigaAM's
# fixed positional limit (~200 s), so a long file still gets transcribed.
FALLBACK_WINDOW_SECONDS = 60.0

ProgressCallback = Callable[[float, str], None]


class AsrError(RuntimeError):
    """Model loading or recognition failed."""


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    model_id: str
    duration_s: float | None
    elapsed_s: float
    rtf: float | None
    segments: tuple[SpeechSegment, ...] = ()


_model: Any | None = None
_model_key: tuple[str, int] | None = None
_model_lock = threading.Lock()


def _load_model(model_id: str, cpu_threads: int) -> Any:
    global _model, _model_key

    key = (model_id, cpu_threads)
    with _model_lock:
        if _model is not None and _model_key == key:
            return _model

        try:
            import onnx_asr
            import onnxruntime as ort
        except ImportError as exc:
            raise AsrError(
                "Компоненты распознавания не установлены. "
                "Установите extras «asr» или пересоберите приложение."
            ) from exc

        available = ort.get_available_providers()
        if "CPUExecutionProvider" not in available:
            raise AsrError(
                "ONNX Runtime не предоставляет CPUExecutionProvider: "
                f"{', '.join(available) or 'нет провайдеров'}."
            )

        session_options = None
        if cpu_threads > 0:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = cpu_threads
            session_options.inter_op_num_threads = 1

        log.info("loading ASR model %s (int8, CPU, threads=%s)", model_id, cpu_threads or "auto")
        try:
            model = onnx_asr.load_model(
                model_id,
                quantization=QUANTIZATION,
                providers=["CPUExecutionProvider"],
                sess_options=session_options,
            )
        except Exception as exc:  # onnx-asr raises several unrelated types
            raise AsrError(f"Не удалось загрузить модель {model_id}: {exc}") from exc

        _model = model
        _model_key = key
        log.info("ASR model loaded: %s", model_id)
        return model


def _resolve_diarization(
    model_id: str,
    model_path: str,
    models_dir: Path | None,
) -> tuple[str | None, Path | None]:
    """Resolve the selected diarization model to (HF repo, local dir).

    Precedence: explicit local path (``custom``) → model downloaded through the
    catalogue → Hugging Face repository from the manifest.  Returns
    ``(None, None)`` when nothing is known, which makes diarization use the
    built-in default.
    """
    use_custom = model_id == "custom" or (not model_id and bool(model_path))
    if use_custom and model_path:
        candidate = Path(model_path)
        if candidate.is_dir():
            return None, candidate
        log.warning("diarization model path does not exist: %s", candidate)

    repo_id: str | None = None
    if model_id and model_id != "custom":
        entry = find_model(load_manifest(manifest_dir()), model_id)
        if entry is not None:
            repo_id = str(entry.get("repo") or "") or None

    if models_dir is not None and model_id and model_id != "custom":
        candidate = Path(models_dir) / model_id
        if candidate.is_dir() and any(candidate.glob("*.onnx")):
            return repo_id, candidate

    return repo_id, None


def _recognize_fixed_windows(
    model: Any,
    wav_path: Path,
    notify: ProgressCallback,
    *,
    window_s: float = FALLBACK_WINDOW_SECONDS,
) -> tuple[str, tuple[SpeechSegment, ...]]:
    """Fallback for a missing VAD model: cut the WAV into fixed windows."""
    audio = read_pcm16_mono(wav_path, sample_rate=SAMPLE_RATE)
    total = len(audio) / SAMPLE_RATE
    if total <= 0:
        return "", ()

    window = max(float(window_s), 5.0)
    texts: list[str] = []
    segments: list[SpeechSegment] = []
    offset = 0.0
    index = 0

    while offset < total:
        end = min(offset + window, total)
        index += 1
        notify(
            0.3 + 0.6 * (end / total),
            f"Распознавание сегментов: {index}",
        )
        chunk = audio[int(offset * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        text = model.recognize(chunk, sample_rate=SAMPLE_RATE)
        cleaned = text.strip() if isinstance(text, str) else ""
        if cleaned:
            texts.append(cleaned)
            segments.append(SpeechSegment(offset, end, cleaned))
        offset = end

    return " ".join(texts).strip(), tuple(segments)


def transcribe(
    source: Path,
    *,
    model_id: str = DEFAULT_MODEL,
    cpu_threads: int = 0,
    ffmpeg: str | None = None,
    work_dir: Path,
    progress: ProgressCallback | None = None,
    vad_enabled: bool = True,
    vad_max_segment_s: float = DEFAULT_MAX_SEGMENT_S,
    vad_min_silence_ms: float = DEFAULT_MIN_SILENCE_MS,
    vad_speech_pad_ms: float = DEFAULT_SPEECH_PAD_MS,
    diarize_enabled: bool = False,
    diarize_threshold: float = 0.70,
    diarize_model: str = "",
    diarize_model_path: str = "",
    models_dir: Path | None = None,
) -> TranscriptionResult:
    """Convert *source* if needed and recognise it on the CPU.

    Files longer than :data:`SHORT_AUDIO_SECONDS` are split by Silero VAD into
    speech segments, recognised one by one and glued back into one transcript.
    If VAD is disabled or unavailable, a fixed-window fallback keeps long files
    working without hitting GigaAM's positional limit.

    When *diarize_enabled* is true, recognised segments are clustered by
    speaker (Wespeaker embeddings) so each segment gets a ``speaker`` label.

    The caller is responsible for serialising calls: the model and ONNX Runtime
    session are shared, and one worker avoids memory spikes and races.
    """
    notify = progress or (lambda _value, _phase: None)
    started = time.perf_counter()

    try:
        notify(0.08, "Подготовка аудио")
        work_dir.mkdir(parents=True, exist_ok=True)
        wav_path, duration = prepare_wav(
            source,
            work_dir / "normalized-16k-mono.wav",
            ffmpeg=ffmpeg,
        )

        notify(0.2, "Загрузка модели")
        model = _load_model(model_id, cpu_threads)

        segments: tuple[SpeechSegment, ...] = ()
        used_vad = False

        if vad_enabled and (duration is None or duration > SHORT_AUDIO_SECONDS):
            notify(0.28, "Нарезка по сегментам")
            try:
                collected: list[SpeechSegment] = []
                for segment in segment_and_recognize(
                    model,
                    wav_path,
                    cpu_threads=cpu_threads,
                    max_segment_s=vad_max_segment_s,
                    min_silence_ms=vad_min_silence_ms,
                    speech_pad_ms=vad_speech_pad_ms,
                ):
                    collected.append(segment)
                    if duration and duration > 0:
                        value = 0.32 + 0.55 * min(segment.end_s / duration, 1.0)
                    else:
                        value = 0.5
                    notify(value, f"Распознавание сегментов: {len(collected)}")
                segments = tuple(collected)
                used_vad = True
                log.info("VAD segmentation: %d segments for %s", len(segments), source.name)
            except VadError as exc:
                log.warning("VAD unavailable, using fixed windows: %s", exc)
            except Exception as exc:  # fall back instead of failing the whole job
                log.warning("VAD segmentation failed, using fixed windows: %s", exc)

        if used_vad:
            text = join_segments(segments)
        elif duration is not None and duration > FALLBACK_WINDOW_SECONDS:
            notify(0.3, "Распознавание по окнам")
            text, segments = _recognize_fixed_windows(model, wav_path, notify)
        else:
            notify(0.4, "Распознавание")
            try:
                text = model.recognize(str(wav_path), channel="mean")
            except AudioError:
                raise
            except Exception as exc:
                raise AsrError(f"Ошибка распознавания: {exc}") from exc

            if not isinstance(text, str):
                raise AsrError("Модель вернула неожиданный результат.")
            text = text.strip()
            if text:
                segments = (SpeechSegment(0.0, duration or 0.0, text),)

        # Диаризация: кластеризация сегментов по спикерам (Wespeaker embeddings).
        if diarize_enabled and segments:
            notify(0.88, "Разбор по участникам")
            try:
                audio = read_pcm16_mono(wav_path, sample_rate=SAMPLE_RATE)
                segment_dicts = [s.as_dict() for s in segments]
                repo_id, model_path = _resolve_diarization(
                    diarize_model, diarize_model_path, models_dir
                )
                diarized = diarize_segments(
                    segment_dicts,
                    audio,
                    SAMPLE_RATE,
                    cpu_threads=cpu_threads,
                    threshold=diarize_threshold,
                    model_path=model_path,
                    repo_id=repo_id,
                )
                segments = tuple(
                    SpeechSegment(
                        s["start_s"], s["end_s"], s["text"], speaker=s.get("speaker")
                    )
                    for s in diarized
                )
                log.info(
                    "Diarization: %d segments, %d speakers",
                    len(segments),
                    len({s.speaker for s in segments if s.speaker}),
                )
            except DiarizationError as exc:
                log.warning("Diarization unavailable: %s", exc)

        elapsed = max(time.perf_counter() - started, 0.001)
        rtf = duration / elapsed if duration and duration > 0 else None
        notify(0.92, "Сохранение результата")
        return TranscriptionResult(
            text=text.strip(),
            model_id=model_id,
            duration_s=duration,
            elapsed_s=elapsed,
            rtf=rtf,
            segments=segments,
        )
    except AudioError as exc:
        raise AsrError(str(exc)) from exc
