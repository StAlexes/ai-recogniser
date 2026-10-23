"""Lazy GigaAM/onnx-asr service used by the background job worker."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from airec.asr.audio import AudioError, prepare_wav

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gigaam-v3-e2e-ctc"
QUANTIZATION = "int8"

# GigaAM ONNX has fixed positional embeddings (~5000 frames).  Audio longer
# than this limit fails with an ONNX broadcast/shape error instead of a
# readable message.  Until VAD/chunking is implemented (M1 remainder), reject
# overly long input up front with a human-friendly error.
MAX_AUDIO_SECONDS = 300

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


def transcribe(
    source: Path,
    *,
    model_id: str = DEFAULT_MODEL,
    cpu_threads: int = 0,
    ffmpeg: str | None = None,
    work_dir: Path,
    progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    """Convert *source* if needed and recognise it on the CPU.

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

        if duration and duration > MAX_AUDIO_SECONDS:
            raise AsrError(
                f"Аудио слишком длинное для текущей модели "
                f"({duration:.0f} с, лимит {MAX_AUDIO_SECONDS} с). "
                "Разбейте файл на части покороче или подождите нарезки по "
                "сегментам (VAD) в следующем релизе."
            )

        notify(0.2, "Загрузка модели")
        model = _load_model(model_id, cpu_threads)

        notify(0.4, "Распознавание")
        try:
            text = model.recognize(str(wav_path), channel="mean")
        except AudioError:
            raise
        except Exception as exc:
            raise AsrError(
                f"Ошибка распознавания: {exc}. "
                "Если файл длинный, попробуйте разобрать его на части."
            ) from exc

        if not isinstance(text, str):
            raise AsrError("Модель вернула неожиданный результат.")

        elapsed = max(time.perf_counter() - started, 0.001)
        rtf = duration / elapsed if duration and duration > 0 else None
        notify(0.92, "Сохранение результата")
        return TranscriptionResult(
            text=text.strip(),
            model_id=model_id,
            duration_s=duration,
            elapsed_s=elapsed,
            rtf=rtf,
        )
    except AudioError as exc:
        raise AsrError(str(exc)) from exc
