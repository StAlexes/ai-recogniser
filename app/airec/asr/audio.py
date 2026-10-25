"""Audio/video input handling for the ASR worker.

All formats except WAV are decoded by ffmpeg into the canonical 16 kHz mono
PCM16 WAV that onnx-asr expects.  ffmpeg is optional at import time: a missing
binary becomes a clear per-job error instead of preventing the core from
starting.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".m4a",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
}

FFMPEG_TIMEOUT_S = 30 * 60
CREATE_NO_WINDOW = 0x08000000


class AudioError(RuntimeError):
    """The input cannot be converted into a readable WAV file."""


def find_ffmpeg(bin_dir: Path | None = None) -> str | None:
    """Locate ffmpeg in the app's bin directory, PATH, or imageio-ffmpeg."""
    candidates: list[Path] = []
    if bin_dir is not None:
        candidates.extend([bin_dir / "ffmpeg.exe", bin_dir / "ffmpeg"])
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path

    try:
        import imageio_ffmpeg

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if bundled.is_file():
            return str(bundled)
    except (ImportError, OSError, RuntimeError):
        log.warning("imageio-ffmpeg is not available; only PCM WAV input will work")
    return None


def _run_ffmpeg(command: list[str]) -> None:
    kwargs: dict[str, object] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "timeout": FFMPEG_TIMEOUT_S,
        "check": False,
    }
    if os.name == "nt":
        kwargs["creationflags"] = CREATE_NO_WINDOW
    try:
        completed = subprocess.run(command, **kwargs)  # type: ignore[arg-type]
    except subprocess.TimeoutExpired as exc:
        raise AudioError(
            "ffmpeg не успел обработать файл за 30 минут. "
            "Попробуйте файл короче или проверьте, что он не повреждён."
        ) from exc
    except OSError as exc:
        raise AudioError(f"Не удалось запустить ffmpeg: {exc}") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        tail = stderr[-600:] if stderr else "без подробностей"
        raise AudioError(f"ffmpeg не смог прочитать файл: {tail}")


def wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            if rate <= 0:
                return None
            return wav.getnframes() / float(rate)
    except (OSError, EOFError, wave.Error):
        return None


def prepare_wav(
    source: Path,
    destination: Path,
    *,
    ffmpeg: str | None,
) -> tuple[Path, float | None]:
    """Return a WAV path suitable for onnx-asr and its duration.

    For a WAV input without ffmpeg we pass the original file through: onnx-asr
    can resample it and mix channels itself.  For every other format ffmpeg is
    mandatory.
    """
    suffix = source.suffix.lower()
    if suffix == ".wav" and ffmpeg is None:
        return source, wav_duration(source)

    if ffmpeg is None:
        raise AudioError(
            "Для mp3/mp4/flac/ogg нужен ffmpeg. Переустановите приложение "
            "или положите ffmpeg.exe в папку bin рядом с программой."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    _run_ffmpeg(command)
    if not destination.is_file() or destination.stat().st_size <= 44:
        raise AudioError("ffmpeg создал пустой WAV: в файле нет звуковой дорожки.")
    return destination, wav_duration(destination)


def read_pcm16_mono(path: Path, *, sample_rate: int = 16_000) -> Any:
    """Read a normalised 16 kHz mono PCM16 WAV into a float32 NumPy array.

    This is the fallback path when VAD is unavailable: the caller cuts the
    array into fixed-length windows and passes each window to onnx-asr
    directly, so no temporary WAV files are needed.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - asr extras are installed
        raise AudioError("Для нарезки аудио нужен numpy (extras «asr»).") from exc

    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            if rate != sample_rate or channels != 1 or width != 2:
                raise AudioError(
                    "Ожидался WAV "
                    f"{sample_rate} Гц mono PCM16, получено {rate} Гц, "
                    f"каналов {channels}, {width * 8} бит на отсчёт."
                )
            frames = wav.readframes(wav.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioError(f"Не удалось прочитать WAV: {exc}") from exc

    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "неизвестно"
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
