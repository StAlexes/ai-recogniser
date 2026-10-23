"""POC: measure GigaAM v3 e2e CTC (int8) end to end on a real Russian WAV.

Reports model load time, audio duration, warmup + timed inference runs, RTFx,
peak RSS and the recognised text as a single JSON object.

Example:
    .\\.venv\\Scripts\\python.exe tools\\poc_gigaam.py --wav data\\samples\\gigaam-example.wav ^
        --json data\\poc\\gigaam-result.json --transcript-out data\\poc\\transcript.txt
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GigaAM v3 e2e CTC int8 end-to-end POC")
    parser.add_argument("--wav", required=True, help="path to a mono 16 kHz PCM WAV file")
    parser.add_argument("--model", default="gigaam-v3-e2e-ctc", help="onnx-asr model id")
    parser.add_argument("--quantization", default="int8", help="onnx-asr quantization (int8, fp32, ...)")
    parser.add_argument("--json", dest="json_out", default=None, help="output JSON path")
    parser.add_argument("--transcript-out", default=None, help="optional plain-text transcript path")
    parser.add_argument("--runs", type=int, default=2, help="number of timed runs after warmup")
    parser.add_argument(
        "--provider",
        choices=("cpu", "dml", "auto"),
        default="cpu",
        help="execution provider: cpu, dml (DirectML) or auto (DML if available, else CPU)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="only load (and download) the model, then exit with the cold load time",
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    return args


class PeakRss:
    """Sample process RSS in a background thread and track a high-water mark."""

    def __init__(self, interval: float = 0.05) -> None:
        self._proc = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.peak = 0

    def _sample(self) -> None:
        while not self._stop.wait(self._interval):
            self.peak = max(self.peak, self._proc.memory_info().rss)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        """Stop sampling and return the sampled peak (plus the process high-water mark)."""
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.peak = max(self.peak, self._proc.memory_info().rss)
        return self.peak

    def lifetime_peak(self) -> int:
        """Windows: peak working set since process start (includes model download)."""
        return max(self.peak, getattr(self._proc.memory_info(), "peak_wset", 0))


def wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return {
            "duration_seconds": round(frames / float(rate), 3) if rate else 0.0,
            "sample_rate": rate,
            "channels": w.getnchannels(),
            "bits_per_sample": w.getsampwidth() * 8,
        }


def resolve_providers(choice: str) -> list[str]:
    available = ort.get_available_providers()
    if choice == "cpu":
        return ["CPUExecutionProvider"]
    if choice == "dml":
        if "DmlExecutionProvider" not in available:
            raise SystemExit(
                "DirectML is not available: DmlExecutionProvider is missing from "
                f"{available}. Install onnxruntime-directml in a separate venv; "
                "do not replace onnxruntime in this one."
            )
        return ["DmlExecutionProvider"]
    if "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider"]
    return ["CPUExecutionProvider"]


def machine_info() -> dict[str, Any]:
    uname = platform.uname()
    vm = psutil.virtual_memory()
    return {
        "os": f"{uname.system} {uname.release}",
        "machine": uname.machine,
        "cpu": uname.processor or uname.machine,
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 1024**3, 2),
    }


def versions() -> dict[str, str]:
    import onnx_asr

    return {
        "python": platform.python_version(),
        "onnx_asr": getattr(onnx_asr, "__version__", "unknown"),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
    }


def emit(result: dict[str, Any], json_out: str | None) -> None:
    """Save the result as UTF-8 JSON and print the same object to stdout."""
    out = Path(json_out).expanduser() if json_out else Path("data/poc/gigaam-result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Windows consoles often run in cp866/cp1251; keep JSON valid when redirected.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    wav = Path(args.wav).expanduser()
    if not wav.is_file():
        print(f"error: WAV not found: {wav}", file=sys.stderr)
        return 2

    providers = resolve_providers(args.provider)
    available = ort.get_available_providers()
    audio = wav_info(wav)
    sampler = PeakRss()
    sampler.start()

    import onnx_asr

    print(
        f"loading {args.model} ({args.quantization}) on {providers}; first run may download ~214 MB",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    model = onnx_asr.load_model(args.model, quantization=args.quantization, providers=providers)
    load_seconds = time.perf_counter() - t0
    rss_after_load = sampler.stop()
    peak_wset_after_load = sampler.lifetime_peak()

    if args.download_only:
        result = {
            "schema": "poc-gigaam/1",
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "mode": "download-only",
            "machine": machine_info(),
            "model": {
                "id": args.model,
                "quantization": args.quantization,
                "providers_available": available,
                "providers_requested": providers,
                "download_and_load_seconds": round(load_seconds, 3),
            },
            "audio": {"path": str(wav), **audio},
            "rss_after_load_mb": round(rss_after_load / 1024**2, 1),
            "peak_wset_mb": round(peak_wset_after_load / 1024**2, 1),
            "versions": versions(),
        }
        emit(result, args.json_out)
        return 0

    # Warmup: populates ORT arenas/caches, initializes preprocessors; not timed.
    print("warmup run", file=sys.stderr)
    sampler2 = PeakRss()
    sampler2.peak = rss_after_load
    sampler2.start()
    t0 = time.perf_counter()
    warmup_text = model.recognize(str(wav))
    warmup_seconds = time.perf_counter() - t0

    timed: list[float] = []
    texts: list[str] = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        text = model.recognize(str(wav))
        elapsed = time.perf_counter() - t0
        timed.append(elapsed)
        texts.append(text)
        print(f"run {i + 1}/{args.runs}: {elapsed:.3f}s -> {len(text)} chars", file=sys.stderr)

    peak = sampler2.stop()
    peak_wset = sampler2.lifetime_peak()
    mean_seconds = sum(timed) / len(timed)
    best_seconds = min(timed)
    duration = float(audio["duration_seconds"])
    text = texts[-1]

    result = {
        "schema": "poc-gigaam/1",
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": "recognize",
        "machine": machine_info(),
        "model": {
            "id": args.model,
            "quantization": args.quantization,
            "providers_available": available,
            "providers_requested": providers,
            "load_seconds": round(load_seconds, 3),
        },
        "audio": {"path": str(wav), **audio},
        "measurements": {
            "warmup_seconds": round(warmup_seconds, 3),
            "runs": args.runs,
            "run_seconds": [round(t, 3) for t in timed],
            "infer_seconds_mean": round(mean_seconds, 3),
            "infer_seconds_best": round(best_seconds, 3),
            "rtfx_mean": round(duration / mean_seconds, 2) if mean_seconds else 0.0,
            "rtfx_best": round(duration / best_seconds, 2) if best_seconds else 0.0,
            "peak_rss_mb": round(peak / 1024**2, 1),
            "peak_wset_mb": round(peak_wset / 1024**2, 1),
            "rss_after_load_mb": round(rss_after_load / 1024**2, 1),
        },
        "text_length_chars": len(text),
        "text_length_words": len(text.split()),
        "text": text,
        "warmup_text": warmup_text,
        "versions": versions(),
    }

    if args.transcript_out:
        tr = Path(args.transcript_out).expanduser()
        tr.parent.mkdir(parents=True, exist_ok=True)
        tr.write_text(text + "\n", encoding="utf-8")

    emit(result, args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
