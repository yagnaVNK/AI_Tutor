"""STT engine wrapper.

Uses faster-whisper, which is the practical "Whisper for production"
choice — same models as openai/whisper but 4-5x faster on CPU and GPU.
The model is loaded once at startup and reused across requests so we
amortize the cold-start cost.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Iterable

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    from faster_whisper import WhisperModel

    model_name = os.environ.get("STT_MODEL", "base")
    requested_device = os.environ.get("STT_DEVICE", "auto").lower()
    # ctranslate2 wheels for CUDA need cuBLAS 12 in PATH. On Windows dev
    # boxes that's rarely true, and the LLM is already using the GPU via
    # LMStudio anyway, so we transparently fall back to CPU instead of
    # crashing every request with "Library cublas64_12.dll is not found".
    devices_to_try = ["cpu"] if requested_device == "cpu" else [requested_device, "cpu"]
    last_err: Exception | None = None
    for dev in devices_to_try:
        compute_type = os.environ.get(
            "STT_COMPUTE_TYPE",
            "int8" if dev in ("cpu", "auto") else "float16",
        )
        try:
            logger.info(
                "Loading whisper model=%s device=%s compute_type=%s",
                model_name, dev, compute_type,
            )
            candidate = WhisperModel(model_name, device=dev, compute_type=compute_type)
            # Probe a tiny silent buffer through the encoder so DLL load
            # failures (cublas, cudnn) surface here instead of on the first
            # real request as a 500 to the user.
            _probe(candidate)
            _model = candidate
            return _model
        except Exception as exc:
            last_err = exc
            logger.warning("Failed to load whisper on device=%s: %s", dev, exc)
    raise RuntimeError(f"Could not load whisper model on any device: {last_err}")


def _probe(model) -> None:
    silence = np.zeros(16000, dtype=np.float32)
    segments, _ = model.transcribe(
        silence, language="en", beam_size=1, vad_filter=False,
        condition_on_previous_text=False,
    )
    # iterating triggers encode/decode — that's where cuBLAS would fail
    for _ in segments:
        pass


def transcribe_bytes(audio_bytes: bytes, language: str | None = None) -> dict:
    """Transcribe a complete audio file (any format soundfile/ffmpeg can read)."""
    model = _get_model()
    try:
        samples, sample_rate = _decode(audio_bytes)
    except Exception as exc:
        # Browsers occasionally emit a tail blob without a valid container
        # header (e.g. when the user taps the mic and releases instantly).
        # Treat that as silence rather than a hard failure.
        logger.warning("Audio decode failed (%d bytes): %s", len(audio_bytes), exc)
        return {"text": "", "language": language or "", "duration": 0.0, "segments": []}
    if samples.size < 1600:  # < 0.1s at 16kHz: nothing useful to transcribe
        return {
            "text": "",
            "language": language or "",
            "duration": float(samples.size) / max(sample_rate, 1),
            "segments": [],
        }
    segments, info = model.transcribe(
        samples,
        language=language,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    text_parts: list[str] = []
    seg_list: list[dict] = []
    for seg in segments:
        text_parts.append(seg.text)
        seg_list.append({"start": seg.start, "end": seg.end, "text": seg.text})
    return {
        "text": "".join(text_parts).strip(),
        "language": info.language,
        "duration": info.duration,
        "segments": seg_list,
    }


def _decode(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode any audio container into mono float32 samples.

    Tries `soundfile` first (fastest path for WAV/FLAC/OGG). For WebM/Opus
    coming from the browser's MediaRecorder we fall back to PyAV, which
    bundles FFmpeg's libraries in-process — so we don't need an external
    `ffmpeg.exe` on the host.
    """
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except Exception:
        return _decode_with_pyav(audio_bytes)


def _decode_with_pyav(audio_bytes: bytes, target_rate: int = 16000) -> tuple[np.ndarray, int]:
    import av  # PyAV — comes with faster-whisper

    container = av.open(io.BytesIO(audio_bytes))
    try:
        stream = container.streams.audio[0]
    except IndexError:
        container.close()
        raise ValueError("No audio stream found in input")

    resampler = av.audio.resampler.AudioResampler(
        format="flt", layout="mono", rate=target_rate
    )
    chunks: list[np.ndarray] = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            chunks.append(arr.astype(np.float32, copy=False).reshape(-1))
    for resampled in resampler.resample(None):
        arr = resampled.to_ndarray()
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        chunks.append(arr.astype(np.float32, copy=False).reshape(-1))
    container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32), target_rate
    return np.concatenate(chunks), target_rate
