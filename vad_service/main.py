"""VAD microservice using Silero-VAD.

Exposes a simple HTTP endpoint to check if an audio chunk contains speech.
Useful for detecting interruptions from the user.
"""
import io
import logging
import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

import numpy as np
import onnxruntime
import soundfile as sf

from fastapi import FastAPI, File, UploadFile, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vad_service")

app = FastAPI(title="ai-tutor vad-service", version="0.1.0")

# Silero VAD operates at 16kHz
SAMPLE_RATE = 16000
WINDOW_SIZE = 512  # samples per inference window for 16kHz Silero

# Load Silero VAD ONNX model. The Dockerfile pre-downloads it to /app/silero_vad.onnx
MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
MODEL_PATH_CANDIDATES = [
    "/app/silero_vad.onnx",
    os.path.join(os.path.dirname(__file__), "silero_vad.onnx"),
]


def _download_model(target_path: str) -> bool:
    import urllib.request
    try:
        logger.info("Downloading Silero VAD model to %s ...", target_path)
        urllib.request.urlretrieve(MODEL_URL, target_path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to download Silero VAD model: %s", exc)
        return False


session = None
for candidate in MODEL_PATH_CANDIDATES:
    if not os.path.exists(candidate):
        try:
            os.makedirs(os.path.dirname(candidate) or ".", exist_ok=True)
        except Exception:
            continue
        if not _download_model(candidate):
            continue
    try:
        session = onnxruntime.InferenceSession(
            candidate, providers=["CPUExecutionProvider"]
        )
        logger.info("Loaded Silero VAD model from %s", candidate)
        break
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s: %s", candidate, exc)

if session is None:
    logger.error("Silero VAD model not loaded. /v1/vad will return errors.")


def _decode_with_pyav(audio_bytes: bytes, target_rate: int = 16000) -> np.ndarray:
    import av  # In-process FFmpeg via PyAV

    container = av.open(io.BytesIO(audio_bytes))
    try:
        stream = container.streams.audio[0]
    except IndexError:
        container.close()
        raise ValueError("No audio stream found")

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
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def _decode(audio_bytes: bytes) -> np.ndarray:
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            return _decode_with_pyav(audio_bytes, SAMPLE_RATE)
        return data
    except Exception:
        return _decode_with_pyav(audio_bytes, SAMPLE_RATE)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok" if session is not None else "model_missing", "model": "silero-vad"}


@app.post("/v1/vad")
async def process_vad(file: UploadFile = File(...)):
    """Check if the given audio chunk contains speech."""
    if session is None:
        raise HTTPException(status_code=500, detail="VAD model not loaded")

    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio")

    try:
        samples = _decode(audio_bytes)

        # Pad to at least one window
        if len(samples) < WINDOW_SIZE:
            samples = np.pad(samples, (0, WINDOW_SIZE - len(samples)))

        # Run inference over consecutive windows; report max probability.
        # Silero ONNX expects (batch_size, sequence_length) per call.
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)
        sr_arr = np.array(SAMPLE_RATE, dtype=np.int64)

        max_prob = 0.0
        for start in range(0, len(samples) - WINDOW_SIZE + 1, WINDOW_SIZE):
            chunk = samples[start : start + WINDOW_SIZE].astype(np.float32)
            input_data = np.expand_dims(chunk, axis=0)
            ort_inputs = {
                "input": input_data,
                "h": h,
                "c": c,
                "sr": sr_arr,
            }
            try:
                out, h, c = session.run(None, ort_inputs)
            except Exception:
                # Older Silero ONNX exposes a different signature with state inside the graph.
                ort_inputs = {"input": input_data, "sr": sr_arr}
                out = session.run(None, ort_inputs)[0]
            prob = float(out.ravel()[0])
            if prob > max_prob:
                max_prob = prob

        return {
            "speech_detected": max_prob > 0.5,
            "probability": max_prob,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("VAD failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
