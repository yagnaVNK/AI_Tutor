"""Async clients for the LLM, TTS, STT, and VAD microservices."""
from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://llm_service:8001").rstrip("/")
TTS_SERVICE_URL = os.environ.get("TTS_SERVICE_URL", "http://tts_service:8002").rstrip("/")
STT_SERVICE_URL = os.environ.get("STT_SERVICE_URL", "http://stt_service:8003").rstrip("/")
VAD_SERVICE_URL = os.environ.get("VAD_SERVICE_URL", "http://vad_service:8005").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "lm-studio")
LLM_MODEL = os.environ.get("LLM_MODEL", "local-model")


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------
async def check_vad(audio_bytes: bytes, filename: str = "audio.webm") -> bool:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            files = {"file": (filename, audio_bytes, "application/octet-stream")}
            r = await client.post(f"{VAD_SERVICE_URL}/v1/vad", files=files)
            r.raise_for_status()
            return r.json().get("speech_detected", False)
    except Exception as exc:
        logger.warning(f"VAD check failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------
async def transcribe(audio_bytes: bytes, language: str = "en", filename: str = "audio.webm") -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        files = {"file": (filename, audio_bytes, "application/octet-stream")}
        data = {"language": language}
        r = await client.post(f"{STT_SERVICE_URL}/v1/stt", files=files, data=data)
        r.raise_for_status()
        return (r.json().get("text") or "").strip()


# ---------------------------------------------------------------------------
# LLM (streaming)
# ---------------------------------------------------------------------------
async def stream_llm(messages: list[dict], temperature: float = 0.7) -> AsyncIterator[str]:
    """Yield text chunks as they arrive from the LLM service."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", f"{LLM_SERVICE_URL}/v1/chat/completions",
            headers=headers, json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (
                        obj.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if delta:
                        yield delta


# ---------------------------------------------------------------------------
# TTS (streaming raw PCM)
# ---------------------------------------------------------------------------
async def stream_tts(text: str) -> AsyncIterator[tuple[bytes, dict]]:
    """Yield (pcm_chunk, metadata) — metadata is sent only with the first chunk.

    httpx aggregates raw TCP reads, so a single yielded chunk can split a
    16-bit PCM sample down the middle (odd byte count). The browser's
    Int16Array constructor rejects odd-length buffers, so we carry over
    any trailing odd byte into the next chunk to keep every forwarded
    payload aligned to the 2-byte int16 frame size.
    """
    payload = {"text": text, "stream": True}
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", f"{TTS_SERVICE_URL}/v1/tts", json=payload,
        ) as resp:
            resp.raise_for_status()
            meta = {
                "sample_rate": int(resp.headers.get("X-Sample-Rate", "24000")),
                "format": resp.headers.get("X-Audio-Format", "pcm_s16le"),
                "channels": int(resp.headers.get("X-Channels", "1")),
            }
            sent_meta = False
            carry = b""
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                if carry:
                    chunk = carry + chunk
                    carry = b""
                if len(chunk) % 2:
                    carry = chunk[-1:]
                    chunk = chunk[:-1]
                if not chunk:
                    continue
                if not sent_meta:
                    yield chunk, meta
                    sent_meta = True
                else:
                    yield chunk, {}
            if carry:
                # Pad the final stray byte with a zero so it's still playable.
                tail = carry + b"\x00"
                if not sent_meta:
                    yield tail, meta
                else:
                    yield tail, {}
