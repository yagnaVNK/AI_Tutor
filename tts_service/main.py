"""TTS service.

POST /v1/tts with {"text": "...", "stream": true|false}.
- stream=true returns chunked raw 16-bit PCM mono audio at the engine's
  sample rate. The voice gateway forwards these bytes straight to the
  browser for playback.
- stream=false returns a complete WAV file.
"""
from __future__ import annotations

import io
import logging
import os
from typing import AsyncIterator

import numpy as np
import soundfile as sf
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from engine import load_engine
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts_service")

app = FastAPI(title="ai-tutor tts-service", version="0.1.0")

ENGINE = load_engine()


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    stream: bool = True
    voice: str | None = None


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "engine": ENGINE.name,
        "sample_rate": ENGINE.sample_rate,
    }


@app.post("/v1/tts")
async def tts(req: TTSRequest):
    if req.stream:
        return StreamingResponse(
            _stream(req.text),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(ENGINE.sample_rate),
                "X-Audio-Format": "pcm_s16le",
                "X-Channels": "1",
            },
        )
    return Response(
        content=await _to_wav(req.text),
        media_type="audio/wav",
    )


async def _stream(text: str) -> AsyncIterator[bytes]:
    async for chunk in ENGINE.synthesize(text):
        yield chunk


async def _to_wav(text: str) -> bytes:
    buf = bytearray()
    async for chunk in ENGINE.synthesize(text):
        buf.extend(chunk)
    if not buf:
        raise HTTPException(status_code=500, detail="Engine produced no audio")
    samples = np.frombuffer(bytes(buf), dtype=np.int16)
    out = io.BytesIO()
    sf.write(out, samples, ENGINE.sample_rate, subtype="PCM_16", format="WAV")
    return out.getvalue()
