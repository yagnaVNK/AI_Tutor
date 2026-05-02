"""STT service.

POST /v1/stt with a multipart 'file' field — returns transcribed text.
The voice gateway calls this once it has buffered enough user audio.
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from engine import transcribe_bytes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt_service")

app = FastAPI(title="ai-tutor stt-service", version="0.1.0")
DEFAULT_LANGUAGE = os.environ.get("STT_LANGUAGE", "en")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "model": os.environ.get("STT_MODEL", "base")}


@app.post("/v1/stt")
async def stt(
    file: UploadFile = File(...),
    language: str = Form(default=DEFAULT_LANGUAGE),
):
    if not file:
        raise HTTPException(status_code=400, detail="file is required")
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio")
    try:
        return transcribe_bytes(audio_bytes, language=language or None)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
