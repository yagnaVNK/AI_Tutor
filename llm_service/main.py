"""LLM microservice.

Exposes an OpenAI-compatible /v1/chat/completions endpoint and proxies
to the upstream LLM inference engine (e.g., LMStudio, vLLM, Ollama, or
any OpenAI-compatible server). This indirection lets us:

  * deploy the LLM container independently and scale it on its own GPU
    auto-scaler in Cloud Run,
  * swap the underlying model server without touching backend_api or
    voice_gateway,
  * inject auth / rate limiting / observability in one place.

Configure with env vars:
  LLM_UPSTREAM_URL  OpenAI-compatible base URL (default: host.docker.internal:1234)
  LLM_UPSTREAM_API_KEY  bearer token for upstream
  LLM_MODEL  default model name to inject when callers omit it
"""
from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

import httpx
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("llm_service")
logging.basicConfig(level=logging.INFO)

UPSTREAM_URL = os.environ.get(
    "LLM_UPSTREAM_URL",
    os.environ.get("LLM_ENDPOINT", "http://host.docker.internal:1234"),
).rstrip("/")
UPSTREAM_API_KEY = os.environ.get("LLM_UPSTREAM_API_KEY", os.environ.get("LLM_API_KEY", "lm-studio"))
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "local-model")
REQUEST_TIMEOUT = float(os.environ.get("LLM_TIMEOUT_SECONDS", "300"))

app = FastAPI(title="ai-tutor llm-service", version="0.1.0")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "upstream": UPSTREAM_URL, "model": DEFAULT_MODEL}


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            r = await client.get(
                f"{UPSTREAM_URL}/v1/models",
                headers={"Authorization": f"Bearer {UPSTREAM_API_KEY}"},
            )
            return JSONResponse(r.json(), status_code=r.status_code)
        except httpx.HTTPError as exc:
            logger.warning("Upstream /v1/models failed: %s", exc)
            return JSONResponse(
                {"data": [{"id": DEFAULT_MODEL, "object": "model"}]}
            )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body: dict[str, Any] = await request.json()
    body.setdefault("model", DEFAULT_MODEL)
    stream = bool(body.get("stream"))

    headers = {
        "Authorization": f"Bearer {UPSTREAM_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{UPSTREAM_URL}/v1/chat/completions"

    if stream:
        return StreamingResponse(
            _stream_chat(url, headers, body),
            media_type="text/event-stream",
        )

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            r = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            logger.error("Upstream LLM call failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"Upstream LLM error: {exc}")
        return JSONResponse(r.json(), status_code=r.status_code)


async def _stream_chat(
    url: str, headers: dict[str, str], body: dict[str, Any]
) -> AsyncIterator[bytes]:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            async for chunk in resp.aiter_raw():
                if chunk:
                    yield chunk
