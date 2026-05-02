"""Voice gateway WebSocket server.

Protocol (single bi-di WebSocket at /ws/voice?token=<firebase id token>
&conversation_id=<optional>):

Client -> Server (JSON or binary):
  { "type": "start_audio", "mime": "audio/webm", "language": "en" }
  <binary audio chunks>          # appended to the current utterance buffer
  { "type": "end_audio" }        # commit the utterance: STT -> LLM -> TTS
  { "type": "text", "text": "Hello" }   # text-only chat path
  { "type": "ping" }

Server -> Client (JSON unless noted):
  { "type": "ready", "conversation_id": "..." }
  { "type": "transcript", "text": "..." }
  { "type": "assistant_chunk", "text": "..." }
  { "type": "tts_meta", "sample_rate": 24000, "format": "pcm_s16le", "channels": 1 }
  <binary pcm chunks>            # raw audio frames
  { "type": "assistant_done", "text": "..." }
  { "type": "interrupt" }        # Sent when VAD detects user interruption
  { "type": "error", "detail": "..." }
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

# IMPORTANT: load .env BEFORE importing modules that read env vars at
# module load time (ai_clients, backend_client). Otherwise they fall back
# to Docker hostnames like "http://stt_service:8003" which won't resolve
# when running natively.
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

import ai_clients
import backend_client
import firebase_auth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_gateway")

app = FastAPI(title="ai-tutor voice-gateway", version="0.1.0")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"service": "voice_gateway", "ws": "/ws/voice"})


@app.websocket("/ws/voice")
async def voice_ws(
    websocket: WebSocket,
    token: str = Query(...),
    conversation_id: str | None = Query(default=None),
):
    await websocket.accept()

    try:
        decoded = firebase_auth.verify_token(token)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Auth failed: %s", exc)
        await websocket.send_json({"type": "error", "detail": "auth failed"})
        await websocket.close(code=4401)
        return

    user_uid = decoded.get("uid")
    logger.info("Voice WS connected uid=%s conv=%s", user_uid, conversation_id)

    session = VoiceSession(
        websocket=websocket,
        token=token,
        conversation_id=conversation_id,
    )
    try:
        await session.run()
    except WebSocketDisconnect:
        logger.info("Voice WS disconnected uid=%s", user_uid)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Voice WS crashed: %s", exc)
        try:
            await websocket.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
        await websocket.close(code=1011)


class VoiceSession:
    def __init__(
        self,
        *,
        websocket: WebSocket,
        token: str,
        conversation_id: str | None,
    ) -> None:
        self.ws = websocket
        self.token = token
        self.conversation_id = conversation_id
        self.audio_buf = bytearray()
        self.audio_mime = "audio/webm"
        self.language = os.environ.get("STT_LANGUAGE", "en")
        self.history: list[dict[str, str]] = []
        self.system_prompt: str = ""
        self.busy = False
        self.assistant_task: asyncio.Task | None = None
        self.vad_buffer = bytearray()
        self.last_vad_check = time.time()

    async def run(self) -> None:
        await self._bootstrap()
        await self.ws.send_json(
            {"type": "ready", "conversation_id": self.conversation_id}
        )

        while True:
            message = await self.ws.receive()
            if message.get("type") == "websocket.disconnect":
                return
            if "bytes" in message and message["bytes"] is not None:
                self.audio_buf.extend(message["bytes"])
                
                # Check for VAD interruption if we are busy generating TTS
                if self.busy:
                    self.vad_buffer.extend(message["bytes"])
                    # Check VAD every ~500ms of data (simplistic approach based on time)
                    if time.time() - self.last_vad_check > 0.5 and len(self.vad_buffer) > 4000:
                        self.last_vad_check = time.time()
                        vad_chunk = bytes(self.vad_buffer)
                        self.vad_buffer.clear()
                        
                        # Fire and forget VAD check so we don't block the receive loop
                        asyncio.create_task(self._check_interruption(vad_chunk))
                continue
                
            text = message.get("text")
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                await self.ws.send_json(
                    {"type": "error", "detail": "invalid JSON"}
                )
                continue
            await self._handle_control(payload)

    async def _check_interruption(self, audio_chunk: bytes) -> None:
        if not self.busy:
            return
        
        ext = "webm" if "webm" in self.audio_mime else "wav"
        is_speech = await ai_clients.check_vad(audio_chunk, filename=f"vad.{ext}")
        
        if is_speech and self.busy:
            logger.info("VAD detected speech. Interrupting current assistant generation.")
            if self.assistant_task and not self.assistant_task.done():
                self.assistant_task.cancel()
            self.busy = False
            await self.ws.send_json({"type": "interrupt"})

    async def _bootstrap(self) -> None:
        try:
            me = await backend_client.get_me(self.token)
            self.system_prompt = me.get("custom_system_prompt") or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch profile: %s", exc)
            self.system_prompt = ""

        if self.conversation_id:
            try:
                conv = await backend_client.get_conversation(
                    self.token, self.conversation_id
                )
                self.history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in conv.get("messages", [])
                ]
                if conv.get("system_prompt_override"):
                    self.system_prompt = conv["system_prompt_override"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load conversation history: %s", exc)
        else:
            try:
                conv = await backend_client.create_conversation(
                    self.token, title="Voice chat"
                )
                self.conversation_id = conv["id"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not create conversation: %s", exc)

        if not self.system_prompt:
            self.system_prompt = os.environ.get(
                "LLM_DEFAULT_SYSTEM_PROMPT",
                "You are a helpful AI tutor. Be concise, friendly, and clear.",
            )

    async def _handle_control(self, payload: dict[str, Any]) -> None:
        msg_type = payload.get("type")
        if msg_type == "ping":
            await self.ws.send_json({"type": "pong"})
            return
        if msg_type == "start_audio":
            self.audio_buf.clear()
            self.vad_buffer.clear()
            self.audio_mime = payload.get("mime", "audio/webm")
            self.language = payload.get("language") or self.language
            
            # If user starts speaking while we are busy, interrupt immediately.
            if self.busy:
                logger.info("User explicitly started audio. Interrupting.")
                if self.assistant_task and not self.assistant_task.done():
                    self.assistant_task.cancel()
                self.busy = False
                await self.ws.send_json({"type": "interrupt"})
            return
            
        if msg_type == "end_audio":
            await self._handle_voice_turn()
            return
        if msg_type == "text":
            text = (payload.get("text") or "").strip()
            if text:
                if self.busy and self.assistant_task and not self.assistant_task.done():
                    self.assistant_task.cancel()
                    self.busy = False
                await self._handle_text_turn(text)
            return
        if msg_type == "set_system_prompt":
            self.system_prompt = (payload.get("prompt") or "").strip() or self.system_prompt
            await self.ws.send_json({"type": "system_prompt_set"})
            return
        await self.ws.send_json(
            {"type": "error", "detail": f"unknown type: {msg_type}"}
        )

    async def _handle_voice_turn(self) -> None:
        if self.busy:
            await self.ws.send_json({"type": "error", "detail": "busy"})
            return
        if not self.audio_buf:
            await self.ws.send_json({"type": "error", "detail": "no audio"})
            return
        audio = bytes(self.audio_buf)
        self.audio_buf.clear()
        self.vad_buffer.clear()
        ext = "webm" if "webm" in self.audio_mime else "wav"
        
        try:
            self.busy = True
            transcript = await ai_clients.transcribe(
                audio, language=self.language, filename=f"audio.{ext}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("STT failed: %s", exc)
            await self.ws.send_json({"type": "error", "detail": f"STT failed: {exc}"})
            self.busy = False
            return
            
        await self.ws.send_json({"type": "transcript", "text": transcript})
        
        if not transcript:
            self.busy = False
            return
            
        self.assistant_task = asyncio.create_task(self._answer(transcript, speak=True))
        try:
            await self.assistant_task
        except asyncio.CancelledError:
            logger.info("Voice turn was cancelled by interruption.")
        finally:
            self.busy = False

    async def _handle_text_turn(self, text: str) -> None:
        if self.busy:
            await self.ws.send_json({"type": "error", "detail": "busy"})
            return
            
        self.assistant_task = asyncio.create_task(self._answer(text, speak=False))
        try:
            await self.assistant_task
        except asyncio.CancelledError:
            logger.info("Text turn was cancelled by interruption.")
        finally:
            self.busy = False

    async def _answer(self, user_text: str, *, speak: bool) -> None:
        self.busy = True
        try:
            messages = self._build_messages(user_text)
            full_reply: list[str] = []
            tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

            async def llm_producer() -> None:
                buf = ""
                try:
                    async for chunk in ai_clients.stream_llm(messages):
                        full_reply.append(chunk)
                        await self.ws.send_json(
                            {"type": "assistant_chunk", "text": chunk}
                        )
                        if speak:
                            buf += chunk
                            sentence, buf = _extract_sentence(buf)
                            while sentence:
                                await tts_queue.put(sentence)
                                sentence, buf = _extract_sentence(buf)
                    if speak and buf.strip():
                        await tts_queue.put(buf.strip())
                except asyncio.CancelledError:
                    raise
                finally:
                    if speak:
                        await tts_queue.put(None)

            async def tts_consumer() -> None:
                meta_sent = False
                while True:
                    sentence = await tts_queue.get()
                    if sentence is None:
                        return
                    try:
                        async for chunk, meta in ai_clients.stream_tts(sentence):
                            if meta and not meta_sent:
                                await self.ws.send_json({"type": "tts_meta", **meta})
                                meta_sent = True
                            await self.ws.send_bytes(chunk)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("TTS failed for sentence: %s", exc)

            tasks = [asyncio.create_task(llm_producer())]
            if speak:
                tasks.append(asyncio.create_task(tts_consumer()))
                
            await asyncio.gather(*tasks)

            assistant_text = "".join(full_reply).strip()
            if assistant_text:
                self.history.append({"role": "user", "content": user_text})
                self.history.append({"role": "assistant", "content": assistant_text})

                await self.ws.send_json(
                    {"type": "assistant_done", "text": assistant_text}
                )

                try:
                    await backend_client.post_chat(
                        self.token,
                        message=user_text,
                        conversation_id=self.conversation_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not persist chat: %s", exc)
        except asyncio.CancelledError:
            # We were interrupted. Optionally save partial text.
            assistant_text = "".join(full_reply).strip()
            if assistant_text:
                self.history.append({"role": "user", "content": user_text})
                self.history.append({"role": "assistant", "content": assistant_text})
            raise
        finally:
            self.busy = False

    def _build_messages(self, user_text: str) -> list[dict]:
        msgs: list[dict] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.extend(self.history)
        msgs.append({"role": "user", "content": user_text})
        return msgs


def _extract_sentence(buf: str) -> tuple[str | None, str]:
    """Pull the next sentence-ish chunk out of a streaming buffer."""
    for i, ch in enumerate(buf):
        if ch in ".!?\n" and i >= 12:
            return buf[: i + 1].strip(), buf[i + 1 :]
    if len(buf) > 240:
        return buf.strip(), ""
    return None, buf
