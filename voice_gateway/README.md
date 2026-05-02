# voice_gateway

Real-time WebSocket orchestrator that:

1. Verifies a Firebase ID token on connect.
2. Loads the user's profile + conversation history from `backend_api`.
3. Receives microphone audio chunks from the browser, buffers them, sends them to `stt_service` when the user finishes speaking.
4. Streams the transcript through `llm_service` and pipes sentence-sized chunks to `tts_service` as soon as they're available.
5. Pushes raw PCM frames back over the same WebSocket for instant playback.

## Why one WebSocket and not WebRTC?

Cloud Run does not allow inbound UDP, so WebRTC isn't possible there. WebSockets work everywhere we need to deploy. If you ever move this service to GKE / Compute Engine you can swap in `aiortc` while leaving every other service untouched.

## Endpoints

* `GET  /healthz`
* `WS   /ws/voice?token=<firebase id token>&conversation_id=<optional uuid>`

See `main.py` for the JSON message protocol.
