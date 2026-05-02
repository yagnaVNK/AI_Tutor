# AI Tutor

A low-latency, voice-first AI conversational SaaS. Sign in with Google, chat by text or **talk live through your browser microphone**, upload files to ask questions about them, and customize the assistant with your own system prompt — all powered by your own open-source LLM.

The whole stack is a **monorepo of independent microservices** so each part — STT, LLM, TTS, VAD — can be scaled and GPU-attached independently on Google Cloud Run (or any other container platform).

---

## Highlights

- **Real-time voice** — push-to-talk microphone with sub-second latency end-to-end (STT → LLM → TTS), streamed over a single WebSocket.
- **Interruption support** — Silero VAD listens for the user speaking over the assistant and cancels the in-flight LLM/TTS stream so you can barge in naturally.
- **Bring-your-own LLM** — ships pointing at a local [LMStudio](https://lmstudio.ai/) instance, but the LLM service is just an OpenAI-compatible proxy, so any vLLM/Ollama/OpenAI/Together endpoint works.
- **Multilingual** — Whisper STT + Edge-TTS + any modern LLM all support 50+ languages out of the box.
- **Per-user isolation** — every conversation, message, and uploaded file is scoped to the authenticated Firebase UID. Verified server-side without needing a service-account JSON.
- **Two-mode local dev** — pure Docker Compose for one-command startup, or run each Python service natively in its own terminal so the inference services can use your GPU.

---

## Architecture

```
                                  ┌───────────────────────────────┐
                                  │     Frontend (React + Vite)   │
                                  │  Login · Chat · Mic · Files   │
                                  └──────────┬──────────┬─────────┘
                                             │          │
                            HTTP (REST/JSON) │          │  WebSocket (audio + JSON)
                                             │          │
                                  ┌──────────▼──┐    ┌──▼──────────────┐
                                  │ backend_api │    │ voice_gateway   │
                                  │  (Django)   │    │  (FastAPI WS)   │
                                  └──────┬──────┘    └──┬──┬──┬──┬─────┘
                                         │              │  │  │  │
                                  ┌──────▼──────┐       │  │  │  │
                                  │ PostgreSQL  │       │  │  │  │
                                  │ + pgvector  │       │  │  │  │
                                  └─────────────┘       │  │  │  │
                          ┌─────────────┬───────────────┘  │  │  │
                          │             │                  │  │  │
                  ┌───────▼─────┐ ┌─────▼──────┐ ┌─────────▼──┴──┴─┐
                  │ stt_service │ │ vad_service│ │ llm_service     │
                  │  (Whisper)  │ │ (Silero)   │ │ (proxy → LMStudio│
                  └─────────────┘ └────────────┘ │  / vLLM / etc.)  │
                                                 └──────────────────┘
                                                 ┌──────────────────┐
                                                 │ tts_service      │
                                                 │ (Edge-TTS / Qwen │
                                                 │  / Vui)          │
                                                 └──────────────────┘
```

### Services

| Service         | Tech                                         | Role                                                                 | GPU? |
| --------------- | -------------------------------------------- | -------------------------------------------------------------------- | :--: |
| `frontend`      | React 18 + Vite + Tailwind                   | UI: Google sign-in, chat, push-to-talk mic, file upload, prompt editor | – |
| `backend_api`   | Django 5 + DRF                               | Firebase token verification, conversations/messages/files in Postgres  | – |
| `voice_gateway` | FastAPI + WebSockets                         | Real-time orchestrator — VAD ⇄ STT → LLM → TTS streaming             | – |
| `stt_service`   | FastAPI + faster-whisper                     | Multilingual speech-to-text                                          | ✓ |
| `vad_service`   | FastAPI + Silero-VAD (ONNX)                  | Detects speech for interruption                                      | – |
| `llm_service`   | FastAPI proxy → OpenAI-compatible upstream   | Streaming LLM completions (LMStudio / vLLM / Ollama / etc.)          | ✓ |
| `tts_service`   | FastAPI + Edge-TTS (default) / Qwen3-TTS / Vui | Multilingual streaming text-to-speech, emits 16-bit PCM             | ✓ |

### Why WebSockets and not WebRTC?

Google Cloud Run does **not** support inbound UDP, and WebRTC requires UDP. WebSockets give us streaming audio over plain TCP/HTTP, which lets `voice_gateway` deploy as a stateless container that scales horizontally on Cloud Run. If you ever migrate it to GKE or Compute Engine, swap in `aiortc` without touching any other service.

### Streaming pipeline (voice path)

1. Browser captures mic audio with `MediaRecorder` (WebM/Opus, 250 ms chunks).
2. Each chunk is sent as a binary WebSocket frame to `voice_gateway`.
3. On `end_audio` the gateway forwards the buffer to `stt_service` (faster-whisper).
4. The transcript streams to `llm_service` which streams tokens back to the gateway.
5. The gateway buffers tokens by **sentence** and ships each finished sentence to `tts_service` immediately — so audio playback starts before the LLM has finished thinking.
6. `tts_service` returns 16-bit PCM mono chunks, which the gateway forwards as binary frames; the browser plays them through Web Audio with sub-100ms scheduling latency.
7. Throughout playback `vad_service` is checking incoming mic frames so the user can interrupt mid-sentence and the in-flight `assistant_task` is cancelled.

---

## Quick start

You need **either** Docker (one-command startup) **or** Python 3.12+, Node 20+, and a local LMStudio instance (more flexibility, GPU access). Pick one.

### 1. Clone and configure

```bash
git clone https://github.com/yagnaVNK/AI_Tutor.git
cd AI_Tutor
cp .env.example .env          # then edit it (Firebase keys, LMStudio URL)
```

You'll need a [Firebase project](https://console.firebase.google.com/) with **Google sign-in** enabled. Copy the web SDK config into the `VITE_FIREBASE_*` variables in `.env` and copy the project ID into `FIREBASE_PROJECT_ID`. No service-account JSON required for token verification — we use Google's public JWKs.

### 2a. One-command Docker Compose

```bash
docker compose up --build
```

- Frontend: <http://localhost:3000>
- Backend API: <http://localhost:8000>
- Voice gateway WS: `ws://localhost:8004/ws/voice`

LMStudio runs on the host, so the LLM service inside Docker reaches it via `host.docker.internal:1234` (already configured in `docker-compose.yml`). Start LMStudio's OpenAI-compatible server on port `1234` before bringing the stack up.

### 2b. Native local dev (recommended for GPU)

Use this when you want the inference services on your GPU without Docker GPU passthrough hassle.

```bash
# Create one shared venv for all Python services
python -m venv .venv
.venv\Scripts\activate           # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-all.txt

# Frontend
cd frontend && npm install && cd ..
```

Then run each service in its own terminal (each reads `.env` automatically):

```bash
# 1. backend_api (Django)
cd backend_api
python manage.py migrate
python manage.py runserver 0.0.0.0:8000

# 2. llm_service (proxy to LMStudio at http://127.0.0.1:1234)
cd llm_service && uvicorn main:app --port 8001

# 3. tts_service (Edge-TTS by default)
cd tts_service && uvicorn main:app --port 8002

# 4. stt_service (faster-whisper, CPU by default)
cd stt_service && uvicorn main:app --port 8003

# 5. vad_service (Silero VAD via ONNX)
cd vad_service && uvicorn main:app --port 8005

# 6. voice_gateway (orchestrator)
cd voice_gateway && uvicorn main:app --port 8004

# 7. frontend
cd frontend && npm run dev
```

By default `.env` uses **SQLite with WAL mode** so no Postgres is required for local dev. Flip `USE_POSTGRES=true` and start a Postgres container if you want the production-equivalent backend:

```bash
docker compose up postgres redis
```

---

## Configuration

All services read from the same root `.env` (see `.env.example` for the full list). The interesting knobs:

| Variable | Default | Notes |
| --- | --- | --- |
| `USE_POSTGRES` | `false` | `false` → SQLite (WAL mode) for local dev. `true` → connect to Postgres. |
| `LLM_UPSTREAM_URL` | `http://127.0.0.1:1234` | Any OpenAI-compatible endpoint (LMStudio, vLLM, Ollama, OpenAI, Together…). |
| `LLM_MODEL` | `meta-llama-3-8b-instruct` | Must match a model id served by your upstream. |
| `STT_MODEL` | `base` | faster-whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3`. |
| `STT_DEVICE` | `cpu` | `cpu` is the safe default. Set to `cuda` if you have CUDA + cuBLAS 12 in PATH. |
| `STT_LANGUAGE` | `en` | Whisper language hint (or empty to auto-detect). |
| `TTS_ENGINE` | `edge-tts` | `edge-tts` (default), `qwen3-tts`, `vui`, or `sine` (debug beep). |
| `TTS_VOICE` | `en-US-AriaNeural` | Edge-TTS voice id. List available voices: `edge-tts --list-voices`. |
| `FIREBASE_PROJECT_ID` | – | Required for server-side ID-token verification. |
| `VITE_FIREBASE_*` | – | Web SDK config consumed by the React frontend. |

---

## Repository layout

```
AI_Tutor/
├── README.md
├── .env.example
├── docker-compose.yml          # full local stack (Postgres + Redis + all services)
├── requirements-all.txt        # combined deps for native local dev
│
├── frontend/                   # React 18 + Vite + Tailwind web app
│   ├── src/
│   │   ├── screens/            # LoginScreen, ChatScreen
│   │   ├── components/         # Sidebar, MessageList, Composer, SettingsModal
│   │   └── services/           # firebase.js, api.js, voiceClient.js
│   └── vite.config.js
│
├── backend_api/                # Django 5 + DRF — auth, conversations, files
│   ├── api/                    # models, views, serializers, authentication, llm_client
│   └── ai_tutor_core/          # settings, urls, wsgi
│
├── voice_gateway/              # FastAPI + WebSocket orchestrator
│   ├── main.py                 # WS route: VAD → STT → LLM → TTS pipeline
│   ├── ai_clients.py           # async HTTP clients for the inference services
│   └── firebase_auth.py        # ID token verification
│
├── stt_service/                # faster-whisper STT (Python-native FFmpeg via PyAV)
├── vad_service/                # Silero-VAD ONNX
├── llm_service/                # OpenAI-compatible streaming proxy
└── tts_service/                # Edge-TTS / Qwen3-TTS / Vui
```

---

## Key design decisions

- **Microservices, not a monolith.** TTS and STT have very different cold-start costs, GPU requirements, and scaling characteristics from the stateless API. Splitting them lets each scale-to-zero independently on Cloud Run.
- **No `firebase-admin` for token verification.** We use `google.oauth2.id_token.verify_firebase_token`, which only needs the project ID and the public JWKs that Google publishes. That removes the operational pain of mounting a service-account JSON in every container.
- **Sentence-level TTS streaming.** The voice gateway flushes each completed sentence to `tts_service` as soon as the LLM emits it. The user hears the assistant start speaking before the response is done generating.
- **In-process FFmpeg via PyAV.** The STT and TTS services need to decode WebM/Opus (browser MediaRecorder) and MP3 (Edge-TTS) without requiring an `ffmpeg` binary on the host. PyAV bundles FFmpeg's libraries.
- **SQLite-first local dev.** Postgres is great in production but slow to spin up locally. The Django backend defaults to SQLite (WAL mode + busy timeout) so a contributor can be running in under a minute.

---

## Production deployment (Google Cloud Run)

Each service has its own `Dockerfile` and is independently deployable:

```bash
# Example: deploy stt_service with a GPU
gcloud run deploy stt-service \
  --source ./stt_service \
  --region us-central1 \
  --gpu 1 --gpu-type nvidia-l4 \
  --no-cpu-throttling --memory 8Gi --concurrency 4
```

Wire the URLs into the other services via env vars (`STT_SERVICE_URL`, `LLM_SERVICE_URL`, etc.). Front the public-facing services (`backend_api`, `voice_gateway`) with a single domain — for example via Cloud Run's domain mapping or a load balancer — and point `VITE_API_URL` / `VITE_VOICE_WS_URL` at it before building the frontend.

For Postgres, [Cloud SQL with the Auth Proxy sidecar](https://cloud.google.com/sql/docs/postgres/connect-run) is the path of least resistance.

---

## License

MIT. See `LICENSE` (add one if you want to be explicit).
