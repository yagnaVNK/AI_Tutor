# tts_service

Streaming text-to-speech microservice. Pluggable engines:

| `TTS_ENGINE` value | Backend                                  | Notes                                    |
| ------------------ | ---------------------------------------- | ---------------------------------------- |
| `sine` (default)   | Built-in CPU sine wave generator         | Lets the whole stack run with no GPU     |
| `qwen3-tts`        | Alibaba Qwen3-TTS                        | Stub — wire in the real model in `engine.py` |
| `vui`              | fluxions-ai/vui                          | Stub — wire in the real model in `engine.py` |

When you pick a real engine, install its package in `requirements.txt` and replace the synthesize body in `engine.py` so it streams raw 16-bit PCM mono frames at `TTS_SAMPLE_RATE` (default 24000).

## Endpoints

* `GET  /healthz`
* `POST /v1/tts` body: `{"text": "...", "stream": true}`
  * `stream=true` → chunked raw PCM (`application/octet-stream`)
  * `stream=false` → full WAV (`audio/wav`)
