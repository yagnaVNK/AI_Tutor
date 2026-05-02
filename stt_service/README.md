# stt_service

Speech-to-text microservice powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Environment variables

| Variable           | Default | Description                                  |
| ------------------ | ------- | -------------------------------------------- |
| `STT_MODEL`        | `base`  | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `STT_DEVICE`       | `auto`  | `cpu` / `cuda` / `auto`                      |
| `STT_COMPUTE_TYPE` | `int8` (CPU) / `float16` (GPU) | quantization                  |
| `STT_LANGUAGE`     | `en`    | default language hint                        |
| `PORT`             | `8003`  | port FastAPI listens on                      |

## Endpoints

* `GET  /healthz`
* `POST /v1/stt` — multipart form with `file=<audio>` and optional `language`
