# vad_service

Voice Activity Detection microservice powered by [Silero-VAD](https://github.com/snakers4/silero-vad).

## Endpoints

* `GET  /healthz`
* `POST /v1/vad` — multipart form with `file=<audio>`
  * Returns `{"speech_detected": boolean, "probability": float}`
