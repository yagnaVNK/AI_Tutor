# llm_service

Thin FastAPI wrapper that exposes an OpenAI-compatible API and proxies to the configured upstream LLM (e.g. LMStudio, vLLM, Ollama).

## Why a proxy?

* `voice_gateway` and `backend_api` only speak one protocol (OpenAI's), regardless of which model server is behind the scenes.
* You can deploy this container to **Cloud Run with GPU** (or GKE, or a VM) and scale it independently.
* In local dev it can simply forward to your laptop's LMStudio at `http://host.docker.internal:1234`.

## Environment variables

| Variable               | Default                              | Description                              |
| ---------------------- | ------------------------------------ | ---------------------------------------- |
| `LLM_UPSTREAM_URL`     | `http://host.docker.internal:1234`   | Base URL of the upstream OpenAI-style API |
| `LLM_UPSTREAM_API_KEY` | `lm-studio`                          | Bearer token sent to the upstream        |
| `LLM_MODEL`            | `local-model`                        | Default model id                         |
| `LLM_TIMEOUT_SECONDS`  | `300`                                | HTTP timeout                             |
| `PORT`                 | `8001`                               | Port FastAPI listens on                  |

## Endpoints

* `GET  /healthz`
* `GET  /v1/models`
* `POST /v1/chat/completions` (supports `stream: true`)
