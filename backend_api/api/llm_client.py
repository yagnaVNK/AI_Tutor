"""Tiny client around the LLM service.

The LLM service exposes an OpenAI-compatible /v1/chat/completions
endpoint backed by LMStudio (or any compatible model server). We talk
to it with httpx so the same code works in Django (sync) and could be
re-used elsewhere with minimal changes.
"""
from __future__ import annotations

import logging
from typing import Iterable

import httpx
from django.conf import settings

logger = logging.getLogger(__name__)


def build_messages(
    *,
    system_prompt: str,
    history: Iterable[dict],
    user_message: str,
    file_contexts: Iterable[str] = (),
) -> list[dict]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    file_blob = "\n\n---\n\n".join(t for t in file_contexts if t).strip()
    if file_blob:
        messages.append(
            {
                "role": "system",
                "content": (
                    "The user has attached the following documents. Use them "
                    "as context when answering.\n\n" + file_blob
                ),
            }
        )

    for m in history:
        if m.get("role") in ("user", "assistant", "system"):
            messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_message})
    return messages


def chat_complete(messages: list[dict]) -> str:
    """Non-streaming completion. Returns the full assistant text."""
    url = f"{settings.LLM_SERVICE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": settings.LLM_MODEL,
        "messages": messages,
        "stream": False,
        "temperature": 0.7,
    }
    headers = {
        "Authorization": f"Bearer {settings.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"]
