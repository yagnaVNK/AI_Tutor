"""Async HTTP client for talking to backend_api on behalf of the user.

We always forward the user's Firebase ID token so backend_api enforces
strict per-user data isolation — the gateway has no admin powers and
can never see another user's data.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "http://backend_api:8000").rstrip("/")


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


async def get_me(token: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{BACKEND_API_URL}/api/me", headers=_headers(token))
        r.raise_for_status()
        return r.json()


async def get_conversation(token: str, conversation_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{BACKEND_API_URL}/api/conversations/{conversation_id}/",
            headers=_headers(token),
        )
        r.raise_for_status()
        return r.json()


async def create_conversation(token: str, title: str = "Voice chat") -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"{BACKEND_API_URL}/api/conversations/",
            headers=_headers(token),
            json={"title": title},
        )
        r.raise_for_status()
        return r.json()


async def post_chat(
    token: str,
    *,
    message: str,
    conversation_id: str | None = None,
    file_ids: list[str] | None = None,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(
            f"{BACKEND_API_URL}/api/chat",
            headers=_headers(token),
            json={
                "message": message,
                "conversation_id": conversation_id,
                "file_ids": file_ids or [],
            },
        )
        r.raise_for_status()
        return r.json()
