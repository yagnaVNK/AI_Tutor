"""Lightweight Firebase token verification for the voice gateway.

The token comes in the WebSocket connect URL as ?token=... (browsers can't
set custom headers on the WS handshake). We verify it with
`google.oauth2.id_token.verify_firebase_token` which only needs the
project id — no service-account JSON or ADC required.
"""
from __future__ import annotations

import logging
import os

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

logger = logging.getLogger(__name__)

_request_session = google_requests.Request()


def _get_project_id() -> str:
    project_id = os.environ.get("FIREBASE_PROJECT_ID") or os.environ.get(
        "VITE_FIREBASE_PROJECT_ID"
    )
    if not project_id:
        raise RuntimeError("FIREBASE_PROJECT_ID is required to verify tokens")
    return project_id


def verify_token(token: str) -> dict:
    project_id = _get_project_id()
    decoded = google_id_token.verify_firebase_token(
        token, _request_session, audience=project_id
    )
    if "uid" not in decoded:
        decoded["uid"] = decoded.get("user_id") or decoded.get("sub")
    return decoded
