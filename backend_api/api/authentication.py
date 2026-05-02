"""Firebase ID token authentication for Django REST Framework.

Frontend signs in with Firebase (Google), gets an ID token, and sends it
as `Authorization: Bearer <token>`. We verify the token using
`google.oauth2.id_token.verify_firebase_token`, which only needs the
project id and downloads Google's public signing keys at runtime — no
service-account JSON or Application Default Credentials required.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from django.conf import settings
from django.contrib.auth import get_user_model
from rest_framework import authentication, exceptions

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

from .models import UserProfile

logger = logging.getLogger(__name__)
User = get_user_model()

_request_session = google_requests.Request()


def _get_project_id() -> str:
    project_id = (
        os.environ.get("FIREBASE_PROJECT_ID")
        or os.environ.get("VITE_FIREBASE_PROJECT_ID")
        or getattr(settings, "FIREBASE_PROJECT_ID", "")
    )
    if not project_id:
        raise exceptions.AuthenticationFailed(
            "Server is missing FIREBASE_PROJECT_ID."
        )
    return project_id


class FirebaseAuthentication(authentication.BaseAuthentication):
    keyword = "Bearer"

    def authenticate(self, request) -> Optional[Tuple[User, dict]]:
        auth_header = authentication.get_authorization_header(request).decode("utf-8")
        if not auth_header:
            return None
        parts = auth_header.split()
        if len(parts) != 2 or parts[0] != self.keyword:
            return None
        token = parts[1]

        project_id = _get_project_id()
        try:
            decoded = google_id_token.verify_firebase_token(
                token, _request_session, audience=project_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Firebase token verification failed: %s", exc)
            raise exceptions.AuthenticationFailed("Invalid Firebase token.")

        firebase_uid = decoded.get("user_id") or decoded.get("sub")
        email = decoded.get("email") or f"{firebase_uid}@firebase.local"
        display_name = decoded.get("name", "")
        photo_url = decoded.get("picture", "")

        if not firebase_uid:
            raise exceptions.AuthenticationFailed("Token missing uid.")

        user, created = User.objects.get_or_create(
            username=firebase_uid,
            defaults={"email": email, "first_name": display_name[:30]},
        )
        if not created and user.email != email:
            user.email = email
            user.save(update_fields=["email"])

        # Avoid an UPDATE on every request: only touch UserProfile when the
        # row is missing or the cached fields actually changed.
        try:
            profile = user.profile
        except UserProfile.DoesNotExist:
            profile = None
        if profile is None:
            UserProfile.objects.create(
                user=user,
                firebase_uid=firebase_uid,
                display_name=display_name,
                photo_url=photo_url,
            )
        elif (
            profile.firebase_uid != firebase_uid
            or profile.display_name != display_name
            or profile.photo_url != photo_url
        ):
            profile.firebase_uid = firebase_uid
            profile.display_name = display_name
            profile.photo_url = photo_url
            profile.save(update_fields=["firebase_uid", "display_name", "photo_url"])

        return user, decoded

    def authenticate_header(self, request) -> str:
        return self.keyword
