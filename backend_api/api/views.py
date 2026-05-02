"""REST API views.

Every view filters by `request.user`. There is no admin-style cross-user
data access — one user can never see or mutate another user's data.
"""
from __future__ import annotations

import logging

from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from . import llm_client
from .file_utils import extract_text
from .models import Conversation, Message, UploadedFile, UserProfile
from .serializers import (
    ChatRequestSerializer,
    ConversationDetailSerializer,
    ConversationSerializer,
    MessageSerializer,
    UploadedFileSerializer,
    UserProfileSerializer,
)

logger = logging.getLogger(__name__)


class MeView(APIView):
    """Return / update the authenticated user's profile (incl. system prompt)."""

    def get(self, request):
        profile, _ = UserProfile.objects.get_or_create(
            user=request.user,
            defaults={"firebase_uid": request.user.username},
        )
        return Response(UserProfileSerializer(profile).data)

    def patch(self, request):
        profile, _ = UserProfile.objects.get_or_create(
            user=request.user,
            defaults={"firebase_uid": request.user.username},
        )
        serializer = UserProfileSerializer(profile, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)


class ConversationViewSet(viewsets.ModelViewSet):
    serializer_class = ConversationSerializer

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user)

    def get_serializer_class(self):
        if self.action in ("retrieve",):
            return ConversationDetailSerializer
        return ConversationSerializer

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=["get"])
    def messages(self, request, pk=None):
        conversation = self.get_object()
        qs = conversation.messages.all()
        return Response(MessageSerializer(qs, many=True).data)


class UploadedFileViewSet(viewsets.ModelViewSet):
    serializer_class = UploadedFileSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def get_queryset(self):
        return UploadedFile.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        upload = serializer.validated_data["file"]
        text = extract_text(upload, upload.name, upload.content_type or "")
        serializer.save(
            user=self.request.user,
            original_name=upload.name,
            mime_type=upload.content_type or "",
            size_bytes=getattr(upload, "size", 0) or 0,
            extracted_text=text,
        )


class ChatView(APIView):
    """Send a text chat message; persists user/assistant messages and returns the reply."""

    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        conv = self._get_or_create_conversation(request.user, data.get("conversation_id"))
        profile, _ = UserProfile.objects.get_or_create(
            user=request.user,
            defaults={"firebase_uid": request.user.username},
        )

        system_prompt = (
            conv.system_prompt_override
            or profile.custom_system_prompt
            or settings.LLM_DEFAULT_SYSTEM_PROMPT
        )

        history = [
            {"role": m.role, "content": m.content}
            for m in conv.messages.all()
        ]

        file_contexts: list[str] = []
        file_ids = data.get("file_ids") or []
        if file_ids:
            files = UploadedFile.objects.filter(user=request.user, id__in=file_ids)
            file_contexts = [
                f"# {f.original_name}\n{f.extracted_text}"
                for f in files
                if f.extracted_text
            ]

        Message.objects.create(conversation=conv, role="user", content=data["message"])

        try:
            reply = llm_client.chat_complete(
                llm_client.build_messages(
                    system_prompt=system_prompt,
                    history=history,
                    user_message=data["message"],
                    file_contexts=file_contexts,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM call failed: %s", exc)
            return Response(
                {"detail": "LLM service unavailable."},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        assistant_msg = Message.objects.create(
            conversation=conv, role="assistant", content=reply
        )
        conv.save(update_fields=["updated_at"])

        return Response(
            {
                "conversation_id": str(conv.id),
                "message": MessageSerializer(assistant_msg).data,
            }
        )

    @staticmethod
    def _get_or_create_conversation(user, conversation_id):
        if conversation_id:
            return get_object_or_404(Conversation, id=conversation_id, user=user)
        return Conversation.objects.create(user=user, title="New conversation")
