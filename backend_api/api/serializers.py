from rest_framework import serializers

from .models import Conversation, Message, UploadedFile, UserProfile


class UserProfileSerializer(serializers.ModelSerializer):
    email = serializers.EmailField(source="user.email", read_only=True)

    class Meta:
        model = UserProfile
        fields = (
            "firebase_uid",
            "email",
            "display_name",
            "photo_url",
            "custom_system_prompt",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("firebase_uid", "created_at", "updated_at")


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ("id", "role", "content", "audio_url", "created_at")
        read_only_fields = ("id", "created_at")


class ConversationSerializer(serializers.ModelSerializer):
    message_count = serializers.IntegerField(source="messages.count", read_only=True)

    class Meta:
        model = Conversation
        fields = (
            "id",
            "title",
            "system_prompt_override",
            "message_count",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("id", "created_at", "updated_at", "message_count")


class ConversationDetailSerializer(ConversationSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta(ConversationSerializer.Meta):
        fields = ConversationSerializer.Meta.fields + ("messages",)


class UploadedFileSerializer(serializers.ModelSerializer):
    file = serializers.FileField(write_only=True)
    download_url = serializers.SerializerMethodField()

    class Meta:
        model = UploadedFile
        fields = (
            "id",
            "original_name",
            "mime_type",
            "size_bytes",
            "conversation",
            "extracted_text",
            "file",
            "download_url",
            "created_at",
        )
        read_only_fields = (
            "id",
            "original_name",
            "mime_type",
            "size_bytes",
            "extracted_text",
            "download_url",
            "created_at",
        )

    def get_download_url(self, obj: UploadedFile) -> str:
        if not obj.file:
            return ""
        request = self.context.get("request")
        url = obj.file.url
        return request.build_absolute_uri(url) if request else url


class ChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField(allow_blank=False, trim_whitespace=False)
    conversation_id = serializers.UUIDField(required=False, allow_null=True)
    file_ids = serializers.ListField(
        child=serializers.UUIDField(),
        required=False,
        default=list,
    )
