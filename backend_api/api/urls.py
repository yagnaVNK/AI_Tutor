from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import ChatView, ConversationViewSet, MeView, UploadedFileViewSet

router = DefaultRouter()
router.register(r"conversations", ConversationViewSet, basename="conversation")
router.register(r"files", UploadedFileViewSet, basename="file")

urlpatterns = [
    path("me", MeView.as_view(), name="me"),
    path("chat", ChatView.as_view(), name="chat"),
    path("", include(router.urls)),
]
