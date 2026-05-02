from django.contrib import admin

from .models import Conversation, Message, UploadedFile, UserProfile

admin.site.register(UserProfile)
admin.site.register(Conversation)
admin.site.register(Message)
admin.site.register(UploadedFile)
