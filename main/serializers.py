from rest_framework import serializers
from .models import (
    VideoExample
)


class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoExample
        fields = ['id', 'video', 'uploaded_at', 'emotions']
        read_only_fields = ['id', 'uploaded_at', 'emotions']


class FrameEmotionSerializer(serializers.Serializer):
    img = serializers.CharField()
