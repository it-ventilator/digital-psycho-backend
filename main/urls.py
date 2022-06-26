from django.urls import path, include
from rest_framework import routers

from .views import VideoViewSet, FrameEmotionView


router = routers.DefaultRouter()

router.register('videos', VideoViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('get_emotions/', FrameEmotionView.as_view())
]
