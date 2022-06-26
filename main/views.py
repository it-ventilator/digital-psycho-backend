import base64
import io
import cv2

from rest_framework import viewsets
from rest_framework import generics
from rest_framework import permissions
from rest_framework.response import Response

import numpy as np
from PIL import Image

from main.models import VideoExample, OrganizationMember
from main.serializers import VideoSerializer, FrameEmotionSerializer
from utils.face_emotion import FaceEmotionDetect
from utils.emotion_detect import LABELS


deep_face_emotion_detect = FaceEmotionDetect(mode='deep')


class VideoViewSet(viewsets.ModelViewSet):
    permission_classes = (permissions.AllowAny,)
    queryset = VideoExample.objects.all()
    serializer_class = VideoSerializer


class FrameEmotionView(generics.GenericAPIView):
    permission_classes = (permissions.AllowAny, )
    serializer_class = FrameEmotionSerializer

    def post(self, request, format=None):
        img_base64 = base64.decodebytes(request.data['img'].encode('utf-8'))
        img = Image.open(io.BytesIO(img_base64))
        img = np.array(img)

        if len(img.shape) > 2 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        items = deep_face_emotion_detect.find_all_emotions(img)

        if not items:
            return Response({'message': 'there is no faces'})

        emotions = list(map(lambda x: round(x * 100), items[0][1].detach().numpy()))
        person = OrganizationMember.find_closest_member(img)

        response = {
            'emotions': LABELS,
            'percentage': emotions,
            'bbox': items[0][0]
        }

        if person:
            response['person'] = {
                'img': person.face_example.url,
                'name': person.name,
                'telegram': person.telegram
            }

        return Response(response)
