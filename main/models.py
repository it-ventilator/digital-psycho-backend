from django.db import models
from django.core.validators import FileExtensionValidator

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from main.tasks import set_video_emotions
from utils.face_embeddings import get_image_embedding


class VideoExample(models.Model):
    video = models.FileField(
        upload_to='video_example/',
        validators=[FileExtensionValidator(allowed_extensions=['MOV','avi','mp4','webm','mkv'])]
    )

    uploaded_at = models.DateTimeField(auto_now=True)
    emotions = models.JSONField(default=dict, blank=True)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        set_video_emotions.delay(self.id)


class OrganizationMember(models.Model):
    face_example = models.ImageField(upload_to='face_examples/')
    name = models.CharField(max_length=50)
    telegram = models.CharField(max_length=20)
    embedding = models.JSONField(default=dict, blank=True)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        self.embedding = {
            'embedding': get_image_embedding(self.face_example.path)[0].tolist()
        }
        super().save()

    @staticmethod
    def find_closest_member(img):
        embeddings = get_image_embedding(img)
        
        members_embeddings = OrganizationMember.objects.all()
        exist_embeddings = [m.embedding['embedding'] for m in members_embeddings]

        similarities = cosine_similarity(embeddings, exist_embeddings)[0]

        if np.max(similarities) < 0.93:
            return

        return members_embeddings[int(np.argmax(similarities))]
