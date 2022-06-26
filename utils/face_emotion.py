import cv2
import torch
from PIL import Image

import face_recognition
from utils.emotion_detect import get_model, get_emotions_distribution


class FaceFinder:
    def __init__(self, mode='fast'):
        self.faceCascade = cv2.CascadeClassifier('utils/models/face_cascade.xml')
        self.mode = mode

    def find_all_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.mode == 'fast':
            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            faces = [[face[0] - 20, face[1] - 20, face[2] + 40, face[3] + 40] for face in faces if face[2] * face[3] > 10]
        else:
            face_locations = face_recognition.face_locations(gray)
            faces = [(face[1], face[0], face[1] - face[3], face[2] - face[0]) for face in face_locations]

        return faces
    
    
class FaceEmotionDetect:
    def __init__(self, mode='fast'):
        self.model = get_model()
        self.face_finder = FaceFinder(mode=mode)
    
    def find_all_emotions(self, frame):
        faces = self.face_finder.find_all_faces(frame)
        valid = [True for _ in range(len(faces))]

        labels = []
        
        for i, (x, y, w, h) in enumerate(faces):
            face_img = frame[y:y+h, x:x+w]

            if face_img.shape[0] != 0 and face_img.shape[1] != 0:
                emotion_label = get_emotions_distribution(self.model, Image.fromarray(face_img))
                labels.append(torch.softmax(emotion_label, 1)[0])
            else:
                valid[i] = False

        faces = [face for i, face in enumerate(faces) if valid[i]]

        return list(zip(faces, labels))
