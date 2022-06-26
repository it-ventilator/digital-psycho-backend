import cv2
import dlib
import numpy as np


pose_predictor = dlib.shape_predictor('utils/models/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('utils/models/dlib_face_recognition_resnet_model_v1.dat')  


def dlib_detector(img):
    dlib_face_detector = dlib.get_frontal_face_detector()
    dlib_face_locations = dlib_face_detector(img)
    return dlib_face_locations  # xmin,ymin,xmax,ymax


def encodings(img, face_locations, pose_predictor, face_encoder):
    predictors = [pose_predictor(img, face_location) for face_location in face_locations]
    return [np.array(face_encoder.compute_face_descriptor(img, predictor, 1)) for predictor in predictors]


def get_image_embedding(img):
    if isinstance(img, str):
        img = cv2.imread(img)
    
    face_locations = dlib_detector(img)
    face_encodings = encodings(img, face_locations, pose_predictor, face_encoder)
    
    if not face_encodings:
        return face_encodings[0]
    
    return face_encodings
