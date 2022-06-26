from glob import glob
from pathlib import Path

import cv2
import pandas as pd
from utils.face_emotion import FaceEmotionDetect

MODELS_DIR = Path(__file__).parent / 'utils' / 'models'

detect = FaceEmotionDetect()
emotions = list()

LABELS = ['angry', 'happy', 'neutral', 'sad', 'surprise']
MAPPING = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6,
}


def get_video_emotion(path):
    cap = cv2.VideoCapture(path)
    face_emotion_detect = FaceEmotionDetect()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        items = face_emotion_detect.find_all_emotions(frame)

        if not items:
            emotions.append([0] * len(LABELS))
        else:
            emotions.append(list(map(lambda x: round(x * 100), items[0][1].detach().numpy())))

    emotions_sum = pd.DataFrame(emotions, columns=LABELS).sum()
    label = emotions_sum[emotions_sum == emotions_sum.max()].index.tolist()[0]

    return MAPPING[label]


def handle_videos(main_path, save_path):
    values = []
    for path in glob(f'{main_path}/*'):
        emotion = get_video_emotion(path)
        values.append({'filename': Path(path).name.split(), 'emotion': emotion})
    df = pd.DataFrame(values)
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    handle_videos(
        main_path='data',
        save_path='emotions.csv',
    )
