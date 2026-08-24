"""Real-time drowsiness detection: webcam -> face/eye detection -> CNN
classification -> fatigue score -> alarm.

Combines two independent cues into one fatigue score:
  - eyes closed for many consecutive frames
  - mouth open (yawning) detected
"""
import json
import os
import threading
import time

import cv2
import numpy as np
import pygame
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EYE_MODEL_PATH = os.path.join(BASE_DIR, "models", "eye_model.keras")
EYE_CLASSES_PATH = os.path.join(BASE_DIR, "models", "eye_classes.json")
MOUTH_MODEL_PATH = os.path.join(BASE_DIR, "models", "mouth_model.keras")
MOUTH_CLASSES_PATH = os.path.join(BASE_DIR, "models", "mouth_classes.json")
ALARM_PATH = os.path.join(BASE_DIR, "alarm.wav")

EYE_IMG_SIZE = (64, 64)
MOUTH_IMG_SIZE = (64, 64)

# How many consecutive "trouble" frames before the alarm fires.
DROWSY_SCORE_THRESHOLD = 15
SCORE_MIN = 0
SCORE_MAX = 30
EYE_SCORE_STEP = 1
YAWN_SCORE_STEP = 2  # a yawn is a stronger single signal than one closed-eye frame

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
LEFT_EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
RIGHT_EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_righteye_2splits.xml")


def load_classes(path):
    with open(path) as f:
        return json.load(f)


class AlarmPlayer:
    """Plays the alarm on a background thread so it doesn't block the video loop,
    and avoids re-triggering overlapping playback."""

    def __init__(self, path):
        pygame.mixer.init()
        self.sound = pygame.mixer.Sound(path)
        self._playing = False

    def play_if_idle(self):
        if self._playing:
            return
        self._playing = True
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self):
        channel = self.sound.play()
        while channel is not None and channel.get_busy():
            time.sleep(0.05)
        self._playing = False


def preprocess_eye(gray_frame, box):
    x, y, w, h = box
    crop = gray_frame[y:y + h, x:x + w]
    crop = cv2.resize(crop, EYE_IMG_SIZE)
    crop = crop.astype("float32") / 255.0
    return crop.reshape(1, EYE_IMG_SIZE[0], EYE_IMG_SIZE[1], 1)


def preprocess_mouth(color_frame, face_box):
    x, y, w, h = face_box
    mouth_top = y + int(h * 2 / 3)
    crop = color_frame[mouth_top:y + h, x:x + w]
    # color_frame comes straight from cv2.VideoCapture, which is BGR (OpenCV's
    # own convention). The training images were saved to disk via cv2.imwrite
    # (which converts to true RGB in the file) and loaded back as RGB by Keras,
    # so we have to match that here or the model sees red/blue-swapped colors.
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, MOUTH_IMG_SIZE)
    crop = crop.astype("float32") / 255.0
    return crop.reshape(1, MOUTH_IMG_SIZE[0], MOUTH_IMG_SIZE[1], 3)


def classify_eye(model, class_names, gray_frame, box):
    x = preprocess_eye(gray_frame, box)
    pred = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(pred))
    return class_names[idx], float(pred[idx])


def classify_mouth(model, class_names, color_frame, face_box):
    x = preprocess_mouth(color_frame, face_box)
    pred = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(pred))
    return class_names[idx], float(pred[idx])


def main():
    eye_model = keras.models.load_model(EYE_MODEL_PATH)
    eye_classes = load_classes(EYE_CLASSES_PATH)
    mouth_model = keras.models.load_model(MOUTH_MODEL_PATH)
    mouth_classes = load_classes(MOUTH_CLASSES_PATH)
    alarm = AlarmPlayer(ALARM_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    score = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        eyes_closed_this_frame = False
        yawn_this_frame = False

        if len(faces) > 0:
            face_box = max(faces, key=lambda b: b[2] * b[3])
            fx, fy, fw, fh = face_box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 1)

            min_eye_size = (max(20, fw // 8), max(20, fh // 8))
            face_gray = gray[fy:fy + fh, fx:fx + fw]
            left_eyes = LEFT_EYE_CASCADE.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=min_eye_size)
            right_eyes = RIGHT_EYE_CASCADE.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=min_eye_size)

            eye_labels = []
            debug_lines = []
            for eye_name, (ex, ey, ew, eh) in list(zip(["L", "R"], list(left_eyes)[:1] + list(right_eyes)[:1])):
                abs_box = (fx + ex, fy + ey, ew, eh)
                label, conf = classify_eye(eye_model, eye_classes, gray, abs_box)
                eye_labels.append(label)
                debug_lines.append(f"{eye_name}-eye: {label} ({conf:.2f})")
                cv2.rectangle(frame, (fx + ex, fy + ey), (fx + ex + ew, fy + ey + eh), (0, 255, 0), 1)

            if eye_labels and all(label == "Closed" for label in eye_labels):
                eyes_closed_this_frame = True

            mouth_label, mouth_conf = classify_mouth(mouth_model, mouth_classes, frame, face_box)
            debug_lines.append(f"mouth: {mouth_label} ({mouth_conf:.2f})")
            if mouth_label == "yawn":
                yawn_this_frame = True

            for i, line in enumerate(debug_lines):
                cv2.putText(frame, line, (10, 110 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            print(" | ".join(debug_lines))

        if eyes_closed_this_frame:
            score += EYE_SCORE_STEP
        elif yawn_this_frame:
            score += YAWN_SCORE_STEP
        else:
            score -= 1

        score = max(SCORE_MIN, min(SCORE_MAX, score))

        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if score > DROWSY_SCORE_THRESHOLD:
            cv2.putText(frame, "DROWSY!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
            alarm.play_if_idle()

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
