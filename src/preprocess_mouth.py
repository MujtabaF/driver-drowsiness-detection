"""Crops the bottom-third-of-face 'mouth region' out of the raw mouth/yawn dataset,
so the images we train on match what we'll feed the model at live inference time
(face-detect -> crop mouth region -> classify).

Input:  dataset/mouth/{train,valid}/{yawn,no_yawn}/*.jpg   (full face/frame images)
Output: dataset/mouth_cropped/{train,valid}/{yawn,no_yawn}/*.jpg   (mouth-region crops)
"""
import os
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "dataset", "mouth")
DST_DIR = os.path.join(BASE_DIR, "dataset", "mouth_cropped")

SPLITS = ["train", "valid"]
CLASSES = ["yawn", "no_yawn"]

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def crop_mouth_region(img, face_box):
    """Given a BGR image and a (x, y, w, h) face box, return the cropped
    bottom-third-of-face region as a BGR image."""
    x, y, w, h = face_box
    mouth_top = y + int(h * 2 / 3)
    return img[mouth_top:y + h, x:x + w]


def detect_largest_face(gray_img, face_cascade):
    """Run face_cascade.detectMultiScale on gray_img and return the single
    largest (x, y, w, h) box, or None if no face was found."""
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=3, minSize=(100, 100))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def process_folder(face_cascade, split, class_name):
    src_folder = os.path.join(SRC_DIR, split, class_name)
    dst_folder = os.path.join(DST_DIR, split, class_name)
    os.makedirs(dst_folder, exist_ok=True)

    processed, skipped = 0, 0
    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)
        img = cv2.imread(src_path)
        if img is None:
            skipped += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_box = detect_largest_face(gray, face_cascade)
        if face_box is None:
            skipped += 1
            continue

        mouth_crop = crop_mouth_region(img, face_box)
        cv2.imwrite(os.path.join(dst_folder, filename), mouth_crop)
        processed += 1

    print(f"{split}/{class_name}: processed={processed} skipped={skipped}")


def main():
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    for split in SPLITS:
        for class_name in CLASSES:
            process_folder(face_cascade, split, class_name)


if __name__ == "__main__":
    main()
