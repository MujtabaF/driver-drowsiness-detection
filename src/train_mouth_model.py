"""Trains a small CNN to classify a cropped mouth-region image as yawn or no_yawn.

Expects (produced by preprocess_mouth.py):
    dataset/mouth_cropped/train/yawn/*.jpg
    dataset/mouth_cropped/train/no_yawn/*.jpg
    dataset/mouth_cropped/valid/yawn/*.jpg
    dataset/mouth_cropped/valid/no_yawn/*.jpg
"""
import json
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "mouth_cropped", "train")
VALID_DIR = os.path.join(BASE_DIR, "dataset", "mouth_cropped", "valid")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mouth_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "mouth_classes.json")

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15


def build_datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    )
    valid_ds = keras.utils.image_dataset_from_directory(
        VALID_DIR,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    class_names = train_ds.class_names
    print(f"Classes (index order): {class_names}")

    normalize = layers.Rescaling(1.0 / 255)
    # RandomBrightness assumes its input is in the (0, 255) range by default, so it
    # must run BEFORE Rescaling -- applying it after normalization would treat
    # already-[0,1] pixels as if they were on a 0-255 scale and blow the values up.
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (normalize(augment(x, training=True)), y))
    valid_ds = valid_ds.map(lambda x, y: (normalize(x), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, class_names


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    train_ds, valid_ds, class_names = build_datasets()
    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_PATH)
    with open(CLASSES_PATH, "w") as f:
        json.dump(class_names, f)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Class order: {class_names} (index 0 -> {class_names[0]}, index 1 -> {class_names[1]})")


if __name__ == "__main__":
    main()
