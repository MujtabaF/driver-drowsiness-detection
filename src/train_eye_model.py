"""Trains a small CNN to classify a cropped eye image as Open or Closed.

Expects:
    dataset/eyes/train/Open/*.jpg
    dataset/eyes/train/Closed/*.jpg
    dataset/eyes/valid/Open/*.jpg
    dataset/eyes/valid/Closed/*.jpg
"""
import json
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "eyes", "train")
VALID_DIR = os.path.join(BASE_DIR, "dataset", "eyes", "valid")
MODEL_PATH = os.path.join(BASE_DIR, "models", "eye_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "eye_classes.json")

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15


def build_datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    )
    valid_ds = keras.utils.image_dataset_from_directory(
        VALID_DIR,
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"Classes (index order): {class_names}")

    normalize = layers.Rescaling(1.0 / 255)
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (augment(normalize(x), training=True), y))
    valid_ds = valid_ds.map(lambda x, y: (normalize(x), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, class_names


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 1)),
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
