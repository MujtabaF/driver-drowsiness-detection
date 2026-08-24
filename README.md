# Driver Drowsiness Detection

Real-time driver fatigue detection from a webcam feed, combining two independent cues:

- **Eye state** (open/closed) via a CNN trained on grayscale eye crops
- **Yawning** (mouth open/closed) via a separate CNN trained on mouth-region crops

Face and eye regions are located with OpenCV Haar cascades; both cues feed into a single
rolling fatigue score, and an alarm sound fires once the score crosses a threshold.

## Architecture

Two independent binary CNNs rather than one multi-class model, since eyes and mouth come
from different face regions (different cascades) and are conceptually separate signals
that get combined downstream into one fatigue score:

- `eye_model.keras` — 64x64 grayscale eye crop -> Open / Closed
- `mouth_model.keras` — 64x64 RGB mouth-region crop -> yawn / no_yawn

Both are small CNNs (3x Conv2D+MaxPooling blocks, Dropout, Dense head) trained with
data augmentation (flip/rotation/zoom, plus brightness for the mouth model).

The mouth-region crop is taken as the bottom third of the detected face bounding box
(`y + 2h/3` to `y + h`), rather than a separate mouth cascade — OpenCV doesn't ship a
reliable one, and this keeps training and live inference using the exact same crop logic.

## Setup

```bash
pip install -r requirements.txt
```

### 1. Dataset

Download the "yawn eye dataset new" dataset from Kaggle (search for that name, or
"Driver Drowsiness Dataset"). It ships as `archive.zip` with this structure:
`dataset_new/{train,test}/{Open,Closed,yawn,no_yawn}/`.

Extract eyes directly (already tightly cropped) and rename `test` -> `valid`:

```bash
unzip -j archive.zip 'dataset_new/train/Open/*'   -d dataset/eyes/train/Open
unzip -j archive.zip 'dataset_new/train/Closed/*' -d dataset/eyes/train/Closed
unzip -j archive.zip 'dataset_new/test/Open/*'    -d dataset/eyes/valid/Open
unzip -j archive.zip 'dataset_new/test/Closed/*'  -d dataset/eyes/valid/Closed

unzip -j archive.zip 'dataset_new/train/yawn/*'    -d dataset/mouth/train/yawn
unzip -j archive.zip 'dataset_new/train/no_yawn/*' -d dataset/mouth/train/no_yawn
unzip -j archive.zip 'dataset_new/test/yawn/*'     -d dataset/mouth/valid/yawn
unzip -j archive.zip 'dataset_new/test/no_yawn/*'  -d dataset/mouth/valid/no_yawn
```

The `yawn`/`no_yawn` images are full face frames, not mouth crops, so they need
preprocessing before training (crops the bottom-third-of-face mouth region using face
detection, matching what happens at inference time):

```bash
python3 src/preprocess_mouth.py
```

### 2. Train both models

```bash
python3 src/train_eye_model.py
python3 src/train_mouth_model.py
```

Saves `models/eye_model.keras` + `models/eye_classes.json` and
`models/mouth_model.keras` + `models/mouth_classes.json`.

### 3. Generate the alarm sound

```bash
python3 src/generate_alarm.py
```

Synthesizes `alarm.wav` (a two-tone alert) so the project has no external audio
dependency.

### 4. Run it

```bash
python3 src/detect.py
```

Opens a webcam window with live face/eye boxes, per-cue predictions with confidence
printed to the console, a running fatigue score, and a red-border + alarm once the
score passes threshold. Press `q` to quit.

## Key engineering decisions (and bugs caught along the way)

- **Face-detect + bottom-third crop for mouth, instead of a dedicated mouth cascade** —
  OpenCV's bundled `haarcascade_smile.xml` is tuned for smile shape, not open-mouth
  yawns, and is unreliable. Reusing the already-detected face box keeps one consistent
  pipeline and guarantees training/inference use identical crop logic.
- **`minSize` filtering on Haar cascade detections** — loosening `scaleFactor`/
  `minNeighbors` to reduce missed detections also let through spurious tiny "face"
  boxes (as small as 24px) that don't correspond to a real face. Left unfiltered,
  these silently corrupted training data (garbage crops with correct folder labels)
  and were traced by inspecting the actual detected box-size distribution rather than
  assuming the miss/skip counts alone told the full story.
- **Augmentation ordering bug** — `RandomBrightness` defaults to assuming `[0, 255]`
  input, but was being applied after `Rescaling(1/255)` had already normalized to
  `[0, 1]`, blowing pixel values up to ~18 and collapsing the mouth model to
  constant-class predictions (~50% val accuracy, identical every epoch). Fixed by
  running brightness augmentation before normalization.
- **BGR vs RGB channel-order mismatch at inference** — training images were written
  to disk via `cv2.imwrite` (converts to true RGB in the file) and read back as RGB
  by Keras, but live webcam frames from `cv2.VideoCapture` are BGR. Feeding raw BGR
  frames to the mouth model caused it to confidently (90-100%) misclassify almost
  every frame as "yawn." Root-caused by adding per-frame confidence logging rather
  than trusting the aggregate alarm output, then fixed with an explicit
  `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` before inference.
