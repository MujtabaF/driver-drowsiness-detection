# Understanding This Project

This is your reference for explaining the project in an interview. It's organized so
you can skim the headers for a quick refresher, or read a section deeply if you get
asked to go further on something specific.

## 1. The problem, in one sentence

Detect driver fatigue in real time from a webcam by watching two independent physical
cues — eyes closing for too long, and yawning — and sound an alarm before it becomes
dangerous.

If asked "why these two cues": eye closure is the most direct signal of
microsleep, but on its own it's noisy (a normal blink also closes the eyes briefly).
Yawning is a slower, independent physiological signal of fatigue that doesn't have that
false-positive problem, so combining them makes the system more robust than either
alone — one cue papers over the other's weak spot.

## 2. Data pipeline: from a Kaggle zip to trainable folders

The dataset ("yawn eye dataset new" on Kaggle) ships as a zip with this structure:

```
dataset_new/{train,test}/{Open,Closed,yawn,no_yawn}/
```

Two classes (`Open`, `Closed`) were already tight crops of just an eye — pulled
straight out with `unzip -j` (junk the internal folder paths) into
`dataset/eyes/{train,valid}/{Open,Closed}` (renaming `test`→`valid`, the more common
term for a held-out evaluation split during training).

The other two classes (`yawn`, `no_yawn`) turned out to be **full 640x480 face
frames**, not mouth crops — discovered by actually opening a few sample images with
PIL and printing their sizes, rather than assuming the dataset matched the eyes'
format. That single discovery is what led to needing a whole separate preprocessing
script (see section 4).

**Key habit to mention in an interview:** at every stage, decisions were made by
inspecting real data (image sizes, a handful of images opened by eye, distributions of
detected box sizes) rather than assuming the tutorial/dataset would "just work." Twice
that inspection caught something that would have silently produced a broken or
underperforming model (see sections 5 and 6).

## 3. Why two separate CNNs instead of one 4-class model

`Open`/`Closed`/`yawn`/`no_yawn` could in principle be lumped into one 4-way
classifier, but this project uses two independent binary models instead:

- They come from **different face regions**, detected by **different cascades**
  (eye cascades vs. the face box directly) — so at inference time you never have a
  single crop that could even be fed to one shared model.
- They're conceptually **separate signals** that get recombined downstream into one
  fatigue score with different weights (a yawn is scored as a stronger single event
  than one closed-eye frame — see `YAWN_SCORE_STEP` vs `EYE_SCORE_STEP` in
  `detect.py`). Keeping them as separate models keeps that scoring logic simple and
  interpretable, instead of trying to decode a combined 4-way softmax into two
  independent yes/no decisions.

## 4. The mouth-crop preprocessing step (`src/preprocess_mouth.py`)

Since the yawn/no_yawn images are full face frames, training directly on them would
mean the model has to learn to ignore everything except mouth shape on its own — a
much harder learning problem for ~1200 images than if we crop to just the relevant
region first.

There's no dedicated "mouth" Haar cascade bundled with OpenCV (only
`haarcascade_smile.xml`, which is tuned specifically for smile curvature, not a
general open/closed mouth — unreliable for yawns). So instead:

1. Run the existing frontal-face cascade on each image (`detectMultiScale` on the
   grayscale version — Haar cascades are a classical, pre-deep-learning detection
   method that works on grayscale intensity patterns).
2. Take the **bottom third** of the detected face box as the mouth region:
   `mouth_top = y + int(h * 2/3)`, crop `img[mouth_top:y+h, x:x+w]`.
3. Save that crop to `dataset/mouth_cropped/...`, preserving the folder-name label.

**Why this matters more than it sounds like it should — train/inference skew.**
This crop logic has to be applied identically at training time and at live
inference time. If training used one cropping method and live detection used
another (or none), the model would see a fundamentally different kind of input live
than it learned on, and accuracy would collapse in deployment even though it looked
fine in evaluation. This exact class of bug bit us twice later (section 6) — it's the
single most important idea to be able to explain clearly in an interview, because
it's a real production ML failure mode, not just a toy-project detail.

## 5. Model architecture and training

Both models are small CNNs of the same shape (`build_model()` in
`train_eye_model.py` / `train_mouth_model.py`):

```
Input -> Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Conv2D(128) -> MaxPool
      -> Dropout(0.25) -> Flatten -> Dense(128) -> Dropout(0.5) -> Dense(2, softmax)
```

- **Conv2D** layers slide a small learned filter across the image to detect local
  patterns (edges, curves) regardless of position — much more parameter-efficient
  than flattening straight to a dense layer, which would have to re-learn "an edge"
  separately for every pixel position.
- **MaxPooling2D** downsamples after each conv block, keeping the strongest signal in
  each window and reducing computation.
- Filter counts increase (32→64→128) so early layers learn simple features and later
  layers combine them into more abstract ones (eye shape, mouth-open shape).
- **Dropout** randomly zeroes neurons during training — necessary here because the
  datasets are small (~600-1200 images per class), so the network can otherwise
  memorize training examples instead of generalizing.
- Output is `Dense(2, softmax)` — a probability distribution over the two classes,
  trained with categorical cross-entropy.

**Data loading & augmentation:** `keras.utils.image_dataset_from_directory` infers
labels from subfolder names and batches/resizes images. Augmentation
(`RandomFlip`, `RandomRotation`, `RandomZoom`, and `RandomBrightness` for the mouth
model) is applied only to the training set, not validation — the point of validation
is to measure real-world generalization, so it must see the same kind of input the
model will see live, without artificial variation.

**Callbacks:** `EarlyStopping(monitor="val_accuracy", patience=4,
restore_best_weights=True)` stops training once validation accuracy stalls for 4
epochs and rolls back to the best epoch's weights (guards against overfitting past
the point of diminishing returns). `ModelCheckpoint` saves the best-performing
version to disk during training.

**Results:** eye model reached ~98.6% validation accuracy; mouth model ~89.7% (after
fixing the bugs in the next section — its first two training attempts were stuck at
~50%, i.e. chance level).

## 6. Three real bugs, and how each was actually found

This is the part most worth rehearsing for an interview — it's the difference
between "I followed a tutorial" and "I built and debugged a system."

**Bug 1 — Haar cascade false positives corrupting training labels.**
The initial mouth-crop preprocessing skipped ~45% of images (face not detected).
Loosening `scaleFactor` (1.1→1.05) and `minNeighbors` (5→3) recovered most of them —
but that tradeoff also let through spurious tiny "face" boxes (as small as 24px) that
don't correspond to real faces. *Found by actually opening some of the resulting
crops* — one was just a blurry sliver of background/clothing. *Fixed* by analyzing
the distribution of detected box sizes across the dataset (real faces clustered
150-260px; bogus ones at 24-32px) and adding `minSize=(100,100)` to reject the
implausible detections rather than picking a threshold blindly.

**Bug 2 — augmentation ordering bug collapsing the mouth model to chance level.**
Even after fixing bug 1, the mouth model's train *and* validation accuracy were both
stuck at ~50% — not just failing to generalize, failing to even fit the training set,
which a 1.1M-parameter CNN should trivially be able to do with ~800 images unless
something was actually broken. *Isolated* by testing the model without augmentation
first (it then trained to 94% train accuracy immediately, proving the architecture
and data were fine) and then testing `RandomBrightness` in isolation, which revealed
pixel values as high as 18 instead of the expected `[0,1]` range. Root cause:
`RandomBrightness` defaults to assuming a `[0,255]` input range, but was being
applied *after* `Rescaling(1/255)` had already normalized to `[0,1]` — so it treated
already-tiny values as if they were on a 0-255 scale and blew them up, wrecking
training (saturated activations, broken gradients). *Fixed* by reordering: run
`RandomBrightness` (and the rest of augmentation) before normalization, not after.

**Bug 3 — BGR vs RGB channel mismatch, live inference only.**
The full pipeline trained fine and looked correct, but live testing showed the mouth
model confidently predicting "yawn" (90-100% confidence) on almost every single
frame, even with the mouth closed — while the eye model was working correctly the
whole time (visible in the console log as normal, occasional blinks). *Diagnosed* by
adding per-frame confidence logging (not just the final alarm state) so it was
possible to see this was a confident wrong answer, not borderline noise, and that it
was specific to one of the two models, not a general failure. Root cause:
`cv2.VideoCapture` frames are in OpenCV's native **BGR** channel order, but the
training images were written to disk via `cv2.imwrite` (which converts to standard
**RGB** for the JPEG file) and loaded back as RGB by Keras — so the live pipeline was
feeding the model red/blue-channel-swapped images it had never seen anything like in
training. *Fixed* with an explicit `cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)` right
before inference. (The eye model was immune to this because grayscale conversion via
`cv2.cvtColor(..., COLOR_BGR2GRAY)` correctly accounts for OpenCV's own channel
convention regardless of BGR vs RGB — there's no analogous "which channel is which"
ambiguity once you've collapsed to a single grayscale value.)

**The common thread across all three:** none of them showed up by reading the code —
they showed up by looking at actual outputs (sample images, accuracy curves, live
confidence values) and treating a suspiciously clean or suspiciously broken number as
a question to answer, not a result to accept.

## 7. The real-time detection loop (`src/detect.py`)

Per frame:

1. Convert to grayscale, run the face cascade (`minSize=(100,100)` again, same
   false-positive guard as training).
2. Within the largest detected face, run left/right eye cascades (each with a
   face-size-relative `minSize` floor for the same reason), classify each detected
   eye crop as Open/Closed.
3. Classify the bottom-third-of-face mouth crop as yawn/no_yawn (with the BGR→RGB
   fix applied).
4. Update a bounded fatigue **score**: +1 if both detected eyes are Closed, +2 if a
   yawn is detected (weighted higher — a single yawn is a stronger signal than one
   closed-eye frame), otherwise -1, clamped to `[0, 30]`.
5. If the score exceeds a threshold (15), draw a red border, show "DROWSY!", and play
   an alarm tone on a background thread (`AlarmPlayer`, via `pygame.mixer`) so audio
   playback never blocks the video loop.

**Why a bounded, decaying score instead of an instant single-frame alarm:** a single
misclassified frame (a long blink, a brief false detection) shouldn't trigger a false
alarm; requiring the score to accumulate over multiple consecutive "trouble" frames
before crossing threshold — and to decay back down during normal frames — makes the
system robust to occasional noise while still reacting within about half a second of
sustained closed eyes.

**Alarm generation:** rather than bundling a third-party audio file,
`generate_alarm.py` synthesizes a two-tone alert tone from scratch with `numpy`
(sine waves + a fade envelope) and writes it as a `.wav` — no external asset
dependency, and it's a small, self-contained example of raw signal generation.

## 8. Questions you should be ready for

- **"Why not use a pretrained model / MobileNet transfer learning?"** — with datasets
  this small (~600-1200 images/class) and the classes being visually simple
  (open/closed eye shape, open/closed mouth shape), a small CNN trained from scratch
  is enough and trains in seconds; transfer learning would be reasonable if pushing
  for higher accuracy or handling more visual variation (glasses, lighting, angle),
  and would be a natural "next step" answer.
- **"How would you improve accuracy/robustness?"** — more training data with more
  lighting/angle/camera diversity (the current dataset is a fixed, curated Kaggle
  set, not necessarily representative of arbitrary webcams); replacing Haar cascades
  (a ~20-year-old classical method) with a modern face-mesh/landmark detector for
  more precise, rotation-robust eye/mouth localization instead of the crude
  bottom-third-of-face heuristic.
- **"What was the hardest part?"** — the three bugs in section 6, especially that two
  of them looked like "the model just isn't good enough" until actually digging into
  *why*, rather than tuning hyperparameters blindly in response to a bad number.
- **"How do you know it actually works, not just that it runs?"** — validation
  accuracy numbers (98.6% eyes, 89.7% mouth) plus a live test protocol: baseline with
  a neutral face confirming low false-positive rate, then deliberate eye-closure and
  yawn tests confirming the score responds and decays correctly.

## 9. Honest limitations (good to volunteer, not just survive being asked)

- Trained on a fixed public dataset, not on this specific setup's camera/lighting —
  real-world accuracy will vary with hardware and conditions.
- Haar cascades are fast but not robust to head rotation, glasses glare, or
  poor lighting; a missed face detection means a full "no signal" gap for that frame.
- The bottom-third-of-face mouth crop is a heuristic, not a proper mouth
  landmark — it degrades if the head tilts or turns away from the camera.
- No temporal smoothing beyond the simple accumulating score — a proper deployment
  might use a short rolling window of predictions rather than frame-by-frame state.
