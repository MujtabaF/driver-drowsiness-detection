"""Evaluates the trained eye and mouth models on their validation sets.

Reports accuracy, per-class precision / recall / F1, macro F1, ROC-AUC and a
confusion matrix for each model, writes everything to results/metrics.json, and
renders charts (confusion matrices, ROC curves, metric comparison) to results/.

Note: the validation set is also what EarlyStopping / ModelCheckpoint used to
pick the best epoch, so these numbers are slightly optimistic -- there is no
separate held-out test set.
"""
import json
import os
import numpy as np
from tensorflow import keras

from plot_metrics import plot_all

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")

IMG_SIZE = (64, 64)

# (name, validation dir, color mode, positive class -- the "drowsy" signal)
MODELS = [
    ("eye", os.path.join(BASE_DIR, "dataset", "eyes", "valid"), "grayscale", "Closed"),
    ("mouth", os.path.join(BASE_DIR, "dataset", "mouth_cropped", "valid"), "rgb", "yawn"),
]


def load_validation(valid_dir, color_mode, class_names):
    # class_names pins the label order to what the model was trained with, and
    # Rescaling(1/255) is applied manually to match train_*_model.py / detect.py.
    ds = keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode=color_mode,
        image_size=IMG_SIZE,
        batch_size=32,
        shuffle=False,
    )
    images, labels = [], []
    for x, y in ds:
        images.append(x.numpy() / 255.0)
        labels.append(y.numpy())
    return np.concatenate(images), np.concatenate(labels)


def roc_curve(y_true, scores):
    # Sweep the threshold down through every distinct score; returns (fpr, tpr) from (0,0) to (1,1).
    order = np.argsort(-scores, kind="mergesort")
    y_sorted, s_sorted = y_true[order], scores[order]
    last_of_threshold = np.r_[np.where(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tps = np.cumsum(y_sorted)[last_of_threshold]
    fps = (last_of_threshold + 1) - tps
    tpr = np.r_[0, tps / y_true.sum()]
    fpr = np.r_[0, fps / (len(y_true) - y_true.sum())]
    return fpr.tolist(), tpr.tolist()


def roc_auc(y_true, scores):
    # Mann-Whitney U formulation: probability a random positive outranks a random negative.
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def compute_metrics(y_true, y_pred, probs, class_names, positive):
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[i, :].sum()),
        }

    pos_idx = class_names.index(positive)
    y_pos = (y_true == pos_idx).astype(int)
    fpr, tpr = roc_curve(y_pos, probs[:, pos_idx])
    return {
        "samples": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(np.mean([c["f1"] for c in per_class.values()])),
        "positive_class": positive,
        "roc_auc": roc_auc(y_pos, probs[:, pos_idx]),
        "roc_curve": {"fpr": fpr, "tpr": tpr},
        "per_class": per_class,
        "confusion_matrix": {"labels": class_names, "rows_true_cols_pred": cm.tolist()},
    }


def print_report(name, m):
    labels = m["confusion_matrix"]["labels"]
    cm = m["confusion_matrix"]["rows_true_cols_pred"]
    width = max(len(l) for l in labels) + 2

    print(f"\n=== {name} model ({m['samples']} validation images) ===")
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Macro F1 : {m['macro_f1']:.4f}")
    print(f"ROC-AUC  : {m['roc_auc']:.4f}  (positive class = {m['positive_class']})")

    print(f"\n{'class':<{width}}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for cls, c in m["per_class"].items():
        print(f"{cls:<{width}}{c['precision']:>10.4f}{c['recall']:>10.4f}{c['f1']:>10.4f}{c['support']:>10}")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(" " * width + "".join(f"{l:>{width}}" for l in labels))
    for label, row in zip(labels, cm):
        print(f"{label:<{width}}" + "".join(f"{v:>{width}}" for v in row))


def main():
    results = {}
    for name, valid_dir, color_mode, positive in MODELS:
        model = keras.models.load_model(os.path.join(MODELS_DIR, f"{name}_model.keras"))
        with open(os.path.join(MODELS_DIR, f"{name}_classes.json")) as f:
            class_names = json.load(f)

        x, y_true = load_validation(valid_dir, color_mode, class_names)
        probs = model.predict(x, verbose=0)
        y_pred = probs.argmax(axis=1)

        results[name] = compute_metrics(y_true, y_pred, probs, class_names, positive)
        print_report(name, results[name])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {METRICS_PATH}")

    for path in plot_all(results, RESULTS_DIR):
        print(f"Saved chart to {path}")


if __name__ == "__main__":
    main()
