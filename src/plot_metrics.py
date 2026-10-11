"""Renders evaluation charts from the metrics dict produced by evaluate.py.

Can also be run on its own to re-draw charts from an existing results/metrics.json
without re-running the models:

    python src/plot_metrics.py
"""
import json
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Categorical slots in fixed order: series 1 = eye model, series 2 = mouth model.
SERIES = {"eye": "#2a78d6", "mouth": "#eb6834"}
MODEL_TITLES = {"eye": "Eye model (Open / Closed)", "mouth": "Mouth model (yawn / no_yawn)"}
# Sequential ramp for confusion-matrix cell counts: one hue, light -> dark.
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#eef4fc", "#9fc3ef", "#2a78d6", "#123f78"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 11,
    "text.color": TEXT_PRIMARY,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.edgecolor": GRID,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_confusion_matrices(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.4))
    for ax, (name, m) in zip(axes, results.items()):
        labels = m["confusion_matrix"]["labels"]
        cm = m["confusion_matrix"]["rows_true_cols_pred"]
        vmax = max(max(row) for row in cm)
        ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=vmax)
        for i, row in enumerate(cm):
            total = sum(row)
            for j, v in enumerate(row):
                ink = SURFACE if v > vmax * 0.55 else TEXT_PRIMARY
                ax.text(j, i, f"{v}\n{v / total:.1%}", ha="center", va="center", color=ink, fontsize=12)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{MODEL_TITLES[name]}\naccuracy {m['accuracy']:.1%}", loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(results, path):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle="--", label="Random guess (AUC 0.50)")
    for name, m in results.items():
        roc = m["roc_curve"]
        ax.plot(roc["fpr"], roc["tpr"], color=SERIES[name], linewidth=2,
                label=f"{MODEL_TITLES[name]}: AUC {m['roc_auc']:.3f}")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate (false alarms)")
    ax.set_ylabel("True positive rate (drowsy signs caught)")
    ax.set_title("ROC curves (positive class = Closed / yawn)", loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(results, path):
    # One row of bars per metric, models side by side; drowsy-class metrics are
    # the ones that matter most for an alarm, so they're shown explicitly.
    def metric_rows(m):
        pos = m["per_class"][m["positive_class"]]
        return {
            "Accuracy": m["accuracy"],
            "Macro F1": m["macro_f1"],
            "ROC-AUC": m["roc_auc"],
            "Drowsy precision": pos["precision"],
            "Drowsy recall": pos["recall"],
            "Drowsy F1": pos["f1"],
        }

    rows = {name: metric_rows(m) for name, m in results.items()}
    metrics = list(next(iter(rows.values())).keys())
    names = list(rows.keys())
    bar_w = 0.38
    gap = 0.02

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for k, name in enumerate(names):
        xs = [i + (k - (len(names) - 1) / 2) * (bar_w + gap) for i in range(len(metrics))]
        values = [rows[name][metric] for metric in metrics]
        bars = ax.bar(xs, values, width=bar_w, color=SERIES[name], label=MODEL_TITLES[name])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.1%}",
                    ha="center", va="bottom", fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_xticks(range(len(metrics)), metrics)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Validation metrics by model", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(names), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = [
        os.path.join(out_dir, "confusion_matrices.png"),
        os.path.join(out_dir, "roc_curves.png"),
        os.path.join(out_dir, "metrics_comparison.png"),
    ]
    plot_confusion_matrices(results, paths[0])
    plot_roc_curves(results, paths[1])
    plot_metric_comparison(results, paths[2])
    return paths


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    with open(os.path.join(results_dir, "metrics.json")) as f:
        for p in plot_all(json.load(f), results_dir):
            print(f"Saved chart to {p}")
