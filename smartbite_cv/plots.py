"""Report figures (PNG). Matplotlib with the Agg backend so it runs headless on Kaggle."""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .imageio import load_rgb  # noqa: E402

BLUE, ORANGE, GREY = "#2a6fdb", "#e8833a", "#8a8a8a"


def confusion(cm: np.ndarray, classes: list[str], path, title: str) -> None:
    norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(0.32 * n + 3, 0.32 * n + 2.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(n), classes, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(n):
        for j in range(n):
            if cm[i, j] and (i == j or norm[i, j] >= 0.05):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=5,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, label="row-normalised rate")
    ax.set_title(f"Confusion matrix (test): {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def reliability_diagram(before: dict, after: dict, path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, rel, name in ((axes[0], before, "before"), (axes[1], after, "after temperature scaling")):
        centers = [(b["lo"] + b["hi"]) / 2 for b in rel["bins"]]
        acc = [b["accuracy"] if b["count"] else 0 for b in rel["bins"]]
        ax.bar(centers, acc, width=1 / len(centers), edgecolor="white", color=BLUE, label="accuracy")
        ax.plot([0, 1], [0, 1], "--", color=GREY, label="perfect calibration")
        ax.set_xlabel("confidence")
        ax.set_title(f"{name} (ECE {rel['ece']:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Reliability (test): {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def coverage(curves: dict, bands, path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, color in ((1, BLUE), (3, ORANGE)):
        c = curves[k]
        ax.plot(c["coverage"], c["accuracy"], color=color, label=f"top-{k} accuracy on kept images")
    for name, t in (("high", bands.high), ("low", bands.low)):
        c = curves[1]
        if 0 < t < 1:
            mask = c["threshold"] >= t
            if mask.any():
                cov = c["coverage"][np.nonzero(mask)[0].max()]
                ax.axvline(cov, color=GREY, ls=":")
                ax.text(cov, 0.02, f" {name} t={t:.2f}", rotation=90, fontsize=8, va="bottom")
    ax.set_xlabel("coverage (fraction of images with confidence >= threshold)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.01)
    ax.set_xlim(0, 1)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"Coverage vs accuracy (test): {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def per_class_f1(per_class: dict, path, title: str) -> None:
    items = sorted(per_class.items(), key=lambda kv: kv[1]["f1"])
    fig, ax = plt.subplots(figsize=(6.5, 0.22 * len(items) + 1.2))
    ax.barh([k for k, _ in items], [v["f1"] for _, v in items], color=BLUE)
    for i, (_, v) in enumerate(items):
        ax.text(v["f1"] + 0.01, i, f"{v['f1']:.2f} (n={v['support']})", va="center", fontsize=7)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 (test)")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(f"Per-class F1: {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gallery(cases: list[dict], path, title: str, cols: int = 6) -> None:
    if not cases:
        return
    rows = (len(cases) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, case in zip(np.atleast_1d(axes).ravel(), cases):
        try:
            img = load_rgb(case["path"])
            img.thumbnail((256, 256))
            ax.imshow(img)
        except Exception:
            pass
        label = f"true: {case['true']}\npred: {case['pred']} ({case['confidence']:.2f})"
        ax.set_title("\n".join(textwrap.wrap(label, 30, replace_whitespace=False)), fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def comparison_bars(rows: list[dict], path) -> None:
    metrics = [("top1", "top-1"), ("top3", "top-3"), ("macro_f1", "macro-F1")]
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(rows))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, (row, color) in enumerate(zip(rows, (BLUE, ORANGE, GREY))):
        vals = [row[m] for m, _ in metrics]
        ax.bar(x + i * width, vals, width, label=row["name"], color=color)
        for xi, v in zip(x + i * width, vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x + width * (len(rows) - 1) / 2, [m[1] for m in metrics])
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8)
    ax.set_title("Test-set comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
