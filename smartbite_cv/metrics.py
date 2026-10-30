"""Classification, calibration and selective-prediction metrics (numpy only, no torch)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def topk_correct(probs: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return (topk == y[:, None]).any(axis=1)


def per_class_prf(y: np.ndarray, pred: np.ndarray, n_classes: int) -> dict[str, np.ndarray]:
    cm = confusion_matrix(y, pred, n_classes)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    predicted = cm.sum(0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support.astype(int)}


def confusion_matrix(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return cm


def worst_confused_pairs(cm: np.ndarray, classes: list[str], top: int = 15) -> list[dict]:
    """Unordered class pairs ranked by total confusions in both directions."""
    pairs = []
    n = len(classes)
    for i in range(n):
        for j in range(i + 1, n):
            total = int(cm[i, j] + cm[j, i])
            if total:
                pairs.append({"a": classes[i], "b": classes[j], "a_as_b": int(cm[i, j]), "b_as_a": int(cm[j, i]),
                              "total": total})
    return sorted(pairs, key=lambda p: -p["total"])[:top]


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    return float(-log_softmax(logits / temperature, axis=1)[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar T > 0 minimising validation NLL."""
    res = minimize_scalar(lambda log_t: nll(logits, y, float(np.exp(log_t))), bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def calibrate(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits / temperature, axis=1)


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    """Expected calibration error with equal-width confidence bins, plus per-bin data for plots."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        acc = float(correct[mask].mean()) if count else 0.0
        avg_conf = float(conf[mask].mean()) if count else 0.0
        ece += count / len(y) * abs(acc - avg_conf)
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": count, "accuracy": acc,
                     "confidence": avg_conf})
    return {"ece": float(ece), "bins": bins}


def coverage_curve(probs: np.ndarray, y: np.ndarray, k: int = 1) -> dict:
    """Accuracy on the subset with confidence >= t, for every distinct t (selective prediction)."""
    conf = probs.max(1)
    correct = topk_correct(probs, y, k).astype(float)
    order = np.argsort(-conf, kind="stable")
    conf_sorted = conf[order]
    cum_correct = np.cumsum(correct[order])
    n_kept = np.arange(1, len(y) + 1)
    # keep the last index of each run of equal confidences so ties are kept together
    last = np.r_[conf_sorted[1:] != conf_sorted[:-1], True]
    return {"threshold": conf_sorted[last], "coverage": n_kept[last] / len(y),
            "accuracy": cum_correct[last] / n_kept[last], "count": n_kept[last]}


def summarize(probs: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    pred = probs.argmax(1)
    prf = per_class_prf(y, pred, len(classes))
    cm = confusion_matrix(y, pred, len(classes))
    present = prf["support"] > 0
    return {
        "n": int(len(y)),
        "top1": float((pred == y).mean()),
        "top3": float(topk_correct(probs, y, 3).mean()),
        "macro_f1": float(prf["f1"][present].mean()),
        "macro_precision": float(prf["precision"][present].mean()),
        "macro_recall": float(prf["recall"][present].mean()),
        "per_class": {c: {"precision": float(prf["precision"][i]), "recall": float(prf["recall"][i]),
                          "f1": float(prf["f1"][i]), "support": int(prf["support"][i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": cm.tolist(),
        "worst_confused_pairs": worst_confused_pairs(cm, classes),
    }


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean of 0/1 outcomes."""
    rng = np.random.default_rng(seed)
    means = rng.choice(correct.astype(float), size=(n_boot, len(correct)), replace=True).mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
