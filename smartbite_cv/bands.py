"""Confidence bands that drive the app's verification screen. Nothing is ever logged automatically.

high   : top guess pre-selected; the user still has to confirm it
medium : "Is it one of these?" with the top 3
low    : "Couldn't recognise it, search manually" (top 3 still shown as hints); also used when the
         model's best guess is `not_food`

Thresholds are chosen on the VALIDATION set from the coverage-accuracy curves of the calibrated
probabilities, then frozen in a YAML file; the test set only reports how they behave.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import yaml

from .metrics import coverage_curve, topk_correct

NOT_FOOD = "not_food"


@dataclass
class BandThresholds:
    high: float                 # confidence >= high -> "high"
    low: float                  # confidence <  low  -> "low"; in between -> "medium"
    high_top1_target: float
    low_band_max_top3: float
    chosen_on: str = "val"

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Confidence-band thresholds on calibrated probability (chosen on the validation set).\n")
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @classmethod
    def load(cls, path) -> "BandThresholds":
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


def choose_thresholds(probs: np.ndarray, y: np.ndarray, high_top1_target: float = 0.95,
                      low_band_max_top3: float = 0.5, min_count: int = 30) -> BandThresholds:
    """
    high: the smallest confidence t such that predictions with conf >= t are top-1 correct at
          least `high_top1_target` of the time (with >= min_count samples). If unreachable, 1.0
          (the high band is never used).
    low:  the largest t such that, among predictions with conf < t, the true dish is in the top 3
          at most `low_band_max_top3` of the time (>= min_count samples), i.e. below it the
          suggestions are more often wrong than right and manual search is the better default.
          If no such t exists, 0.0 (the low band is only used for `not_food`).
    """
    curve = coverage_curve(probs, y, k=1)
    ok = (curve["accuracy"] >= high_top1_target) & (curve["count"] >= min_count)
    high = float(curve["threshold"][ok].min()) if ok.any() else 1.0

    conf = probs.max(1)
    top3 = topk_correct(probs, y, 3).astype(float)
    order = np.argsort(conf, kind="stable")              # ascending confidence
    conf_sorted, cum = conf[order], np.cumsum(top3[order])
    n_below = np.arange(1, len(y) + 1)
    rate_below = cum / n_below                           # top-3 accuracy of the k least-confident
    low = 0.0
    valid = (n_below >= min_count) & (rate_below <= low_band_max_top3)
    if valid.any():
        k = int(np.nonzero(valid)[0].max())
        low = float(np.nextafter(conf_sorted[k], 1.0))   # strictly above the k-th lowest confidence
    low = min(low, high)
    return BandThresholds(high=high, low=low, high_top1_target=high_top1_target,
                          low_band_max_top3=low_band_max_top3)


def assign_band(confidence: float, top_label: str, t: BandThresholds) -> tuple[str, str]:
    """(band, reason) for one prediction."""
    if top_label == NOT_FOOD:
        return "low", "not_food"
    if confidence >= t.high:
        return "high", "confident"
    if confidence >= t.low:
        return "medium", "uncertain"
    return "low", "low_confidence"


def band_report(probs: np.ndarray, y: np.ndarray, classes: list[str], t: BandThresholds) -> dict:
    """How the frozen thresholds behave on a labelled set (fraction per band, accuracy per band)."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    top1 = pred == y
    top3 = topk_correct(probs, y, 3)
    bands = np.array([assign_band(float(c), classes[p], t)[0] for c, p in zip(conf, pred)])
    out = {}
    for b in ("high", "medium", "low"):
        m = bands == b
        out[b] = {"fraction": float(m.mean()), "count": int(m.sum()),
                  "top1": float(top1[m].mean()) if m.any() else None,
                  "top3": float(top3[m].mean()) if m.any() else None}
    return out
