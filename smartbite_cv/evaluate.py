"""Final evaluation of a trained run. Uses the test split exactly once.

 * logits on val and test with the best checkpoint
 * temperature scaling fitted on val; ECE/NLL before and after (val and test)
 * confidence-band thresholds chosen on val (coverage-accuracy curves), reported on test
 * test metrics: top-1/top-3 (+ bootstrap 95% CI), macro-F1, per-class P/R/F1, confusion matrix,
   worst confused pairs, failure-case gallery

    python -m smartbite_cv.evaluate --run runs/efficientnet_b0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from .bands import BandThresholds, band_report, choose_thresholds
from .config import load_config, resolve_path
from .metrics import (bootstrap_ci, calibrate, coverage_curve, fit_temperature, nll, reliability, summarize,
                      topk_correct)
from .modeling import infer_logits, load_trained, make_loader, split_rows
from . import plots


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained run on val (calibration) and test")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--profile", choices=["local", "kaggle"], default="local")
    ap.add_argument("--out", type=Path, help="default: <run>/eval")
    ap.add_argument("--high-top1-target", type=float, default=0.95)
    ap.add_argument("--low-band-max-top3", type=float, default=0.5)
    args = ap.parse_args(argv)

    run = args.run
    out = args.out or run / "eval"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    with open(run / "config_resolved.json", encoding="utf-8") as f:
        cfg = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_trained(run / "best.pt", device)
    splits, classes, roots, data_cfg = split_rows(cfg, args.profile)
    assert classes == meta["classes"], "class list changed since training"
    cache_dir = cfg["data"].get("cache_dir")
    cache_dir = resolve_path(cache_dir) if cache_dir else None

    logits, labels, loaders = {}, {}, {}
    for split in ("val", "test"):
        loaders[split] = make_loader(splits[split], classes, roots, data_cfg, cfg, train=False, cache_dir=cache_dir)
        logits[split], labels[split] = infer_logits(model, loaders[split], device, amp=False)
    np.savez_compressed(out / "logits.npz", val_logits=logits["val"], val_y=labels["val"],
                        test_logits=logits["test"], test_y=labels["test"], classes=np.array(classes))

    temperature = fit_temperature(logits["val"], labels["val"])
    probs = {s: calibrate(logits[s], temperature) for s in logits}
    raw = {s: softmax(logits[s], axis=1) for s in logits}
    calib = {s: {"nll_before": nll(logits[s], labels[s]), "nll_after": nll(logits[s], labels[s], temperature),
                 "ece_before": reliability(raw[s], labels[s])["ece"],
                 "ece_after": reliability(probs[s], labels[s])["ece"]} for s in logits}

    bands = choose_thresholds(probs["val"], labels["val"], args.high_top1_target, args.low_band_max_top3)
    bands.save(out / "bands.yaml")
    with open(out / "calibration.json", "w", encoding="utf-8") as f:
        json.dump({"temperature": temperature, **calib}, f, indent=2)

    y, p = labels["test"], probs["test"]
    test = summarize(p, y, classes)
    test["top1_ci95"] = bootstrap_ci(p.argmax(1) == y)
    test["top3_ci95"] = bootstrap_ci(topk_correct(p, y, 3))
    val = summarize(probs["val"], labels["val"], classes)
    curves = {k: coverage_curve(p, y, k) for k in (1, 3)}
    metrics = {
        "model": meta["model_cfg"]["name"], "classes": classes, "temperature": temperature,
        "calibration": calib, "bands": bands.__dict__, "bands_on_test": band_report(p, y, classes, bands),
        "val": {k: val[k] for k in ("n", "top1", "top3", "macro_f1")},
        "test": test,
        "coverage_at": {f"{t:.2f}": _coverage_at(curves[1], t) for t in (0.3, 0.5, 0.7, 0.9)},
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    name = meta["model_cfg"]["name"]
    plots.confusion(np.array(test["confusion_matrix"]), classes, out / "figures" / "confusion_matrix.png", name)
    plots.reliability_diagram(reliability(raw["test"], y), reliability(p, y), out / "figures" / "reliability.png",
                              name)
    plots.coverage(curves, bands, out / "figures" / "coverage_accuracy.png", name)
    plots.per_class_f1(test["per_class"], out / "figures" / "per_class_f1.png", name)
    failures = _failure_cases(p, y, loaders["test"], classes)
    plots.gallery(failures, out / "figures" / "failure_cases.png", f"{name}: most confident test errors")
    with open(out / "failure_cases.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)
    print(json.dumps({k: metrics["test"][k] for k in ("top1", "top3", "macro_f1")}
                     | {"ece_after": calib["test"]["ece_after"], "T": temperature}, indent=2))


def _coverage_at(curve: dict, threshold: float) -> dict:
    mask = curve["threshold"] >= threshold
    if not mask.any():
        return {"coverage": 0.0, "accuracy": None}
    i = int(np.nonzero(mask)[0].max())
    return {"coverage": float(curve["coverage"][i]), "accuracy": float(curve["accuracy"][i])}


def _failure_cases(probs: np.ndarray, y: np.ndarray, loader, classes: list[str], n: int = 24) -> list[dict]:
    pred = probs.argmax(1)
    wrong = np.nonzero(pred != y)[0]
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:n]
    ds = loader.dataset
    return [{"path": str(ds.path(int(i))), "dataset": ds.rows[int(i)]["dataset"],
             "relpath": ds.rows[int(i)]["relpath"], "true": classes[y[i]], "pred": classes[pred[i]],
             "confidence": float(probs[i].max()),
             "top3": [classes[j] for j in np.argsort(-probs[i])[:3]]} for i in wrong]


if __name__ == "__main__":
    main()
