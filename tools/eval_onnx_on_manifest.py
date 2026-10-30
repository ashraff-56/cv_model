"""Re-score the test split through the deployed path (ONNX bundle + original image files).

Training-time evaluation reads resized cache copies; this checks that the shipped inference path on
the original files gives the same accuracy.

    python tools/eval_onnx_on_manifest.py --model-dir artifacts/efficientnet_b0 --out reports/metrics/onnx_test_efficientnet_b0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from smartbite_cv.config import load_config, resolve_path  # noqa: E402
from smartbite_cv.metrics import reliability, summarize  # noqa: E402
from smartbite_cv.predict import FoodRecognizer  # noqa: E402
from smartbite_cv.prepare import dataset_roots, read_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    cfg = load_config(ROOT / "configs" / "data.yaml")
    roots = dataset_roots(cfg, "local")
    rows = [r for r in read_manifest(resolve_path(cfg["manifest"])) if r["split"] == args.split]
    rec = FoodRecognizer(args.model_dir)
    probs, ys = [], []
    for i, r in enumerate(rows):
        probs.append(rec.probabilities(roots[r["dataset"]] / r["relpath"]))
        ys.append(rec.classes.index(r["label"]))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    p, y = np.stack(probs), np.asarray(ys)
    s = summarize(p, y, rec.classes)
    out = {"model": rec.meta["modelName"], "split": args.split, "n": len(y), "top1": s["top1"], "top3": s["top3"],
           "macro_f1": s["macro_f1"], "ece": reliability(p, y)["ece"], "input": "original files via ONNX bundle"}
    print(json.dumps(out, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
