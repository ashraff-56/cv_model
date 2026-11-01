"""Evaluate the deployed ONNX bundle on a folder of labelled photos (e.g. team phone photos).

Layout: <folder>/<class_slug>/*.jpg  (slugs from configs/classes.yaml). Uses the exact inference
path the app will use (predict.FoodRecognizer), including EXIF rotation and calibration.

    python -m smartbite_cv.eval_folder --model-dir artifacts/efficientnet_b0 --folder eval_phone_photos
    python -m smartbite_cv.eval_folder --model-dir artifacts/efficientnet_b0_v2 --folder data/external/wikipedia         --exclude data/external/wikipedia/excluded.csv     # duplicates of training images + rejected labels
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .bands import band_report
from .metrics import bootstrap_ci, reliability, summarize, topk_correct
from .predict import FoodRecognizer

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--folder", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--exclude", type=Path, help="CSV with a `file` column (paths relative to --folder) to skip")
    args = ap.parse_args(argv)
    excluded = set()
    if args.exclude:
        with open(args.exclude, encoding="utf-8") as f:
            excluded = {r["file"] for r in csv.DictReader(f)}

    rec = FoodRecognizer(args.model_dir)
    probs, ys, skipped = [], [], []
    for cls_dir in sorted(p for p in args.folder.iterdir() if p.is_dir()):
        if cls_dir.name not in rec.classes:
            skipped.append(f"{cls_dir.name}/ (not a model class)")
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in EXTS and f.relative_to(args.folder).as_posix() not in excluded:
                try:
                    probs.append(rec.probabilities(f))
                    ys.append(rec.classes.index(cls_dir.name))
                except Exception as exc:
                    skipped.append(f"{f}: {exc}")
    if not probs:
        raise SystemExit(f"no labelled images found under {args.folder}")
    p, y = np.stack(probs), np.asarray(ys)
    s = summarize(p, y, rec.classes)
    result = {"n": len(y), "top1": s["top1"], "top1_ci95": bootstrap_ci(p.argmax(1) == y),
              "top3": s["top3"], "top3_ci95": bootstrap_ci(topk_correct(p, y, 3)),
              "macro_f1_over_present_classes": s["macro_f1"], "ece": reliability(p, y)["ece"],
              "bands": band_report(p, y, rec.classes, rec.bands),
              "per_class": {k: v for k, v in s["per_class"].items() if v["support"]},
              "worst_confused_pairs": s["worst_confused_pairs"], "excluded": len(excluded), "skipped": skipped}
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
