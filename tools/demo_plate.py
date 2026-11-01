"""Run the full photo -> dish -> plate -> grams -> nutrients pipeline on some photos and show each step.

There is no weighed ground truth for these photos: this is a visual check of what the app would show
(and a CPU timing), not an accuracy measurement. Accuracy numbers come from smartbite_cv.plate_portion.

    python tools/demo_plate.py photo1.jpg photo2.jpg --out reports/plate/demo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smartbite_cv.imageio import load_rgb  # noqa: E402
from smartbite_cv.nutrition import CsvNutritionProvider  # noqa: E402
from smartbite_cv.plate import PlateDetector, draw, plate_only_view  # noqa: E402
from smartbite_cv.predict import FoodRecognizer  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--model-dir", default="artifacts/efficientnet_b0_v2")
    ap.add_argument("--portion-dir", default="artifacts/portion")
    ap.add_argument("--plate-portion-dir", default="artifacts/plate_portion")
    ap.add_argument("--nutrition-csv", default="nutrition/nutrition_v2.csv")
    ap.add_argument("--out", type=Path, default=Path("reports/plate/demo"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    det = PlateDetector(device="cpu")
    provider = CsvNutritionProvider(args.nutrition_csv)
    with_plate = FoodRecognizer(args.model_dir, nutrition=provider, portion_dir=args.portion_dir,
                                plate_portion_dir=args.plate_portion_dir, plate_detector=det)
    old = FoodRecognizer(args.model_dir, nutrition=provider, portion_dir=args.portion_dir)
    results = []
    for p in args.images:
        img = load_rgb(p)
        pred = with_plate.predict(img)
        label = pred["suggestions"][0]["label"]
        t0 = time.perf_counter()
        q = with_plate.quantity_for(img, label)
        seconds = time.perf_counter() - t0
        q_old = old.quantity_for(img, label)
        n = with_plate.nutrition_for(label, q["grams"])
        plate = det.detect(img)   # again, only to draw it
        left = draw(np.asarray(img), plate)
        left = cv2.resize(left, (round(left.shape[1] * 320 / left.shape[0]), 320))
        right = (cv2.cvtColor(plate_only_view(np.asarray(img), plate), cv2.COLOR_RGB2BGR) if plate is not None
                 else np.full((320, 320, 3), 128, np.uint8))
        panel = np.hstack([left, right])
        caption = f"{pred['suggestions'][0]['displayName']}: {q['grams']:g} g, {n['kcal']:.0f} kcal, {n['proteinG']} g protein"
        cv2.putText(panel, caption, (8, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imwrite(str(args.out / f"{p.stem}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 88])
        row = {"photo": p.name, "dish": label, "band": pred["band"], "confidence": pred["suggestions"][0]["confidence"],
               "plate": q.get("plate"), "warning": q.get("warning"), "method": q["method"],
               "photoEstimateG": q.get("photoEstimateG"), "grams": q["grams"], "rangeG": q.get("rangeG"),
               "oldPhotoEstimateG": q_old.get("photoEstimateG"), "typicalServingG": q.get("typicalServingG"),
               "kcal": n["kcal"], "proteinG": n["proteinG"], "carbsG": n["carbsG"], "fatG": n["fatG"],
               "quantitySeconds": round(seconds, 2)}
        results.append(row)
        print(f"{p.name[:38]:<38} {label:<16} plate={'yes' if q.get('plate') else 'no ':<3} "
              f"photo {q.get('photoEstimateG')} g (old model {q_old.get('photoEstimateG')} g) -> {q['grams']:g} g, "
              f"{n['kcal']:.0f} kcal, P {n['proteinG']} C {n['carbsG']} F {n['fatG']}  [{seconds:.1f}s]", flush=True)
    (args.out / "demo.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
