"""Copy the trained models from their export folders into models/, under names that say what each one does.

    artifacts/efficientnet_b0_v2  ->  models/food_name_finder.*                 (what food is in the photo?)
    artifacts/plate_portion       ->  models/food_weight_finder_with_plate.*    (how many grams? uses the plate)
    artifacts/portion             ->  models/food_weight_finder_no_plate.*      (how many grams? when no plate is found)
    artifacts/mobilenetv2_v2      ->  models/food_name_finder_small.*           (smaller, less accurate food name finder)

Each model becomes <name>.onnx + <name>_settings.json (+ <name>_how_sure.yaml for the food name finders).
Run after retraining:  python tools/publish_models.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {   # name in models/  ->  (export folder, settings file inside it)
    "food_name_finder": ("artifacts/efficientnet_b0_v2", "model_meta.json"),
    "food_weight_finder_with_plate": ("artifacts/plate_portion", "portion_meta.json"),
    "food_weight_finder_no_plate": ("artifacts/portion", "portion_meta.json"),
    "food_name_finder_small": ("artifacts/mobilenetv2_v2", "model_meta.json"),
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "models")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (folder, settings) in MODELS.items():
        src = ROOT / folder
        if not (src / "model.onnx").exists():
            print(f"skipped {name}: {src / 'model.onnx'} not found")
            continue
        shutil.copy2(src / "model.onnx", args.out / f"{name}.onnx")
        shutil.copy2(src / settings, args.out / f"{name}_settings.json")
        if (src / "bands.yaml").exists():
            shutil.copy2(src / "bands.yaml", args.out / f"{name}_how_sure.yaml")
        print(f"{folder} -> models/{name}.onnx")


if __name__ == "__main__":
    main()
