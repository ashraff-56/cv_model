"""Inference with the exported ONNX bundles: photo in -> top-3 dishes, estimated grams, nutrients.

No PyTorch needed at inference time (onnxruntime + Pillow + numpy). The result is a
SUGGESTION for the verification screen; `autoLogged` is always False.

Grams come from the photo: the estimate is combined with the dish's typical serving according to the model's own
uncertainty, and a ~68% range is reported. The plate model (models/food_weight_finder_with_plate.onnx) uses the plate
as the ruler (needs torch + transformers for the plate detector); when no plate is found it falls back to
models/food_weight_finder_no_plate.onnx, then to the typical serving. Models can also be export folders.

    python -m smartbite_cv.predict photo.jpg                  # dish + grams + nutrients (models/ by default)
    python -m smartbite_cv.predict photo.jpg --no-plate       # skip the plate detector (faster, less accurate)
    python -m smartbite_cv.predict photo.jpg --grams 250      # user-entered grams
    python -m smartbite_cv.predict photo.jpg --label daal     # user picked another dish
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .bands import BandThresholds, assign_band
from .config import REPO_ROOT
from .imageio import ImageSource, load_rgb
from .model_files import model_files
from .nutrition import CsvNutritionProvider, NutritionProvider, default_provider, nutrients_for

MODELS = REPO_ROOT / "models"
DEFAULT_MODEL_DIR = MODELS / "food_name_finder.onnx"
DEFAULT_NUTRITION_CSV = REPO_ROOT / "nutrition" / "nutrition_v2.csv"   # matches food_name_finder's 76 dishes
MESSAGES = {
    "high": "Looks like {name}. Please confirm or change it.",
    "medium": "Is it one of these?",
    "low": "Couldn't recognise it. Please search manually.",
}
_RESAMPLE = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR}


def preprocess(img: Image.Image, img_size: int, resize_short: int, mean, std,
               interpolation: str = "bicubic") -> np.ndarray:
    """Same pixels as torchvision Resize(short side) -> CenterCrop -> ToTensor -> Normalize."""
    w, h = img.size
    if w <= h:
        new_w, new_h = resize_short, int(resize_short * h / w)
    else:
        new_w, new_h = int(resize_short * w / h), resize_short
    img = img.resize((new_w, new_h), _RESAMPLE[interpolation])
    left = int(round((new_w - img_size) / 2.0))
    top = int(round((new_h - img_size) / 2.0))
    img = img.crop((left, top, left + img_size, top + img_size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return x.transpose(2, 0, 1)[None]


class FoodRecognizer:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, nutrition: NutritionProvider | None = None,
                 threads: int | None = None, portion_dir: str | Path | None = None,
                 plate_portion_dir: str | Path | None = None, plate_detector=None):
        import onnxruntime as ort

        self.files = model_files(model_dir, "model_meta.json")
        with open(self.files.settings, encoding="utf-8") as f:
            self.meta = json.load(f)
        if self.files.how_sure is None:
            raise FileNotFoundError(f"confidence thresholds missing next to {self.files.onnx}")
        self.bands = BandThresholds.load(self.files.how_sure)
        self.classes = [c["slug"] for c in self.meta["classes"]]
        self.display = {c["slug"]: c["displayName"] for c in self.meta["classes"]}
        self.not_food = self.meta.get("notFoodLabel", "not_food")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(self.files.onnx), opts,
                                            providers=["CPUExecutionProvider"])
        self.nutrition = nutrition
        self.portion = None
        if portion_dir:
            from .portion import PortionEstimator
            self.portion = PortionEstimator(portion_dir, threads)
        self.plate_portion = None
        if plate_portion_dir:
            from .plate_portion import PlatePortionEstimator
            self.plate_portion = PlatePortionEstimator(plate_portion_dir, plate_detector, threads)

    def probabilities(self, image: ImageSource | Image.Image) -> np.ndarray:
        img = image if isinstance(image, Image.Image) else load_rgb(image)
        m = self.meta
        x = preprocess(img, m["imgSize"], m["resizeShortSide"], m["mean"], m["std"], m["interpolation"])
        logits = self.session.run(["logits"], {"input": x})[0][0].astype(np.float64)
        z = logits / m["temperature"]
        z -= z.max()
        p = np.exp(z)
        return p / p.sum()

    def predict(self, image: ImageSource | Image.Image, top_k: int = 3) -> dict:
        p = self.probabilities(image)
        order = np.argsort(-p)
        top_label = self.classes[order[0]]
        band, reason = assign_band(float(p[order[0]]), top_label, self.bands)
        suggestions = [{"label": self.classes[i], "displayName": self.display[self.classes[i]],
                        "confidence": round(float(p[i]), 4)}
                       for i in order if self.classes[i] != self.not_food][:top_k]
        name = suggestions[0]["displayName"] if suggestions else ""
        return {
            "band": band,
            "reason": reason,
            "message": MESSAGES[band].format(name=name) if reason != "not_food"
            else "This doesn't look like food. Please search manually.",
            "preselected": suggestions[0]["label"] if band == "high" else None,
            "suggestions": suggestions,
            "notFoodProbability": round(float(p[self.classes.index(self.not_food)]), 4)
            if self.not_food in self.classes else None,
            "autoLogged": False,  # every result must be confirmed by the user before logging
            "model": self.meta["modelName"],
        }

    def nutrition_for(self, label: str, grams: float | None = None) -> dict:
        provider = self.nutrition or default_provider()
        if grams is None:
            info = provider.get(label)
            if info is None:
                raise KeyError(f"no nutrition data for label {label!r}")
            grams = info.default_serving_g
        return nutrients_for(label, grams, provider)

    def quantity_for(self, image: ImageSource | Image.Image, label: str) -> dict:
        """Estimated grams of `label` in the photo (portion model + typical serving), with a ~68% range."""
        provider = self.nutrition or default_provider()
        info = provider.get(label)
        if info is None:
            raise KeyError(f"no nutrition data for label {label!r}")
        img = image if isinstance(image, Image.Image) else load_rgb(image)
        if self.plate_portion is not None:
            q = self.plate_portion.estimate(img, info.default_serving_g)
            if q is not None:
                q["method"] = "plate-relative photo estimate combined with typical serving"
                return q
        if self.portion is None:
            return {"grams": info.default_serving_g, "method": "typical serving (no plate found, no portion model)"
                    if self.plate_portion is not None else "typical serving (no portion model loaded)",
                    "typicalServingG": info.default_serving_g}
        q = self.portion.estimate(img, info.default_serving_g)
        q["method"] = "photo estimate combined with typical serving" + (
            " (no plate found)" if self.plate_portion is not None else "")
        return q


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Recognise a food photo (top-3 + calories)")
    ap.add_argument("image")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="dish model: .onnx file or export folder")
    ap.add_argument("--grams", type=float, help="portion in grams (default: estimated from the photo)")
    ap.add_argument("--label", help="dish the user picked (default: top suggestion)")
    ap.add_argument("--portion-dir", default=str(MODELS / "food_weight_finder_no_plate.onnx"),
                    help="grams model used when no plate is found ('' to turn off)")
    ap.add_argument("--plate-portion-dir", default=str(MODELS / "food_weight_finder_with_plate.onnx"),
                    help="grams model that uses the plate as a ruler")
    ap.add_argument("--no-plate", action="store_true", help="skip the plate detector (faster, less accurate)")
    ap.add_argument("--nutrition-csv", default=str(DEFAULT_NUTRITION_CSV), help="nutrition table matching the dish model")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    provider = CsvNutritionProvider(args.nutrition_csv) if args.nutrition_csv else None
    rec = FoodRecognizer(args.model_dir, nutrition=provider, portion_dir=args.portion_dir or None,
                         plate_portion_dir=None if args.no_plate else (args.plate_portion_dir or None))
    img = load_rgb(args.image)
    result = rec.predict(img)
    label = args.label or (result["suggestions"][0]["label"] if result["suggestions"] else None)
    if label:
        if args.grams is None:
            result["quantity"] = rec.quantity_for(img, label)
        grams = args.grams if args.grams is not None else result["quantity"]["grams"]
        result["nutrition"] = rec.nutrition_for(label, grams)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"[{result['band']}] {result['message']}")
    for i, s in enumerate(result["suggestions"], 1):
        print(f"  {i}. {s['displayName']:<28} {s['confidence']:.1%}")
    if result.get("quantity"):
        q = result["quantity"]
        rng = f" (likely {q['rangeG'][0]:g}-{q['rangeG'][1]:g} g)" if "rangeG" in q else ""
        print(f"\nEstimated amount: {q['grams']:g} g{rng}  [{q['method']}]")
        if q.get("warning"):
            print(f"  ! {q['warning']}")
    if result.get("nutrition"):
        n = result["nutrition"]
        approx = " (approximate)" if n["approximate"] else ""
        print(f"\n{n['displayName']}, {n['grams']:g} g: {n['kcal']:.0f} kcal{approx} | protein {n['proteinG']} g | "
              f"carbs {n['carbsG']} g | fat {n['fatG']} g   [source: {n['source']} {n['sourceId']}]")
    print("Nothing has been logged. The user must confirm the dish and portion first.")


if __name__ == "__main__":
    main()
