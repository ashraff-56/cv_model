"""Inference with the exported ONNX bundle: image in -> top-3 calibrated suggestions + band.

No PyTorch needed at inference time (onnxruntime + Pillow + numpy). The result is a
SUGGESTION for the verification screen; `autoLogged` is always False.

    python -m smartbite_cv.predict photo.jpg                      # top-3 + default serving
    python -m smartbite_cv.predict photo.jpg --grams 250          # calories for 250 g of the top guess
    python -m smartbite_cv.predict photo.jpg --grams 250 --label daal   # user picked another dish
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
from .nutrition import NutritionProvider, default_provider, nutrients_for

DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts" / "efficientnet_b0"
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
                 threads: int | None = None):
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        with open(self.model_dir / "model_meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.bands = BandThresholds.load(self.model_dir / "bands.yaml")
        self.classes = [c["slug"] for c in self.meta["classes"]]
        self.display = {c["slug"]: c["displayName"] for c in self.meta["classes"]}
        self.not_food = self.meta.get("notFoodLabel", "not_food")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(self.model_dir / "model.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.nutrition = nutrition

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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Recognise a food photo (top-3 + calories)")
    ap.add_argument("image")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--grams", type=float, help="portion in grams (default: the dish's default serving)")
    ap.add_argument("--label", help="dish the user picked (default: top suggestion)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    rec = FoodRecognizer(args.model_dir)
    result = rec.predict(args.image)
    label = args.label or (result["suggestions"][0]["label"] if result["suggestions"] else None)
    if label:
        result["nutrition"] = rec.nutrition_for(label, args.grams)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"[{result['band']}] {result['message']}")
    for i, s in enumerate(result["suggestions"], 1):
        print(f"  {i}. {s['displayName']:<28} {s['confidence']:.1%}")
    if result.get("nutrition"):
        n = result["nutrition"]
        approx = " (approximate)" if n["approximate"] else ""
        print(f"\n{n['displayName']}, {n['grams']:g} g: {n['kcal']:.0f} kcal{approx} | protein {n['proteinG']} g | "
              f"carbs {n['carbsG']} g | fat {n['fatG']} g   [source: {n['source']} {n['sourceId']}]")
    print("Nothing has been logged. The user must confirm the dish and portion first.")


if __name__ == "__main__":
    main()
