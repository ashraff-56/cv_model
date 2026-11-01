"""Inference path: numpy preprocessing == torchvision eval transform, and an end-to-end ONNX bundle."""
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

from smartbite_cv.bands import BandThresholds
from smartbite_cv.data import build_transforms, eval_resize
from smartbite_cv.imageio import load_rgb
from smartbite_cv.predict import FoodRecognizer, preprocess


@pytest.mark.parametrize("size", [(320, 240), (240, 320), (1000, 333), (256, 256)])
def test_numpy_preprocess_matches_torchvision(photo, size):
    img = photo(3, size)
    ref = build_transforms(224, train=False)(img).numpy()
    got = preprocess(img, 224, eval_resize(224, 0.875), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))[0]
    assert got.shape == ref.shape == (3, 224, 224)
    assert np.abs(got - ref).max() < 1e-5


def test_exif_rotation_is_applied(photo):
    img = photo(4, (300, 200))
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 CW to display" - how phones store portrait shots
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    assert load_rgb(buf.getvalue()).size == (200, 300)


@pytest.fixture
def tiny_bundle(tmp_path):
    """A real ONNX bundle from a small random-weight timm model (no download)."""
    import timm

    from smartbite_cv.export import export_onnx, parity_check

    classes = ["biryani", "samosa", "daal", "not_food"]
    model = timm.create_model("mobilenetv2_050", pretrained=False, num_classes=len(classes)).eval()
    export_onnx(model, 96, tmp_path / "model.onnx")
    assert parity_check(model, tmp_path / "model.onnx", 96) < 1e-3
    meta = {"modelName": "test", "classes": [{"slug": c, "displayName": c.title(), "variants": []} for c in classes],
            "notFoodLabel": "not_food", "imgSize": 96, "resizeShortSide": eval_resize(96, 0.875),
            "interpolation": "bicubic", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "temperature": 1.5}
    (tmp_path / "model_meta.json").write_text(json.dumps(meta))
    BandThresholds(high=0.6, low=0.3, high_top1_target=0.95, low_band_max_top3=0.5).save(tmp_path / "bands.yaml")
    return tmp_path, model


def test_bundle_end_to_end(tiny_bundle, photo, tmp_path):
    bundle, model = tiny_bundle
    rec = FoodRecognizer(bundle)
    path = tmp_path / "meal.jpg"
    photo(5).save(path)
    res = rec.predict(path)
    assert res["autoLogged"] is False
    assert res["band"] in {"high", "medium", "low"}
    assert len(res["suggestions"]) == 3 and all(s["label"] != "not_food" for s in res["suggestions"])
    confs = [s["confidence"] for s in res["suggestions"]]
    assert confs == sorted(confs, reverse=True)
    # calibrated probabilities = softmax(torch logits / T)
    x = preprocess(load_rgb(path), 96, eval_resize(96, 0.875), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    with torch.no_grad():
        ref = torch.softmax(model(torch.from_numpy(x)) / 1.5, dim=1).numpy()[0]
    assert np.allclose(rec.probabilities(path), ref, atol=1e-4)
    n = rec.nutrition_for("biryani", 200)
    assert n["grams"] == 200 and n["kcal"] > 0


@pytest.fixture
def tiny_portion_bundle(tmp_path):
    """A real exported portion bundle from a random-weight PortionNet (no download, no training)."""
    import torch

    from smartbite_cv.portion import PortionNet, export_onnx as export_portion

    run = tmp_path / "portion_run"
    (run / "eval").mkdir(parents=True)
    model_cfg = {"name": "mobilenetv2_050", "init_log_mass": float(np.log(200.0))}
    torch.save({"model": PortionNet(model_cfg, pretrained=False).state_dict(), "model_cfg": model_cfg,
                "img_hw": [60, 80]}, run / "best.pt")
    (run / "eval" / "portion_metrics.json").write_text(json.dumps({
        "uncertainty": {"sigma_scale_fitted_on_val": 1.0},
        "prior_used_for_fusion_eval": {"sigma_log": 0.6},
        "model": {"mae_g": 40.0, "mae_pct_of_mean": 19.0, "median_ape_pct": 15.0}}))
    export_portion(run, tmp_path / "portion")
    return tmp_path / "portion"


def test_quantity_combines_photo_estimate_with_typical_serving(tiny_bundle, tiny_portion_bundle, photo, tmp_path):
    from smartbite_cv.nutrition import NutritionInfo

    class Provider:
        def get(self, label):
            return NutritionInfo(label, label.title(), 150.0, 10.0, 15.0, 5.0, 300.0, "test", "1", "", "", False, "")

    bundle, _ = tiny_bundle
    rec = FoodRecognizer(bundle, nutrition=Provider(), portion_dir=tiny_portion_bundle)
    img = photo(7)
    q = rec.quantity_for(img, "biryani")
    # zero-initialised head: photo estimate = exp(init) = 200 g; typical serving 300 g; both enter the result
    assert abs(q["photoEstimateG"] - 200.0) < 1.0 and q["typicalServingG"] == 300.0
    assert 200.0 < q["grams"] < 300.0
    assert q["rangeG"][0] < q["grams"] < q["rangeG"][1]
    portrait = img.rotate(90, expand=True)
    assert abs(rec.quantity_for(portrait, "biryani")["photoEstimateG"] - 200.0) < 1.0
    n = rec.nutrition_for("biryani", q["grams"])
    assert abs(n["kcal"] - 150.0 * q["grams"] / 100) < 0.6


def test_quantity_without_portion_model_uses_typical_serving(tiny_bundle, photo):
    bundle, _ = tiny_bundle
    from smartbite_cv.nutrition import NutritionInfo

    class Provider:
        def get(self, label):
            return NutritionInfo(label, label.title(), 150.0, 10.0, 15.0, 5.0, 300.0, "test", "1", "", "", False, "")

    q = FoodRecognizer(bundle, nutrition=Provider()).quantity_for(photo(1), "daal")
    assert q["grams"] == 300.0 and "typical serving" in q["method"]


def test_plate_path_is_used_first_and_falls_back_without_a_plate(tiny_bundle, tiny_portion_bundle, photo):
    import cv2

    from smartbite_cv.nutrition import NutritionInfo
    from smartbite_cv.plate import Plate, border_fraction, mask_geometry

    class Provider:
        def get(self, label):
            return NutritionInfo(label, label.title(), 150.0, 10.0, 15.0, 5.0, 300.0, "test", "1", "", "", False, "")

    class FakeDetector:   # stands in for OWLv2 + SAM 2.1
        found = True

        def detect(self, img):
            if not self.found:
                return None
            m = np.zeros((img.height, img.width), np.uint8)
            cv2.ellipse(m, (img.width // 2, img.height // 2), (img.width // 3, img.height // 4), 0, 0, 360, 1, -1)
            m = m.astype(bool)
            return Plate("plate", "a plate", 0.9, (0, 0, img.width, img.height), m, mask_geometry(m), border_fraction(m))

    bundle, _ = tiny_bundle
    det = FakeDetector()
    rec = FoodRecognizer(bundle, nutrition=Provider(), portion_dir=tiny_portion_bundle,
                         plate_portion_dir=tiny_portion_bundle, plate_detector=det)
    q = rec.quantity_for(photo(3), "biryani")
    assert q["method"].startswith("plate-relative") and q["plate"]["inFrame"] and "warning" not in q
    det.found = False
    q = rec.quantity_for(photo(3), "biryani")
    assert q["method"] == "photo estimate combined with typical serving (no plate found)"
