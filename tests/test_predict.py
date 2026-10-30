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
