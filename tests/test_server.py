"""The example FastAPI backend: request handling and JSON shape, with tiny models and a stand-in plate detector."""
import io

import cv2
import numpy as np
import pytest
from test_predict import tiny_bundle, tiny_portion_bundle  # noqa: F401  (fixtures)

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

from smartbite_cv.nutrition import NutritionInfo  # noqa: E402
from smartbite_cv.plate import Plate, border_fraction, mask_geometry  # noqa: E402
from smartbite_cv.predict import FoodRecognizer  # noqa: E402


class Provider:
    def get(self, label):
        if label == "unknown_dish":
            return None
        return NutritionInfo(label, label.title(), 150.0, 10.0, 15.0, 5.0, 300.0, "test", "1", "", "", False, "")


class FakeDetector:   # stands in for OWLv2 + SAM 2.1
    def detect(self, img):
        m = np.zeros((img.height, img.width), np.uint8)
        cv2.ellipse(m, (img.width // 2, img.height // 2), (img.width // 3, img.height // 4), 0, 0, 360, 1, -1)
        m = m.astype(bool)
        return Plate("plate", "a plate", 0.9, (0, 0, img.width, img.height), m, mask_geometry(m), border_fraction(m))


@pytest.fixture
def client(tiny_bundle, tiny_portion_bundle, monkeypatch):  # noqa: F811
    import examples.fastapi_server as server

    bundle, _ = tiny_bundle
    monkeypatch.setattr(server, "build_recognizer", lambda: FoodRecognizer(
        bundle, nutrition=Provider(), portion_dir=tiny_portion_bundle, plate_portion_dir=tiny_portion_bundle,
        plate_detector=FakeDetector()))
    with TestClient(server.app) as c:
        yield c


def test_detect_food_returns_a_suggestion_with_grams_and_nutrients(client, photo):
    assert client.get("/health").json() == {"ok": True}
    buf = io.BytesIO()
    photo(4).save(buf, format="JPEG")
    r = client.post("/detect-food", files={"photo": ("meal.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200
    out = r.json()
    assert out["autoLogged"] is False and len(out["suggestions"]) == 3
    if out["reason"] != "not_food":
        assert out["quantity"]["method"].startswith("plate-relative")
        assert out["quantity"]["rangeG"][0] < out["quantity"]["grams"] < out["quantity"]["rangeG"][1]
        assert out["nutrition"]["grams"] == out["quantity"]["grams"]


def test_bad_uploads_and_unknown_dishes_get_clear_errors(client):
    assert client.post("/detect-food", files={"photo": ("x.txt", b"hello", "text/plain")}).status_code == 400
    assert client.post("/nutrition", data={"label": "biryani", "grams": 250}).json()["kcal"] == 375.0
    assert client.post("/nutrition", data={"label": "unknown_dish", "grams": 250}).status_code == 404
    assert client.post("/nutrition", data={"label": "biryani", "grams": -5}).status_code == 422
