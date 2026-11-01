"""Minimal FastAPI backend for the SmartBite food scan: copy it into the real API or run it as is.

POST /detect-food   multipart "photo"           -> top-3 dishes, estimated grams with a range, nutrients
POST /nutrition     form "label" + "grams"      -> nutrients after the user picks another dish or edits grams
GET  /health                                    -> {"ok": true} once the models are loaded

Nothing is logged here: every result is a suggestion for the confirm screen ("autoLogged": false). The models
load once at startup (the first start also downloads the plate detector from Hugging Face). One scan takes
about 5 s on a CPU, mostly plate detection, so scans run one at a time.

    pip install -r requirements.txt fastapi uvicorn python-multipart
    uvicorn examples.fastapi_server:app --host 0.0.0.0 --port 8000      # run from the repo root

Environment: SMARTBITE_PLATE_DETECTOR=<folder> uses a saved copy of the plate detector (servers without internet;
see INTEGRATION.md), SMARTBITE_THREADS=<n> limits CPU threads per model.
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from smartbite_cv.imageio import load_rgb
from smartbite_cv.nutrition import CsvNutritionProvider
from smartbite_cv.predict import FoodRecognizer

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
state: dict = {}
scan_lock = threading.Lock()


def build_recognizer() -> FoodRecognizer:
    detector = None
    folder = os.environ.get("SMARTBITE_PLATE_DETECTOR")
    if folder:
        from smartbite_cv.plate import PlateDetector
        detector = PlateDetector(owl_id=f"{folder}/owlv2", sam_id=f"{folder}/sam2")
    threads = int(os.environ["SMARTBITE_THREADS"]) if os.environ.get("SMARTBITE_THREADS") else None
    return FoodRecognizer("artifacts/efficientnet_b0_v2",
                          nutrition=CsvNutritionProvider("nutrition/nutrition_v2.csv"),
                          threads=threads,
                          portion_dir="artifacts/portion",
                          plate_portion_dir="artifacts/plate_portion",
                          plate_detector=detector)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rec"] = build_recognizer()
    yield
    state.clear()


app = FastAPI(title="SmartBite food scan", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"ok": "rec" in state}


@app.post("/detect-food")
def detect_food(photo: UploadFile = File(...)) -> dict:
    data = photo.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Photo is larger than 15 MB. Send a smaller JPEG.")
    try:
        img = load_rgb(data)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "This file is not a photo. Send a JPEG, PNG or HEIC image.")
    rec: FoodRecognizer = state["rec"]
    with scan_lock:
        result = rec.predict(img)
        result["quantity"] = result["nutrition"] = None
        if result["reason"] != "not_food" and result["suggestions"]:
            label = result["suggestions"][0]["label"]
            try:
                q = rec.quantity_for(img, label)
            except KeyError:   # no nutrition row for this dish: the app falls back to manual search
                return result
            result["quantity"] = q
            result["nutrition"] = rec.nutrition_for(label, q["grams"])
    return result


@app.post("/nutrition")
def nutrition(label: str = Form(...), grams: float = Form(..., gt=0, le=5000)) -> dict:
    try:
        return state["rec"].nutrition_for(label, grams)
    except KeyError:
        raise HTTPException(404, f"No nutrition data for {label!r}. Use manual search for this food.")
