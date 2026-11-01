# SmartBite food scan: integration guide for the backend

This guide is for whoever connects the SmartBite AI models to the app's backend. The Flutter app sends a food photo.
The backend returns the dish (top 3 guesses), the estimated grams with a likely range, and the nutrients (kcal,
protein, carbs, fat). **Nothing is logged automatically:** the user confirms or edits the dish and grams first.

Everything runs in the Python backend (FastAPI). The phone app only uploads the photo and shows the result.

```
photo ─▶ dish classifier (top 3) ─▶ plate detector ─▶ plate-only top-down view ─▶ grams model
                                                                               ─▶ grams × nutrition table
                                                                               ─▶ JSON for the confirm screen
```

## 1. What you need from this repo

| Path | What it is |
|---|---|
| `smartbite_cv/` | The Python package. **Required**: it finds the plate and prepares the images exactly as the models expect. The model files alone are not enough. |
| `artifacts/efficientnet_b0_v2/` | Dish classifier, 76 dishes (`model.onnx`, `model_meta.json`, `bands.yaml`) |
| `artifacts/plate_portion/` | Grams model that uses the plate as a ruler (`model.onnx`, `portion_meta.json`) |
| `artifacts/portion/` | Fallback grams model, used when no plate is found |
| `nutrition/nutrition_v2.csv` | kcal, protein, carbs and fat per 100 g, plus a typical serving, for each dish |
| `examples/fastapi_server.py` | A working backend you can run as is or copy from |
| `requirements.txt` | Python packages |
| Plate detector (OWLv2 + SAM 2.1) | **Not in the repo.** It downloads from Hugging Face on first use, about 740 MB (section 5 covers offline servers) |

You do **not** need the training data, `reports/`, `notebooks/` or `tools/` to run the models.
`artifacts/mobilenetv2_v2/` is a smaller baseline classifier kept for comparison; the backend doesn't use it.

## 2. Set up

Python 3.12 is what everything was tested with.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

The first start downloads the plate detector from Hugging Face, so the server needs internet once (or see section 5).

## 3. Run the example backend

From the repo root:

```bash
uvicorn examples.fastapi_server:app --host 0.0.0.0 --port 8000
```

Wait until `GET /health` returns `{"ok": true}` (the first start takes longest because it downloads the plate detector). Then try it:

```bash
curl -F "photo=@meal.jpg" http://localhost:8000/detect-food
curl -F "label=biryani" -F "grams=250" http://localhost:8000/nutrition
```

FastAPI's interactive docs are at `http://localhost:8000/docs`.

### Endpoints

| Endpoint | Input | Returns |
|---|---|---|
| `POST /detect-food` | multipart field `photo` (JPEG, PNG or HEIC, up to 15 MB) | dish suggestions, grams, nutrients (below) |
| `POST /nutrition` | form fields `label` (a dish id such as `biryani`) and `grams` (0–5000) | nutrients for that dish and amount |
| `GET /health` | – | `{"ok": true}` once the models are loaded |

Errors: `400` = not a photo, `413` = photo over 15 MB, `404` = no nutrition data for that dish (use manual search),
`422` = invalid input (for example negative grams).

### What `/detect-food` returns

| Field | Meaning |
|---|---|
| `band` | `high`, `medium` or `low` confidence (see section 6) |
| `message` | Text for the confirm screen, for example "Looks like Haleem. Please confirm or change it." |
| `suggestions` | Top 3 dishes: `label` (dish id), `displayName`, `confidence` (0–1) |
| `preselected` | The dish id to pre-select (only in the `high` band), otherwise `null` |
| `reason` | `confident`, `uncertain`, `low_confidence` or `not_food` |
| `quantity.grams` | Estimated grams for the top dish (photo estimate combined with that dish's typical serving) |
| `quantity.rangeG` | Likely range `[low, high]` in grams (about 68% likely to contain the true amount) |
| `quantity.photoEstimateG` | Grams from the photo alone, before combining with the typical serving |
| `quantity.plate` | What was found: `kind` (`plate` or `bowl`), `tiltDeg` (camera angle), `inFrame` (whole plate visible) |
| `quantity.warning` | Present when part of the plate is outside the photo; show it and offer a retake |
| `quantity.method` | Which estimate was used (plate model, fallback model, or typical serving) |
| `nutrition` | `kcal`, `proteinG`, `carbsG`, `fatG` for `quantity.grams`, plus `source` and `approximate` |
| `autoLogged` | Always `false`: the app must not save the meal until the user confirms |

`quantity` and `nutrition` are `null` when the photo doesn't look like food, or when the top dish has no row in the
nutrition table.

An actual result for a Wikipedia photo of haleem in a bowl, shortened: `band` "high", top suggestion Haleem
(99.6%), plate found (`bowl`, tilt 6°, in frame), photo estimate 219.3 g, typical serving 255 g, `grams` 228.7
(range 147.5–354.6), nutrition 178 kcal, 13.6 g protein, 20.0 g carbs, 5.3 g fat.

## 4. Use it inside your own FastAPI code

Load the models **once at startup**, not per request:

```python
from smartbite_cv.predict import FoodRecognizer
from smartbite_cv.nutrition import CsvNutritionProvider

rec = FoodRecognizer("artifacts/efficientnet_b0_v2",
                     nutrition=CsvNutritionProvider("nutrition/nutrition_v2.csv"),
                     portion_dir="artifacts/portion",
                     plate_portion_dir="artifacts/plate_portion")

result = rec.predict(photo)                  # photo: file path, bytes or a PIL image
label = result["suggestions"][0]["label"]
q = rec.quantity_for(photo, label)           # grams, range, plate info, retake warning
n = rec.nutrition_for(label, q["grams"])     # kcal, protein, carbs, fat
```

Paths are relative to the repo root; use absolute paths if your server starts somewhere else. One scan takes about
5 seconds on a CPU. `examples/fastapi_server.py` runs scans one at a time behind a lock so several requests don't
overload the CPU.

## 5. Servers without internet: save the plate detector once

On a machine with internet, from the repo root:

```python
from transformers import Owlv2Processor, Owlv2ForObjectDetection, Sam2Processor, Sam2Model
for c in (Owlv2Processor, Owlv2ForObjectDetection):
    c.from_pretrained("google/owlv2-base-patch16-ensemble").save_pretrained("artifacts/plate_detector/owlv2")
for c in (Sam2Processor, Sam2Model):
    c.from_pretrained("facebook/sam2.1-hiera-tiny").save_pretrained("artifacts/plate_detector/sam2")
```

Copy `artifacts/plate_detector/` to the server. Then either start the example backend with
`SMARTBITE_PLATE_DETECTOR=artifacts/plate_detector`, or pass the detector yourself:

```python
from smartbite_cv.plate import PlateDetector

rec = FoodRecognizer(..., plate_detector=PlateDetector(owl_id="artifacts/plate_detector/owlv2",
                                                       sam_id="artifacts/plate_detector/sam2"))
```

The saved detector is about 740 MB, too big to keep in this GitHub repo; store it on the server or in cloud storage.

## 6. What the app should do with the result

| Situation | App behaviour |
|---|---|
| `band` = `high` | Pre-select the top dish; the user still taps confirm |
| `band` = `medium` | "Is it one of these?" with the 3 suggestions |
| `band` = `low` or `reason` = `not_food` | "Couldn't recognise it, search manually"; show the 3 suggestions as hints |
| `quantity.warning` present | Show it and offer a retake with the whole plate in view; the estimate is unreliable |
| Always | Show `grams` with its `rangeG`, let the user edit the grams, and save only after confirmation |
| User picks another dish or edits grams | Call `POST /nutrition` with the new `label` and `grams` |

Tips for users: photograph the whole plate or bowl, from above or at an angle, without cropping it.

## 7. Speed and resources

- About **5 seconds per photo on a CPU**, mostly the plate detector. If the server has an NVIDIA GPU with CUDA
  PyTorch installed, the detector uses it automatically.
- Disk: about 60 MB of models in the repo, plus about 740 MB for the plate detector.
- Memory: give it room. On a 16 GB laptop with a busy browser the detector ran out of memory, so a server with
  free RAM to spare is the safe choice.
- `SMARTBITE_THREADS=<n>` limits CPU threads per model in the example backend.

## 8. How accurate it is

Measured on 503 weighed Nutrition5k test dishes, average error per dish:

| Photo | Error |
|---|---|
| From above | 28 g (14%) |
| Camera farther away | 29 g (15%) |
| Side angle, whole plate visible | 41 g (22%) |
| Side angle, all photos | 55 g (27%) |

Dish recognition on real web photos: 56.9% top-1 and 74.4% top-3 (internal test: 82.6% / 92.9%).

Known limits: the grams model was trained and tested on one American cafeteria's dishes and plates, and no Pakistani
meal has been weighed yet. It tends to estimate too high, especially from side angles. A mixed plate gets one
total, and hidden oil or ghee is invisible. Details: `reports/plate_results.md` and `reports/v2_results.md`.

## 9. Retraining or improving the models

The training code is in `smartbite_cv/`, and the Kaggle GPU notebooks are generated by `tools/build_kaggle.py`
(`--version v2` for the dish classifier, `--version plate` for the grams model). The training datasets are not in
this repo; the download steps are in `README.md`. Some datasets are for non-commercial or academic use only, so keep
the repo private.
