# How to put the SmartBite brains into the app's backend

This guide is for the person who connects the SmartBite models (the "brains") to the app's server.

**The idea in one line:** the phone sends a food photo to the server, and the server sends back what the food is,
how many grams there are, and its calories, protein, carbs and fat. The user checks it and presses confirm.
**Nothing is saved until the user confirms.**

```
photo ──▶ food name finder (3 guesses) ──▶ plate finder ──▶ look at the plate from above
      ──▶ food weight finder (grams) ──▶ nutrition table ──▶ answer for the confirm screen
```

Everything runs on the server, in Python. The phone only uploads the photo and shows the answer.

## 1. What you need from this repo

| What | Why you need it |
|---|---|
| `smartbite_cv/` | The code. **You need it.** It finds the plate and gets each photo ready the way the brains expect. The brain files alone are not enough. |
| `models/food_name_finder.onnx` (+ `_settings.json`, `_how_sure.yaml`) | Says **what food** it is (75 foods) |
| `models/food_weight_finder_with_plate.onnx` (+ `_settings.json`) | Says **how many grams**, using the plate as a ruler |
| `models/food_weight_finder_no_plate.onnx` (+ `_settings.json`) | Says how many grams when no plate is found (backup) |
| `nutrition/nutrition_v2.csv` | Calories, protein, carbs and fat in 100 g of each food, and a normal serving size |
| `examples/fastapi_server.py` | A ready-to-use server. Run it as it is, or copy from it |
| `requirements.txt` | The Python packages to install |
| The plate finder (OWLv2 + SAM 2.1) | **Not in the repo.** It downloads by itself the first time (about 740 MB). Step 5 shows how to keep a copy for servers without internet. |

You don't need the training data, `reports/`, `notebooks/` or `tools/` to run the brains.
`models/food_name_finder_small.onnx` is a smaller, less accurate copy kept for comparison; the server doesn't use it.

## 2. Set up (Python 3.12)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

## 3. Run the ready-made server

From the repo's main folder:

```bash
uvicorn examples.fastapi_server:app --host 0.0.0.0 --port 8000
```

Wait until `http://localhost:8000/health` says `{"ok": true}`. The first start takes the longest because it
downloads the plate finder. Then try it:

```bash
curl -F "photo=@meal.jpg" http://localhost:8000/detect-food
curl -F "label=biryani" -F "grams=250" http://localhost:8000/nutrition
```

You can also click and try everything at `http://localhost:8000/docs`.

### The server's addresses (endpoints)

| Address | What you send | What you get back |
|---|---|---|
| `POST /detect-food` | a photo in the field `photo` (JPEG, PNG or HEIC, up to 15 MB) | what food, how many grams, nutrition |
| `POST /nutrition` | `label` (the food's id, like `biryani`) and `grams` (more than 0, up to 5000) | nutrition for that food and amount |
| `GET /health` | nothing | `{"ok": true}` when the brains are ready |

If something is wrong, the server says so: `400` = that file is not a photo, `413` = photo bigger than 15 MB,
`404` = no nutrition data for that food (let the user search by hand), `422` = a wrong value (like negative grams).

### What `/detect-food` sends back

| Field | What it means |
|---|---|
| `band` | How sure the food name finder is: `high`, `medium` or `low` (step 6 says what to show) |
| `message` | A sentence for the screen, like "Looks like Haleem. Please confirm or change it." |
| `suggestions` | The 3 guesses: `label` (food id), `displayName` (food name), `confidence` (0 to 1) |
| `preselected` | The food to tick for the user, only when `band` is `high`; otherwise `null` |
| `reason` | Why: `confident`, `uncertain`, `low_confidence`, or `not_food` (the photo doesn't look like food) |
| `quantity.grams` | The best guess of the grams for the first guess (photo guess mixed with a normal serving) |
| `quantity.rangeG` | "Probably between" `[low, high]` grams (right about 2 times out of 3) |
| `quantity.photoEstimateG` | The grams from the photo alone, before mixing with a normal serving |
| `quantity.plate` | What it found: `kind` (`plate` or `bowl`), `tiltDeg` (camera angle), `inFrame` (whole plate in the photo?) |
| `quantity.warning` | Only there when part of the plate is cut off. Show it and ask the user to take the photo again |
| `quantity.method` | Which brain made the grams guess (plate brain, backup brain, or just a normal serving) |
| `nutrition` | `kcal`, `proteinG`, `carbsG`, `fatG` for those grams, plus `source` and `approximate` |
| `autoLogged` | Always `false`. Never save the meal until the user confirms |

`quantity` and `nutrition` are `null` when the photo isn't food, or when that food has no row in the nutrition table.

A real answer for a photo of haleem in a bowl (shortened): `band` "high", first guess Haleem (99.6%), found a
`bowl` (tilt 6°, whole bowl in the photo), photo guess 219.3 g, normal serving 255 g, `grams` 228.7
(probably 147.5–354.6), 178 kcal, 13.6 g protein, 20.0 g carbs, 5.3 g fat.

## 4. Use the brains in your own server code

Load the brains **once**, when the server starts. Don't load them again for every photo.

```python
from smartbite_cv.predict import FoodRecognizer
from smartbite_cv.nutrition import CsvNutritionProvider

rec = FoodRecognizer(
    "models/food_name_finder.onnx",
    nutrition=CsvNutritionProvider("nutrition/nutrition_v2.csv"),
    portion_dir="models/food_weight_finder_no_plate.onnx",
    plate_portion_dir="models/food_weight_finder_with_plate.onnx",
)

result = rec.predict(photo)                # photo: a file path, bytes, or a PIL image
label = result["suggestions"][0]["label"]  # the first guess
q = rec.quantity_for(photo, label)         # grams, range, plate info, retake warning
n = rec.nutrition_for(label, q["grams"])   # kcal, protein, carbs, fat
```

The paths start from the repo's main folder. If your server starts somewhere else, use full paths.
One photo takes about 5 seconds on a normal computer. The ready-made server handles one photo at a time, so many
photos at once don't overload the computer.

## 5. Servers without internet: save the plate finder once

On a computer with internet, from the repo's main folder, run this once:

```python
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import Sam2Processor, Sam2Model

owl = "google/owlv2-base-patch16-ensemble"
for c in (Owlv2Processor, Owlv2ForObjectDetection):
    c.from_pretrained(owl).save_pretrained("models/plate_detector/owlv2")

sam = "facebook/sam2.1-hiera-tiny"
for c in (Sam2Processor, Sam2Model):
    c.from_pretrained(sam).save_pretrained("models/plate_detector/sam2")
```

Copy the `models/plate_detector/` folder to the server. Then start the ready-made server with
`SMARTBITE_PLATE_DETECTOR=models/plate_detector`, or give it to `FoodRecognizer` yourself:

```python
from smartbite_cv.plate import PlateDetector

detector = PlateDetector(
    owl_id="models/plate_detector/owlv2",
    sam_id="models/plate_detector/sam2",
)
rec = FoodRecognizer(..., plate_detector=detector)
```

The saved plate finder is about 740 MB, too big for GitHub. Keep it on the server or in cloud storage
(git already ignores `models/plate_detector/`).

## 6. What the app should show

| When… | The app should… |
|---|---|
| `band` is `high` | Tick the first guess for the user. The user still presses confirm. |
| `band` is `medium` | Ask "Is it one of these?" and show the 3 guesses. |
| `band` is `low`, or `reason` is `not_food` | Say "Couldn't recognise it, search manually" and show the 3 guesses as hints. |
| `quantity.warning` is there | Show the warning and ask for a new photo with the whole plate in it. That grams guess is not good. |
| Always | Show the grams with the "probably between" range. Let the user change the grams. Save only after confirm. |
| The user picks another food or changes the grams | Call `POST /nutrition` with the new `label` and `grams`. |

Tip for users: take the photo with the **whole plate or bowl** in it, from above or from the side. Don't cut it off.

## 7. Speed and computer needs

- About **5 seconds per photo** on a normal computer, mostly finding the plate. With an NVIDIA graphics card and
  the CUDA version of PyTorch, the plate finder uses the card by itself and runs faster.
- Disk space: about 60 MB of brains in the repo, plus about 740 MB for the plate finder.
- Memory: give it room. On a 16 GB laptop with a busy web browser, the plate finder ran out of memory.
- `SMARTBITE_THREADS=<number>` limits how many CPU threads each brain uses in the ready-made server.

## 8. How good is it?

Grams, tested on 503 dishes weighed on a kitchen scale (about 200 g each on average). On average it is off by:

| Photo taken… | Off by |
|---|---|
| From straight above | 28 g (14%) |
| From farther away | 29 g (15%) |
| From the side, whole plate visible | 41 g (22%) |
| From the side, any photo | 55 g (27%) |

Food names on real photos from the web: first guess right 57 times out of 100, one of the 3 guesses right 74 times
out of 100 (on the test photos: 83 and 93).

What it can't do yet: the weight finder learned from one American cafeteria's food and plates, and no Pakistani
meal has been weighed yet. It often guesses a bit too much, especially from the side. A plate with several foods
gets one total weight, and it can't see hidden oil or ghee. More detail: `reports/plate_results.md`.

## 9. Training the brains again

The training code is in `smartbite_cv/`. `tools/build_kaggle.py` makes the free Kaggle GPU notebooks
(`--version v2` for food names, `--version plate` for grams). After training, run `python tools/publish_models.py`
to copy the new brains into `models/`. The training photos are not in this repo, and some may only be used for
study, so keep the repo private. Full steps: `docs/TECHNICAL.md`.
