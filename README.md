# SmartBite: take a photo of your food, get its nutrition

## What is this?

SmartBite is an app that helps people keep track of what they eat. You take a photo of your food, and SmartBite
tells you:

1. **What** the food is (for example: biryani).
2. **How much** food there is, in grams.
3. **What is inside it**: calories, protein, carbs and fat.

Then **you** check the answer and press "confirm". SmartBite never saves anything by itself.

This repo has the **brains** (we call them models) and the **code** that runs them.

> Connecting the brains to the app's backend? Read [`INTEGRATION.md`](INTEGRATION.md). It explains every step.

## How does it work?

It works a bit like a person looking at a plate of food:

1. **Look at the food.** The *food name finder* guesses what the food is. It gives 3 guesses, best guess first.
2. **Find the plate.** The *plate finder* looks for the plate or bowl in the photo.
3. **Use the plate as a ruler.** Plates are about the same size in most homes. If the food covers half of the
   plate, that tells us roughly how much food there is. We turn the photo so it looks like we are looking down
   from straight above the plate.
4. **Guess the grams.** The *food weight finder* guesses how many grams of food are on the plate. It also says
   "probably between this much and that much".
5. **Look up the nutrition.** We multiply the grams by a nutrition table to get calories, protein, carbs and fat.

## The brains (models)

They are all in the [`models/`](models/) folder. Each name says what the brain does.

| File | What it does |
|---|---|
| `models/food_name_finder.onnx` | Says **what food** is in the photo (75 foods, like biryani, haleem, samosa) |
| `models/food_weight_finder_with_plate.onnx` | Says **how many grams**, using the plate as a ruler |
| `models/food_weight_finder_no_plate.onnx` | Says **how many grams** when no plate is found (backup) |
| `models/food_name_finder_small.onnx` | A smaller, less accurate food name finder (the app doesn't use it) |
| `nutrition/nutrition_v2.csv` | The nutrition table: calories, protein, carbs and fat in 100 g of each food |

The plate finder is two bigger brains (about 740 MB). They download by themselves the first time you run it.

## Try it on your computer

You need Python 3.12. Open a terminal in this folder and type:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m smartbite_cv.predict my_food_photo.jpg
```

On Mac or Linux, use `source .venv/bin/activate` instead of the second line.

This is what it printed for a photo of haleem in a bowl (a few technical notes removed):

```text
[high] Looks like Haleem. Please confirm or change it.
  1. Haleem                       99.6%
  2. rabri                        0.1%
  3. Daal                         0.1%

Estimated amount: 228.7 g (likely 147.5-354.6 g)

Haleem, 228.7 g: 178 kcal (approximate) | protein 13.6 g | carbs 20.0 g | fat 5.3 g
Nothing has been logged. The user must confirm the dish and portion first.
```

## Put it in the app

The brains run on the **backend** (the server), not on the phone. The phone sends the photo; the server sends
back the answer. A ready-to-use server is in `examples/fastapi_server.py`:

```bash
pip install fastapi uvicorn python-multipart
uvicorn examples.fastapi_server:app --host 0.0.0.0 --port 8000
```

Every step, every field of the answer, and what the app should show are in [`INTEGRATION.md`](INTEGRATION.md).

## How good is it?

We tested it on photos it had never seen before.

| Question | How often it gets it right |
|---|---|
| What food is it? (first guess) | 83 out of 100 test photos, 57 out of 100 real photos from the web |
| What food is it? (one of its 3 guesses) | 93 out of 100 test photos, 74 out of 100 real photos from the web |

For grams, we used 503 dishes that were weighed on a real kitchen scale (on average about 200 g each):

| Photo taken… | On average, it is off by… |
|---|---|
| From straight above | 28 g |
| From farther away | 29 g |
| From the side, whole plate visible | 41 g |
| From the side, any photo | 55 g |

## What it can't do yet

- The weight finder learned from one American cafeteria's food and plates. Nobody has weighed Pakistani meals
  for it yet, so it can be wrong on them.
- It often guesses a little **too much** food, especially from the side.
- If part of the plate is cut off in the photo, the guess is bad. The app should ask the user to take the photo
  again with the whole plate in it.
- A plate with rice, curry and roti together gets **one** total weight, not one for each food.
- It can't see hidden oil or ghee.
- It takes about **5 seconds** per photo on a normal computer.

That is why the user always checks the answer before saving.

## What's in each folder

| Folder | What's inside |
|---|---|
| `models/` | The trained brains the app uses |
| `smartbite_cv/` | The code that runs (and trains) the brains |
| `nutrition/` | The nutrition table and where its numbers come from |
| `examples/` | A ready-to-use backend server |
| `tests/` | Checks that make sure the code still works (`python -m pytest -q`) |
| `reports/` | Full results, charts and test details |
| `docs/` | Technical notes for training the brains again |
| `notebooks/`, `tools/`, `configs/`, `data/` | Everything used to train and test the brains |

## For experts

Training, data and the full results: [`docs/TECHNICAL.md`](docs/TECHNICAL.md), `reports/plate_results.md`
and `reports/v2_results.md`.

## Please remember

- This is **not** medical advice. The numbers are estimates.
- Keep this repo **private**: some of the photos used for training may only be used for study, not for business.
