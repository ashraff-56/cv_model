# The models (the "brains")

Each `.onnx` file is a trained brain. Its name tells you its job.

| File | Its job | Used by the app? |
|---|---|---|
| `food_name_finder.onnx` | Looks at the photo and says **what food it is** (biryani, haleem, samosa… 75 foods). Gives its top 3 guesses. | Yes |
| `food_weight_finder_with_plate.onnx` | Says **how many grams** of food there are. It uses the plate like a ruler. | Yes |
| `food_weight_finder_no_plate.onnx` | Also says **how many grams**, but only used when no plate is found in the photo. | Yes (backup) |
| `food_name_finder_small.onnx` | A smaller, less accurate food name finder. Kept for comparison. | No |

The small files next to each brain are its instructions. Keep them in the same folder with the same names:

- `..._settings.json`: how to prepare the photo for the brain, and the list of foods it knows.
- `..._how_sure.yaml`: when the food name finder should say "I'm sure" and when it should say "I'm not sure".

The plate finder (the part that spots the plate in the photo) is not in this folder. It is two bigger brains
(about 740 MB) that download by themselves the first time you run the app. See `INTEGRATION.md`.

To refresh this folder after training new models, run `python tools/publish_models.py`.
