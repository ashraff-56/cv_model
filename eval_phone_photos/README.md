# Real-phone-photo test set (to be collected)

Public-only testing overstates real-world accuracy: web photos are well lit, centred and styled.
This folder holds the team's own phone photos for a second, more honest test.

## Layout

```
eval_phone_photos/
  biryani/      IMG_0001.jpg ...
  daal/         ...
  not_food/     (desks, people, packaging, screens)
```

Folder names must be class slugs from `configs/classes.yaml`. Folders with other names are skipped
and listed in the output. HEIC files from iPhones work as long as `pillow-heif` is installed.

## Collection protocol (keep it consistent)

* Real meals as eaten: home, canteen, dhaba, restaurant. Do not stage or style them.
* Normal phone camera, portrait or landscape, no filters. EXIF rotation is handled.
* One dish clearly dominant per photo (the model is single-dish). Note mixed plates separately.
* Aim for at least 20 photos per class, from at least 3 people, on at least 3 different days.
* Never copy a photo into this folder that was also used for training (none are, since training is public data only).
* Get consent before adding photos with people in them. This folder is git-ignored by default.

## Run

```
python -m smartbite_cv.eval_folder --model-dir artifacts/efficientnet_b0 --folder eval_phone_photos \
    --out reports/metrics/phone_eval_efficientnet_b0.json
```

The script reports top-1/top-3 with 95% bootstrap intervals, macro-F1 over the classes present, ECE,
band behaviour and the worst confused pairs. It uses the exact deployed inference path (ONNX bundle).
