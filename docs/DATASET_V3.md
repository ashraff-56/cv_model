# SmartBite v3 dataset preparation

## Safety and compatibility

- Do not overwrite `configs/classes_v2.yaml`, `configs/data_v2.yaml`, or `data/splits/manifest_v2.csv`.
- v3 keeps the legacy food class order relative to one another and appends new food classes. `not_food` is last.
- `configs/taxonomy_v3.yaml` is the planning taxonomy; `configs/classes_v3.yaml` is the actual input expected by `smartbite_cv.prepare`.
- Never decode an old ONNX model with the v3 class map. The exact class file and its hash must travel with each checkpoint.
- Unknown food is a prediction outcome, not an output class.

## Current v3 sources

The v3 classifier initially reuses all v2 food sources and adds the Fruits-360 original-size source for the visually compatible fruit and vegetable labels mapped in `classes_v3.yaml`.

Fruits-360 is useful for transfer learning and class coverage, but its objects were captured as isolated specimens against a plain background. It is not a substitute for ordinary phone photos, multiple items, plated foods or Pakistani cooking variations. Treat it as a supplemental source and add a separate phone-photo evaluation set.

FoodSeg103 is kept separate from classifier image folders. Its images contain multiple visible ingredients and pixel masks. It is a segmentation/ingredient-supervision source, not a simple one-image/one-class classification folder. Do not flatten its images into single class labels.

## Local setup

The recommended source is the **original-size** Fruits-360 repository. It contains original captured resolutions and its top-level folders are `Training`, `Validation`, and `Test`. We use only `Training` as a source pool; our own deterministic split is created later.

On Windows PowerShell, from the repository root:

```powershell
New-Item -ItemType Directory -Force data/raw/fruits360 | Out-Null
git clone --depth 1 --filter=blob:none --sparse https://github.com/fruits-360/fruits-360-original-size.git data/raw/fruits360/fruits-360-original-size
git -C data/raw/fruits360/fruits-360-original-size sparse-checkout set Training
```

Expected layout:

```
data/raw/fruits360/fruits-360-original-size/
└── Training/
    ├── Apple/
    ├── Apple Braeburn/
    ├── Banana/
    └── ...
```

The repository README describes the original-size release as 119,807 images across 164 classes in the currently documented release family; the exact branch version may change. The data is licensed CC BY-SA 4.0: retain attribution and comply with ShareAlike when redistributing adapted dataset material. The source's isolated-object, plain-background capture style is a domain limitation, not a reason to omit it.

Then run the raw source audit:

```powershell
python tools/audit_v3_sources.py --profile local
```

Review:

- `reports/data_v3/source_audit.md`
- `reports/data_v3/source_audit.csv`

This audit only counts raw files. It does not establish that images are valid, unique, correctly labelled, licensed or representative.

## Build the v3 manifest only after source audit

Once the folder audit and contact sheets have been reviewed:

```bash
python -m smartbite_cv.prepare --config configs/data_v3.yaml --classes configs/classes_v3.yaml --profile local --workers 4
```

This invokes the existing pipeline: image decoding, pHash/mirror deduplication, embedding-based duplicate audit, cross-label conflict removal, per-class minimums, deterministic splitting and reports. It can take a while and needs local copies of all configured source datasets.

Do not run this command on Kaggle before attaching every dataset in `configs/data_v3.yaml` and confirming that the folder markers match.

## Data acceptance rules

- Aim for 400 diverse images per class as a collection target; this is not a substitute for source diversity.
- The configured preparation minimum is 80 usable unique images. A class below that is dropped from the manifest.
- For newly added classes, target at least two independent sources and a meaningful phone-photo subset before considering the class ready.
- Keep near-duplicate frames and transformed copies in one split group.
- Preserve a phone-photo holdout captured separately from public web/dataset sources.
- Record source license, attribution, version, and permitted use before use in a public or commercial release.
