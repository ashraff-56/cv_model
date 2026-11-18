# SmartBite v3 training workflow

This workflow is intentionally gated. Do not start training from the raw source audit alone.

## 1. Build the v3 manifest

The Colab runtime config must point each dataset's `local` field to its extracted directory. Keep the tracked `configs/data_v3.yaml` unchanged except for shared source metadata such as valid markers.

Run from the repository root:

```bash
python -m smartbite_cv.prepare \
  --config configs/data_v3_colab.yaml \
  --classes configs/classes_v3.yaml \
  --profile local \
  --workers 2
```

This performs image decoding, pHash duplicate removal, optional ImageNet embedding duplicate review, cross-label conflict removal, class caps, and deterministic per-class splits. The final manifest is `data/splits/manifest_v3.csv`; the report is `reports/data_v3/prep_report.md`.

Review the report before training. In particular:
- classes below `min_per_class` are dropped;
- the final train/validation/test counts, not raw folder counts, determine class viability;
- inspect `embedding_pairs.csv` and `prep_stats.json` for suspicious cross-label pairs;
- review representative contact sheets for Pakistani dishes and all classes that are visually ambiguous.

## 2. Run tests

```bash
python -m pytest -q
```

## 3. Train the EfficientNet-B0 v3 baseline

Only after the preparation report and class labels have been approved:

```bash
python -m smartbite_cv.train \
  --config configs/train/base_v3.yaml \
  --config configs/train/efficientnet_b0_v3.yaml \
  --set data.config=/content/cv_model/configs/data_v3_colab.yaml \
  --profile local
```

The model uses the class definitions from `configs/classes_v3.yaml`, and the checkpoint stores the ordered labels, display names, variants, and class-config SHA-256.

## 4. Evaluate once on the held-out test split

```bash
python -m smartbite_cv.evaluate --run runs/efficientnet_b0_v3 --profile local
```

Do not tune thresholds or hyperparameters against the test split. Use validation for model selection and calibration.

## 5. Export

```bash
python -m smartbite_cv.export --run runs/efficientnet_b0_v3 --out artifacts/efficientnet_b0_v3
```

The export reads class display metadata from the checkpoint, preventing the default legacy class file from silently replacing v3 display names.
