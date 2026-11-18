# SmartBite CV v3 taxonomy and dataset audit

**Status:** Phase 1 planning artifacts. No model weights, training labels, or inference behaviour have been changed.

## Decisions

- Target taxonomy: 150 food classes plus a separate `not_food` label.
- Unknown food is an inference outcome (`unknown_food`), not a trained food class.
- Preserve the legacy v2 class slugs and their original configured indices as metadata. Do not use this planning taxonomy to decode an existing ONNX output.
- First training release must be derived from an explicitly frozen manifest. Classes without enough verified images remain disabled until they pass the data gate.
- Classification, instance segmentation, and weighed-portion datasets remain separate because they require different annotations.

## Repository audit findings

The v2 configuration contains 80 entries: 79 food labels and `not_food`. The v2 preparation report says four food labels were dropped for failing the minimum-image threshold:

| Class | Raw images | Training images | Validation | Test | Status |
|---|---:|---:|---:|---:|---|
| kunna | 243 | 0 | 0 | 0 | dropped |
| aloo_gosht | 139 | 0 | 0 | 0 | dropped |
| milk_soda | 197 | 0 | 0 | 0 | dropped |
| sattu | 243 | 0 | 0 | 0 | dropped |

Therefore v2 has 75 active food classes plus `not_food` (76 model outputs), not 79 active food classes. The report records 66,521 collected files, 631 unreadable/small files, 9,924 duplicates removed, 674 label-conflict images, and final splits of 20,001 train, 4,265 validation and 4,290 test images (28,556 total).

The v2 preparation configuration uses `min_per_class: 80`, `cap_per_class: 600`, a 70/15/15 split, pHash deduplication and an EfficientNet-B0 embedding pass. These are useful safeguards, but the source report also shows substantial single-source dependence for several Pakistani foods. Class count alone is not evidence of visual diversity.

## Taxonomy v3

The 150 planned food classes consist of:
- 79 legacy food labels (75 active in v2 plus four dropped labels retained as pending).
- 71 additional classes covering fruit, vegetables, pantry ingredients, packaged snacks, dairy/protein/beverages and international dishes.

Category counts:
- south_asian_dish: 45
- packaged_snack: 20
- ingredient_side: 4
- dairy_protein_beverage: 13
- international_dish: 18
- fruit: 20
- vegetable: 15
- ingredient: 15

The taxonomy is in `configs/taxonomy_v3.yaml`. It defines stable slug-based IDs, legacy mappings, target counts, status, and the separation between `not_food` and `unknown_food`.

## Dataset registry

- `data/registry/dataset_registry.csv`: source-level register for the ten v2 datasets plus candidate datasets for segmentation, fruit coverage and portion estimation.
- `data/registry/class_coverage_v3.csv`: per-class legacy split counts, proposed target count, priority and candidate source families.

Candidate datasets are leads, not approved data. Every image source requires a license/provenance review before inclusion or redistribution. A Kaggle page or downloadable archive does not itself grant commercial-use rights. Record exact license text, attribution requirements, source URL, dataset version and any restrictions before ingesting data.

### Candidate dataset roles

| Dataset | Intended role | Important limitation |
|---|---|---|
| Food-101 | General dish classification | Not a substitute for Pakistani phone-photo coverage; review original image terms |
| FoodSeg103 | Food/ingredient masks | 103 segmentation categories do not directly equal the target class taxonomy |
| Fruits-360 | Fruit class bootstrapping | Controlled capture domain; must add real phone images |
| Open Images | Object detection candidate data | Check per-image licenses and actual food labels |
| Nutrition5k | Measured mass and RGB-D portion research | Train/evaluate the deployed path using RGB-only input; depth must not leak into inference |
| Existing SMF/Khana/Pakistani sources | Local cuisine coverage | Audit source dependence, augmentation copies, image quality and terms |

## Collection and admission rules

1. Do not add an image until its canonical label and source are recorded.
2. Remove exact and near duplicates across all sources, not only within a class.
3. Group images by original photograph, dish instance, photographer/session or video before splitting.
4. Do not split augmented copies or adjacent video frames across train/validation/test.
5. Aim for at least 400 verified images per common class as a collection target, not a hard acceptance rule. Diversity and independent source coverage matter more than raw count.
6. For initial admission, target at least two independent sources and a meaningful real-phone subset; flag classes that remain single-source.
7. Maintain a dedicated unknown-food evaluation set whose food identities are deliberately excluded from training.
8. Keep visually ambiguous labels (e.g. white sugar vs salt in unmarked bowls) out of forced fine-grained classification unless the visual evidence supports them. Allow user confirmation.
9. Keep packaged product branding/packaging separate from generic food identity where packaging can dominate the image.
10. Portion-estimation samples require measured grams from a calibrated kitchen scale and a unique serving identifier. Classification images without measured mass cannot train the mass regressor.

## Portion dataset schema

At minimum, each weighed sample should record:
- `sample_id`, `food_slug`, `image_path`, `mask_path`
- `mass_g` measured on a kitchen scale
- `capture_session_id`, `device_id`, `view_angle`, `lighting`
- `container_type`, optional known container dimensions
- `is_mixed_dish`, `visible_fraction`, `annotation_notes`

For mixed meals, annotate each separable visible component and its individual mass where practical. A single RGB image cannot reveal hidden ingredients or guarantee physical scale; report calibrated estimates and uncertainty rather than false precision.

## Next execution gate

Before training:
1. Run the new audit script/notebook against the actual downloaded source directories.
2. Verify all 150 taxonomy entries against available images and source licenses.
3. Build a frozen v3 manifest with stable train/validation/test groups.
4. Produce per-class and per-source counts, duplicate reports, and visual contact sheets.
5. Keep any class below the data-quality gate disabled rather than training a noisy class.

**No retraining is authorized by this phase.** The next phase is reproducible dataset ingestion and class-by-class coverage verification.
