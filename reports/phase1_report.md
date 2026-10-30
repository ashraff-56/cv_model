# SmartBite: Human-in-the-Loop Food Recognition and Calorie Estimation (Phase 1)

*Report notes. Every number below comes from a logged run: `reports/data/prep_report.md`, the Kaggle
run outputs (`reports/metrics/`) and `nutrition/nutrition.csv`. Not medical advice.*

## 1. Summary

A user photographs a meal. The backend classifies the dominant dish and returns the **three most likely
dishes with calibrated confidences** and a **confidence band** that drives the verification screen.
Calories are computed as *kcal per 100 g × grams the user selects*. The model never estimates portion
size and never logs anything by itself. Two ImageNet-pretrained CNNs were fine-tuned on 32 classes
(18 South Asian dishes, 13 international dishes, and `not_food`): **EfficientNet-B0** (primary) and
**MobileNetV2** (baseline named in the original spec).

## 2. Method

### 2.1 Problem framing

Recognition is posed as **single-dish image classification** with mandatory human confirmation, not
object detection. Public South Asian food data has image-level labels only, so a detector would need
thousands of hand-drawn boxes. The agreed scope is one dominant dish per photo plus manual "add item".
The verification step absorbs residual model error: the model only needs the correct dish *in its top
three*, and the user picks it.

### 2.2 Data sources

Only public data was used. Licence fields are quoted exactly as the uploader declared them on the
source page. Most South Asian sets were scraped from web image search by third parties. An uploader's
"CC0"/"MIT" label does not grant rights in photographs they did not take, so all use here is academic
and non-commercial, and no images are redistributed with the project.

| ID | Dataset (source page) | Declared licence (verbatim) | Declared origin (verbatim) |
|---|---|---|---|
| food101 | Food-101 [1]; Kaggle copy `kmader/food41` (byte-identical to the Hugging Face mirror, MD5-checked) | `copyright-authors`; description: "The Food-101 data set consists of images from Foodspotting [1]. Any use beyond scientific fair use must be negotiated with the respective picture owners according to the Foodspotting terms of use [2]." | Foodspotting |
| l33t20 | `l33tc0d3r/indian-food-classification` | `MIT` | "All the images are extracted from google." |
| anshul | `anshulmehtakaggl/themassiveindianfooddataset` | `CC0-1.0` | "Images from the Internet" |
| sourav80 | `iamsouravbanerjee/indian-food-images-dataset` | `other` | "Google Images" |
| pkdishes | `useractivated/dataset` (Pakistani Dishes Dataset) | `ODbL-1.0` | "Google Images" |
| food20 | `cdart99/food20dataset` | `CC0-1.0` | (none stated) |
| fastfood | `utkarshsaxenadn/fast-food-classification-dataset` (V2) | `CC0-1.0` | "The data set is collected through Internet and processed using windows image processor" |
| food5k | Food-5K, EPFL MMSPG [6] (non-food half only; Kaggle copy `trolukovich/food5k-image-dataset`) | EPFL page: "Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, modify, and distribute the data provided and its documentation for research purpose only. The data provided may not be commercially distributed." | EPFL |

**Rejected after inspection:**

* `harishkumardatalab/food-image-classification-dataset`: its class folders and counts match Food-101,
  `l33tc0d3r` and the fast-food set. The originals were used instead.
* `izbaiman/food-images`: contains the same 80 class folders as `iamsouravbanerjee`, relabelled `CC0-1.0`.
* The `pakistani_dishes_dataset_2` folder of `useractivated/dataset`: exactly 765 images in every class.
  A visual check showed rotated, flipped and stretched copies of a few originals (augmentation). It
  would leak between splits and cannot be caught by hashing, because rotations change the hash.
* `jvageesh11/foodies-ai-food-image-classification-challenge`: licence declared `unknown`.

### 2.3 Classes

Look-alike variants are merged into one *visual* class; the user picks the exact variant after a match
(Phase 2 maps each class to several `foods` documents).

* South Asian (18): biryani, samosa, naan, roti/chapati, paratha, daal (makhani/tadka/etc.),
  curry/karahi (chicken curry, butter chicken, tikka masala, kadai/karahi paneer), haleem, chicken
  tikka/tandoori, pakora, chanay/chole, jalebi, gulab jamun, chai, paratha roll/kathi roll, gol
  gappay/pani puri, halwa (suji/gajar), kulfi.
* International (13): pizza, burger, french fries, fried rice, omelette, fried chicken/wings, sandwich,
  hot dog, donut, ice cream, chocolate cake, spaghetti (bolognese/carbonara), salad (greek/caesar).
* `not_food` (Food-5K non-food images): lets the app say "this doesn't look like food".

No usable public images (or fewer than the 80-image minimum) were found for **nihari, karahi-specific
photos, plain rice, boiled/fried eggs, fruit, shawarma, kheer, saag and lassi**. These are not
recognised and fall back to manual search.

### 2.4 Leakage-safe preparation

Implemented in `smartbite_cv/prepare.py`, run once on all raw files:

1. Decode each image with EXIF orientation applied. Drop unreadable or tiny images (shorter side < 96 px).
2. **Exact and near duplicates:** a 64-bit DCT perceptual hash [14] of each image *and of its mirror
   image*; Hamming distance ≤ 6 counts as the same photo. Clusters are built with union-find.
3. **Re-cropped copies:** pHash missed pairs that were the same photo re-cropped. A visual check found
   such pairs straddling train/test at pHash distance 8-10, while unrelated photos also occur at that
   distance. A second pass therefore compares global ImageNet features (EfficientNet-B0, no food
   training) and merges pairs with cosine similarity ≥ 0.85. The threshold was chosen from contact
   sheets of pairs in cosine bins (`tools/audit_embedding_pairs.py`, `reports/data/embedding_pairs.csv`):
   almost all pairs ≥ 0.90 were re-crops of one photo; 0.85-0.90 still contained some; most pairs below
   0.85 were distinct photos. A false merge only costs one training image, while a missed duplicate
   inflates test accuracy, so the lower threshold was preferred.
4. **Label conflicts:** a duplicate cluster whose members carry different labels (for example identical
   files in `iamsouravbanerjee`'s `chicken_tikka` and `chicken_tikka_masala` folders) is dropped entirely.
5. One representative per cluster: the largest image, with ties broken by content hash.
6. Cap of 600 images per class (seeded random subset) to limit imbalance; classes under 80 are dropped
   (none were).
7. **Split per class 70/15/15 before any augmentation**, ordered by a seeded hash of the image bytes.
   The split therefore does not depend on file names or order (unit-tested), and is identical on any
   machine with the same files (verified on Kaggle by MD5).

| | images |
|---|---:|
| raw files collected | 39,757 |
| unreadable or too small | 3 |
| duplicates removed (pHash + embedding passes) | 3,376 |
| images in label-conflict clusters (38 clusters) | 88 |
| after the per-class cap: **train / val / test** | **10,842 / 2,324 / 2,324** (15,490 total) |

The per-class and per-source table is in `reports/data/prep_report.md`. Five classes are small:
paratha (85), haleem (82), gulab jamun (124), halwa (154) and chicken tikka (155). Their test sets hold
only 13-24 images, so their per-class numbers carry wide uncertainty.

### 2.5 Models and training

* Backbones from timm [5] with ImageNet-1k weights: `efficientnet_b0.ra_in1k` [2] (4,048,540
  parameters with the 32-class head) and `mobilenetv2_100.ra_in1k` [3] (2,264,864 parameters).
* **Stage 1:** frozen backbone (BatchNorm statistics frozen too), new zero-initialised linear head,
  3 epochs, AdamW [12] at lr 1e-3. Zero initialisation was needed: timm's default head
  initialisation gave initial losses far above ln C in a smoke test.
* **Stage 2:** all layers unfrozen, up to 25 epochs, AdamW lr 3e-4, weight decay 0.05, 1 warm-up epoch,
  cosine decay to 1e-6. Early stopping after 6 epochs without validation macro-F1 improvement. The
  checkpoint with the best validation macro-F1 is kept.
* 224 × 224 input; augmentation: random-resized crop (scale 0.35-1), horizontal flip, ±15° rotation
  (p = 0.3), mild colour jitter (colour is a real cue for food), random erasing (p = 0.15). Label
  smoothing 0.1 [13]. Square-root class-balanced sampling (weight ∝ 1/√n_c). Mixed precision on GPU.
* Fixed seed 20261031. GPU kernels are not bit-deterministic, so re-runs can differ slightly.
* Training ran on a Kaggle GPU (see results). The development PC's AMD RX 590 is not usable by PyTorch
  for training on Windows.

### 2.6 Calibration and confidence bands

Softmax outputs of fine-tuned CNNs are usually mis-calibrated: often over-confident, while label
smoothing pushes them toward under-confidence. A single **temperature** T is fitted on the validation split by minimising
negative log-likelihood [4]. Calibration quality is reported as expected calibration error (ECE, 15
equal-width bins) and NLL before and after scaling, on the test split.

Band thresholds are chosen on the **validation** split from coverage-accuracy curves of the
calibrated top-1 confidence (selective classification [11]), then frozen in `bands.yaml`:

* **high**: the lowest confidence t whose accepted subset (confidence ≥ t) has top-1 accuracy ≥ 95%
  (at least 30 images). The app pre-selects the top guess; the user still confirms.
* **low**: the highest t such that, among images with confidence < t, the true dish is in the top 3 at
  most 50% of the time (at least 30 images). Below it the suggestions are more often wrong than right,
  so the app opens manual search and shows the top 3 only as hints. A top guess of `not_food` is also
  "low".
* **medium**: everything in between: "Is it one of these?" with the top 3.

### 2.7 Calorie estimation

`calories(label, grams) = kcal_per_100g × grams / 100`, behind a `NutritionProvider` interface that
Phase 2 replaces with the Firestore `foods` collection. `nutrition/nutrition.csv` is generated by
`tools/build_nutrition.py` from the databases themselves (no hand-typed values). Source priority:
USDA FoodData Central FNDDS 2021-2023 [7], then UK CoFID 2021 [8], then the Indian Nutrient Databank
(INDB) [9]. Each row records database, food ID, description and URL, plus a default serving taken from
FNDDS portion weights or INDB serving sizes. Where a second database has the same dish, its value is
stored as a cross-check, and a difference over 30% flags the row `approximate`.

15 of 31 rows are approximate. They fall into three groups:

* **Stand-ins (no source has the dish):** haleem (lentil soup with meat, probably too low), jalebi
  (funnel cake with sugar), gol gappay (fried puri shell only).
* **Strong disagreements between databases (kcal/100 g):** pakora (FNDDS 125 vs CoFID 293), gulab jamun
  (CoFID 273 vs INDB 586), kulfi (CoFID 457 vs INDB 98), biryani (FNDDS 104 vs CoFID 163), chana/chole
  (CoFID 112 vs INDB 163), halwa (CoFID 342 vs INDB 226), chicken tikka (CoFID 214 vs INDB 145).
* **Non-USDA or merged-variant rows:** chai (FNDDS tea with milk has no sugar), curry/karahi (chicken
  curry stands in for all variants), paratha roll (INDB paneer kathi roll), spaghetti (meat-sauce pasta),
  salad (Greek salad without dressing).

INDB deep-fried recipes come out at about 550-700 kcal/100 g, which suggests all frying oil is counted as
absorbed. INDB was therefore not used as the primary source for fried foods. **These rows need a
dietitian's review before the values go into the production `foods` collection.**

### 2.8 Inference path

The best checkpoint is exported to ONNX (opset 17) and checked against PyTorch: export is refused if
any logit differs by more than 1e-3. Inference uses ONNX Runtime on CPU with numpy/Pillow
preprocessing. A unit test checks this preprocessing against the torchvision evaluation transform to
within 1e-5. Phone photos are EXIF-rotated, and HEIC is supported. The output always carries
`autoLogged: false`.

<!-- RESULTS -->

## 5. Limitations

* **Public web photos only.** Training and test images are mostly well lit, centred and styled web
  photos, often from food blogs. Accuracy on real phone photos of home and street food in Pakistan
  will be lower than reported here. Measuring that gap needs a team-collected test set
  (`eval_phone_photos/`, protocol in its README); none was available for this report.
* **Label noise.** The scraped sets contain watermarked stock photos, mislabelled images (found during
  inspection) and dish-name ambiguity (for example kadai paneer vs. chicken karahi). Some test labels are
  therefore wrong, which lowers measured accuracy by an unknown amount.
* **Single dish per photo.** Mixed plates (rice + salan + roti) return only the dominant dish. The rest
  must be added by search.
* **No portion estimation.** Grams come from the Portion Selector. Calories are only as good as the
  user's portion entry and the per-100 g value. The same dish varies widely between kitchens (oil,
  ghee, sugar).
* **Coverage gaps.** Common Pakistani foods with no usable public data are not recognised (list in
  2.3). Five classes have fewer than 160 images and wide per-class uncertainty.
* **Nutrition values.** 15 of 31 rows are approximate, and for seven dishes the databases disagree by 32-134%.
  The USDA FoodData Central licence statement could not be read in this session (the site renders it
  client-side) and should be checked before citing.
* **Domain of `not_food`.** Food-5K's non-food images are generic scenes. Close negatives (packaged
  snacks, menus, empty plates) are not represented, so `not_food` may not trigger on them.
* **Calibration transfer.** T and the band thresholds were fitted on web-photo validation data. On
  phone photos the model is likely to be less accurate at the same confidence, so thresholds should be
  re-fitted once a phone-photo validation set exists.

## 6. Disclaimer

SmartBite's calorie and nutrient values are estimates for general wellness tracking. They are not
medical or dietetic advice and must not be used to manage a medical condition.

## 7. References

[1] L. Bossard, M. Guillaumin, and L. Van Gool, "Food-101 – Mining discriminative components with random forests," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2014, pp. 446–461.
[2] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 6105–6114.
[3] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 4510–4520.
[4] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 1321–1330.
[5] R. Wightman, "PyTorch Image Models (timm)," GitHub repository, 2019. [Online]. Available: https://github.com/huggingface/pytorch-image-models
[6] A. Singla, L. Yuan, and T. Ebrahimi, "Food/non-food image classification and food categorization using pre-trained GoogLeNet model," in *Proc. 2nd Int. Workshop Multimedia Assisted Dietary Management (MADiMa)*, 2016, pp. 3–11.
[7] U.S. Department of Agriculture, Agricultural Research Service, "FoodData Central: Food and Nutrient Database for Dietary Studies 2021–2023," release 2024-10-31. [Online]. Available: https://fdc.nal.usda.gov/
[8] Office for Health Improvement and Disparities, "McCance and Widdowson's The Composition of Foods Integrated Dataset 2021," GOV.UK. [Online]. Available: https://www.gov.uk/government/publications/composition-of-foods-integrated-dataset-cofid
[9] "Indian Nutrient Databank (INDB)," GitHub repository lindsayjaacks/Indian-Nutrient-Databank-INDB-, commit fbdb62b, 2025. [Online]. Available: https://github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB-
[10] Kaggle datasets as listed in Section 2.2, accessed 2026-10-31: https://www.kaggle.com/datasets/{l33tc0d3r/indian-food-classification, anshulmehtakaggl/themassiveindianfooddataset, iamsouravbanerjee/indian-food-images-dataset, useractivated/dataset, cdart99/food20dataset, utkarshsaxenadn/fast-food-classification-dataset, trolukovich/food5k-image-dataset, kmader/food41}
[11] Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 4878–4887.
[12] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.
[13] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the Inception architecture for computer vision," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 2818–2826.
[14] C. Zauner, "Implementation and benchmarking of perceptual image hash functions," M.S. thesis, Upper Austria Univ. Appl. Sci., Hagenberg, Austria, 2010.
[15] ONNX Runtime developers, "ONNX Runtime," 2018. [Online]. Available: https://onnxruntime.ai/
