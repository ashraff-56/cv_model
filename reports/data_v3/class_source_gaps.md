# v3 class source gaps and recommended next source work

This report identifies new taxonomy classes that do not yet have a mapped image-folder source in `classes_v3.yaml`.

**Gap count: 41 classes.** They remain planned classes, not training-ready classes.

## Classes with no classifier source mapping

| class | category | recommended next action |
|---|---|---|
| watermelon | fruit | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| spinach | vegetable | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| peas | vegetable | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| broccoli | vegetable | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| lettuce | vegetable | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| sugar | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| salt | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| wheat_flour | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| white_rice | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| brown_rice | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| oats | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| red_lentils | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| chickpeas | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| kidney_beans | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| black_lentils | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| cooking_oil | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| ghee | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| butter | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| honey | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| peanut_butter | ingredient | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| biscuits | packaged_snack | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| crackers | packaged_snack | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| cookies | packaged_snack | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| toast | packaged_snack | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| croissant | packaged_snack | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| cake_slice | packaged_snack | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| potato_chips | packaged_snack | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| popcorn | packaged_snack | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| chocolate_bar | packaged_snack | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| breakfast_cereal | packaged_snack | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| plain_milk | dairy_protein_beverage | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| plain_yogurt | dairy_protein_beverage | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| cheese | dairy_protein_beverage | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| boiled_egg | dairy_protein_beverage | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| beef_steak | dairy_protein_beverage | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| orange_juice | dairy_protein_beverage | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| tacos | international_dish | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| sushi | international_dish | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| dumplings | international_dish | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| pancakes | international_dish | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |
| lasagna | international_dish | Check FoodSeg103 mask labels; use only as mask-supervised/crop-derived data after review |

## How to use FoodSeg103 correctly

FoodSeg103 has multi-ingredient images with pixel masks, not a simple one-image/one-class folder structure. Its labels include several classes relevant to these gaps, such as biscuit, chocolate, popcorn, apple, banana, dates, mango, watermelon, rice, bread, egg, chicken, steak, fish, potato, eggplant, garlic, cauliflower, tomato, ginger, lettuce, cucumber, carrot, broccoli, cabbage, onion, pepper and others.

Use it primarily for:
1. Food/ingredient segmentation pretraining.
2. Extracting reviewed, mask-derived object crops for selected classifier classes.
3. Training/evaluating a multi-item segmentation component.

Do not assign the entire mixed-meal image a single ingredient label. Do not assume that the mask class and the visible food's culinary identity always align perfectly.

## Priority for custom collection

The following are visually difficult or poorly represented by ordinary web datasets and should be collected under controlled capture instructions:
- Salt, sugar, flour and other powders: use user confirmation when appearance is ambiguous.
- Oil, ghee, butter and honey: containers, packaging and liquid presentation vary.
- Pakistani local dishes and snacks absent from v2: collect multiple household/restaurant preparations.
- Packaged products: capture both packaging and unpackaged contents; do not let brand packaging become the only cue.

A class only becomes training-ready after the raw source audit, image review, duplicate checks, source-license review and phone-photo holdout coverage.
