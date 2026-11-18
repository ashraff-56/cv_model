# v3 class source gaps and recommended next source work

This report identifies new taxonomy classes that do not yet have a mapped image-folder source in `classes_v3.yaml`.

**Gap count: 41 classes.** They remain planned classes, not training-ready classes.

## Classes with no classifier source mapping

| class | category | Find a licensed dedicated dataset or collect real phone photographs |
|---|---|---|
| watermelon | fruit | FoodSeg103 has a corresponding mask label; verify class mapping and use for segmentation/crop supervision |
| spinach | vegetable | Find a licensed dedicated dataset or collect real phone photographs |
| peas | vegetable | Find a licensed dedicated dataset or collect real phone photographs |
| broccoli | vegetable | FoodSeg103 has a corresponding mask label; verify class mapping and use for segmentation/crop supervision |
| lettuce | vegetable | FoodSeg103 has a corresponding mask label; verify class mapping and use for segmentation/crop supervision |
| sugar | ingredient | Find a licensed dedicated dataset or collect real phone photographs |
| salt | ingredient | Find a licensed dedicated dataset or collect real phone photographs |
| wheat_flour | ingredient | Find a licensed dedicated dataset or collect real phone photographs |
| white_rice | ingredient | FoodSeg103 has a broader/merged related mask label; do not treat as exact class without review |
| brown_rice | ingredient | Find a licensed dedicated dataset or collect real phone photographs |
| oats | ingredient | Find a licensed dedicated dataset or collect real phone photographs |
| red_lentils | ingredient | Find a licensed dedicated dataset or collect real phone photographs |
| chickpeas | ingredient | Find a licensed dedicated dataset or collect real phone photographs |
| kidney_beans | ingredient | Find a licensed dedicated dataset or collect real phone photographs |
| black_lentils | ingredient | Find a licensed dedicated dataset or collect real phone photographs |
| cooking_oil | ingredient | Find a licensed dedicated dataset or collect real phone photographs |
| ghee | ingredient | Find a licensed dedicated dataset or collect real phone photographs |
| butter | ingredient | Find a licensed dedicated dataset or collect real phone photographs |
| honey | ingredient | Find a licensed dedicated dataset or collect real phone photographs |
| peanut_butter | ingredient | Find a licensed dedicated dataset or collect real phone photographs |
| biscuits | packaged_snack | FoodSeg103 has a broader/merged related mask label; do not treat as exact class without review |
| crackers | packaged_snack | Find a licensed dedicated dataset or collect real phone photographs |
| cookies | packaged_snack | Find a licensed dedicated dataset or collect real phone photographs |
| toast | packaged_snack | Find a licensed dedicated dataset or collect real phone photographs |
| croissant | packaged_snack | Find a licensed dedicated dataset or collect real phone photographs |
| cake_slice | packaged_snack | FoodSeg103 has a broader/merged related mask label; do not treat as exact class without review |
| potato_chips | packaged_snack | Find a licensed dedicated dataset or collect real phone photographs |
| popcorn | packaged_snack | FoodSeg103 has a corresponding mask label; verify class mapping and use for segmentation/crop supervision |
| chocolate_bar | packaged_snack | FoodSeg103 has a broader/merged related mask label; do not treat as exact class without review |
| breakfast_cereal | packaged_snack | Find a licensed dedicated dataset or collect real phone photographs |
| plain_milk | dairy_protein_beverage | FoodSeg103 has a corresponding mask label; verify class mapping and use for segmentation/crop supervision |
| plain_yogurt | dairy_protein_beverage | Find a licensed dedicated dataset or collect real phone photographs |
| cheese | dairy_protein_beverage | FoodSeg103 has a broader/merged related mask label; do not treat as exact class without review |
| boiled_egg | dairy_protein_beverage | FoodSeg103 has a corresponding mask label; verify class mapping and use for segmentation/crop supervision |
| beef_steak | dairy_protein_beverage | FoodSeg103 has a corresponding mask label; verify class mapping and use for segmentation/crop supervision |
| orange_juice | dairy_protein_beverage | FoodSeg103 has a broader/merged related mask label; do not treat as exact class without review |
| tacos | international_dish | Find a licensed dedicated dataset or collect real phone photographs |
| sushi | international_dish | Find a licensed dedicated dataset or collect real phone photographs |
| dumplings | international_dish | FoodSeg103 has a broader/merged related mask label; do not treat as exact class without review |
| pancakes | international_dish | Find a licensed dedicated dataset or collect real phone photographs |
| lasagna | international_dish | Find a licensed dedicated dataset or collect real phone photographs |

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
