# Fruits-360 folder mapping audit

Checked against the current upstream GitHub `Training/` directory listings on 2026-11-18.

- Original-size source: [fruits-360-original-size](https://github.com/fruits-360/fruits-360-original-size)
- Supplemental 100x100 source: [fruits-360-100x100](https://github.com/fruits-360/fruits-360-100x100)
- License stated by upstream README: CC BY-SA 4.0 for both repositories.
- Mapped taxonomy classes: 30
- Mapped classes with at least one matching upstream folder: 30
- Mapped classes with no matching upstream folder: 0 (unmatched mappings were removed before this report)
- Fruit/vegetable classes intentionally left without a Fruits-360 mapping: watermelon and broccoli
- Other new taxonomy classes without a classifier source mapping are listed as collection gaps in `data/registry/class_coverage_v3.csv`

This verifies folder-name existence only. It does not count images, decode them, deduplicate them, or prove the labels are visually separable in phone photos.

## Mapped classes and folders

- **apple**: orig:Apple 10, orig:Apple 11, orig:Apple 12, orig:Apple 13, orig:Apple 14, orig:Apple 17, orig:Apple 18, orig:Apple 19, orig:Apple 20, orig:Apple 21, orig:Apple 22, orig:Apple 23, orig:Apple 5, orig:Apple 6, orig:Apple 7, orig:Apple 8, orig:Apple 9, orig:Apple Braeburn 1, orig:Apple Red Yellow 2, orig:apple_crimson_snow_1, orig:apple_golden_1, orig:apple_golden_2, orig:apple_golden_3, orig:apple_granny_smith_1, orig:apple_pink_lady_1, orig:apple_red_1, orig:apple_red_2, orig:apple_red_3, orig:apple_red_delicios_1, orig:apple_red_yellow_1, orig:apple_crimson_snow_1, orig:apple_golden_1, orig:apple_golden_2, orig:apple_golden_3, orig:apple_granny_smith_1, orig:apple_pink_lady_1, orig:apple_red_1, orig:apple_red_2, orig:apple_red_3, orig:apple_red_delicios_1, orig:apple_red_yellow_1
- **banana**: orig:Banana 2, orig:Banana 3, orig:Banana 4
- **orange**: orig:Orange 2, orig:Orange 3, orig:Orange 4, orig:Orange 5, orig:Orange peeled 1
- **mango**: low:Mango 1, low:Mango Red 1, low:Mangostan 1
- **grapes**: orig:Grape 1, orig:Grape pink 2, orig:Grape pink 3
- **strawberry**: orig:Strawberry 2, orig:Strawberry 3
- **guava**: low:Guava 1
- **dates**: orig:Dates 2
- **pomegranate**: low:Pomegranate 1
- **pear**: orig:Pear 1, orig:Pear 10, orig:Pear 11, orig:Pear 12, orig:Pear 13, orig:Pear 3, orig:Pear 5, orig:Pear 6, orig:Pear 7, orig:Pear 8, orig:Pear 9, orig:Pear common 1
- **peach**: orig:Peach 3, orig:Peach 4, orig:Peach 5, orig:Peach 6
- **pineapple**: low:Pineapple 1, low:Pineapple Mini 1
- **papaya**: orig:Papaya 2
- **avocado**: orig:Avocado Black 1, orig:Avocado Black 2, orig:Avocado Green 1
- **lemon**: low:Lemon 1, low:Lemon Meyer 1
- **coconut**: low:Cocos 1
- **kiwi**: orig:Kiwi 3, orig:kiwi 2, orig:Kiwi 3, orig:kiwi 2
- **plum**: orig:Plum 4, orig:Plum 5
- **apricot**: low:Apricot 1
- **potato**: low:Potato Red 1, low:Potato Red 2, low:Potato Sweet 1, low:Potato White 1
- **tomato**: orig:Tomato 1, orig:Tomato 10, orig:Tomato 5, orig:Tomato 7, orig:Tomato 8, orig:Tomato 9, orig:Tomato Cherry Maroon 1, orig:Tomato Cherry Orange 1, orig:Tomato Cherry Red 2, orig:Tomato Cherry Yellow 1, orig:Tomato Maroon 2
- **onion**: orig:Onion 2, orig:Onion Red 2, orig:Onion White 2
- **cucumber**: orig:Cucumber 1, orig:Cucumber 10, orig:Cucumber 11, orig:Cucumber 3, orig:Cucumber 4, orig:Cucumber 5, orig:Cucumber 6, orig:Cucumber 7, orig:Cucumber 8, orig:Cucumber 9
- **carrot**: orig:Carrot 1
- **cauliflower**: low:Cauliflower 1
- **cabbage**: orig:Cabbage red 1, orig:cabbage_white_1, orig:cabbage_white_1
- **eggplant**: orig:eggplant_long_1, low:Eggplant 1, low:Eggplant long 1, orig:eggplant_long_1
- **capsicum**: orig:Pepper 1, orig:Pepper 2, orig:Pepper Orange 2, orig:Pepper Red 2, orig:Pepper Red 3, orig:Pepper Red 4, orig:Pepper Red 5
- **garlic**: orig:garlic bulb 1, orig:garlic bulb 1
- **ginger**: orig:Ginger 1, orig:Ginger 2

## Missing source folders



## Domain limitation

Fruits-360 objects were filmed individually while rotating in front of a white background, then background pixels were replaced with white. This is a useful controlled recognition source but has a substantial domain gap from phone photos of food on plates, tables, in bags, or mixed into meals. Keep a separate phone-photo test set and do not claim phone-photo generalization from Fruits-360 metrics.
