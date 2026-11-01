# Plate-relative portion model: Nutrition5k test dishes (weighed)

| condition | photos | plate found | plate in frame | same guess for all | plate model (new) | first model, raw photo | first model + plate scaling | new - raw (g, 95% CI) |
|---|---|---|---|---|---|---|---|---|
| overhead | 503 | 100% | 87% | 116 g (58%) | 28 g (14%) | 27 g (14%) | 52 g (26%) | +0.8 [-1.3, +2.9] |
| side | 1798 | 99% | 31% | 115 g (58%) | 55 g (27%) | 139 g (70%) | 146 g (73%) | -85.4 [-92.3, -78.6] |
| zoom_0.6 | 503 | 100% | 100% | 116 g (58%) | 29 g (15%) | 101 g (51%) | 50 g (25%) | -71.7 [-80.2, -63.7] |
| zoom_0.8 | 503 | 100% | 100% | 116 g (58%) | 28 g (14%) | 63 g (32%) | 50 g (25%) | -34.9 [-40.2, -29.8] |
| zoom_1.25 | 503 | 100% | 28% | 116 g (58%) | 46 g (23%) | 105 g (53%) | 110 g (55%) | -59.3 [-65.5, -53.8] |
| side_A | 409 | 100% | 7% | 117 g (57%) | 57 g (28%) | 177 g (86%) | 175 g (85%) | -119.9 [-130.9, -109.7] |
| side_B | 464 | 100% | 55% | 116 g (59%) | 50 g (26%) | 107 g (54%) | 127 g (65%) | -56.3 [-64.9, -48.5] |
| side_C | 451 | 99% | 43% | 114 g (58%) | 52 g (26%) | 103 g (53%) | 122 g (62%) | -51.6 [-59.7, -43.5] |
| side_D | 474 | 99% | 18% | 115 g (59%) | 59 g (30%) | 171 g (88%) | 161 g (83%) | -112.4 [-123.2, -101.5] |
| side_plate_in_frame | 562 | 100% | 100% | 128 g (67%) | 41 g (22%) | 107 g (56%) | 129 g (67%) | -65.0 [-74.1, -55.7] |

Mean absolute error in grams (and as % of mean true mass). CIs resample dishes. Negative difference = plate model (new) is better.
