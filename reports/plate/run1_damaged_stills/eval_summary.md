# Plate-relative portion model: Nutrition5k test dishes (weighed)

| condition | photos | plate found | plate in frame | same guess for all | plate model (new) | first model, raw photo | first model + plate scaling | new - raw (g, 95% CI) |
|---|---|---|---|---|---|---|---|---|
| overhead | 503 | 100% | 87% | 116 g (58%) | 30 g (15%) | 27 g (14%) | 52 g (26%) | +2.4 [+0.5, +4.4] |
| side | 1798 | 90% | 28% | 115 g (58%) | 60 g (30%) | 125 g (63%) | 129 g (65%) | -64.3 [-70.8, -57.9] |
| zoom_0.6 | 503 | 100% | 100% | 116 g (58%) | 31 g (15%) | 101 g (51%) | 50 g (25%) | -70.2 [-78.1, -62.6] |
| zoom_0.8 | 503 | 100% | 100% | 116 g (58%) | 30 g (15%) | 63 g (32%) | 50 g (25%) | -32.9 [-37.9, -27.9] |
| zoom_1.25 | 503 | 100% | 28% | 116 g (58%) | 38 g (19%) | 105 g (53%) | 110 g (55%) | -67.0 [-72.9, -61.3] |
| side_A | 409 | 95% | 11% | 117 g (57%) | 60 g (29%) | 164 g (79%) | 161 g (78%) | -104.5 [-116.0, -93.6] |
| side_B | 464 | 86% | 43% | 116 g (59%) | 61 g (31%) | 99 g (50%) | 115 g (58%) | -38.2 [-47.0, -29.5] |
| side_C | 451 | 89% | 38% | 114 g (58%) | 60 g (31%) | 96 g (49%) | 111 g (56%) | -35.2 [-44.0, -26.7] |
| side_D | 474 | 92% | 17% | 115 g (59%) | 61 g (31%) | 145 g (75%) | 133 g (68%) | -84.6 [-95.3, -73.6] |
| side_plate_in_frame | 495 | 100% | 100% | 128 g (62%) | 50 g (24%) | 107 g (52%) | 122 g (59%) | -58.2 [-68.5, -48.5] |

Mean absolute error in grams (and as % of mean true mass). CIs resample dishes. Negative difference = plate model (new) is better.
