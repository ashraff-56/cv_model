# Plate-as-ruler benchmark (Nutrition5k test dishes, weighed)

Reference plate size (training photos): {'plate': 124.85464473121523, 'bowl': 97.15579928497853} px long-axis sigma; usable training plates {'plate': 463, 'bowl': 59} of 600 sampled.

| condition | photos | plate usable | constant | raw | plate | plate_any | plate - raw (g, 95% CI) |
|---|---|---|---|---|---|---|---|
| overhead | 503 | 87% | 116 g (58%) | 27 g (14%) | 52 g (26%) | 52 g (26%) | +24.4 [+20.7, +28.4] |
| side | 1798 | 28% | 115 g (58%) | 125 g (63%) | 129 g (65%) | 111 g (56%) | +4.6 [+1.8, +7.4] |
| zoom_0.6 | 503 | 100% | 116 g (58%) | 101 g (51%) | 50 g (25%) | 50 g (25%) | -50.5 [-59.5, -42.0] |
| zoom_0.8 | 503 | 100% | 116 g (58%) | 63 g (32%) | 50 g (25%) | 50 g (25%) | -12.9 [-19.0, -6.8] |
| zoom_1.25 | 503 | 28% | 116 g (58%) | 105 g (53%) | 110 g (55%) | 69 g (35%) | +4.4 [+2.4, +6.3] |
| side_A | 409 | 11% | 117 g (57%) | 164 g (79%) | 161 g (78%) | 110 g (53%) | -2.8 [-6.5, +0.3] |
| side_B | 464 | 43% | 116 g (59%) | 99 g (50%) | 115 g (58%) | 115 g (59%) | +16.0 [+10.6, +21.6] |
| side_C | 451 | 38% | 114 g (58%) | 96 g (49%) | 111 g (56%) | 118 g (60%) | +15.0 [+10.0, +20.4] |
| side_D | 474 | 17% | 115 g (59%) | 145 g (75%) | 133 g (68%) | 100 g (51%) | -12.2 [-17.6, -7.1] |
| side_plate_in_frame | 495 | 100% | 128 g (62%) | 107 g (52%) | 122 g (59%) | 122 g (59%) | +11.8 [+0.5, +22.5] |

MAE in grams (and as % of mean true mass). Negative plate - raw = the plate method is better.
