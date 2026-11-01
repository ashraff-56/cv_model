# Plate-as-ruler benchmark (Nutrition5k test dishes, weighed)

Reference plate size (training photos): {'plate': 125.04500567333183, 'bowl': 97.3294363472302} px long-axis sigma; usable training plates {'plate': 463, 'bowl': 59} of 600 sampled.

| condition | photos | plate usable | constant | raw | plate | plate_any | plate - raw (g, 95% CI) |
|---|---|---|---|---|---|---|---|
| overhead | 503 | 87% | 116 g (58%) | 27 g (14%) | 52 g (26%) | 52 g (26%) | +24.4 [+20.6, +28.3] |
| side | 1798 | 31% | 115 g (58%) | 139 g (70%) | 146 g (73%) | 118 g (60%) | +7.0 [+4.0, +9.9] |
| zoom_0.6 | 503 | 100% | 116 g (58%) | 101 g (51%) | 50 g (25%) | 50 g (25%) | -50.6 [-59.5, -42.1] |
| zoom_0.8 | 503 | 100% | 116 g (58%) | 63 g (32%) | 50 g (25%) | 50 g (25%) | -12.9 [-19.0, -6.8] |
| zoom_1.25 | 503 | 28% | 116 g (58%) | 105 g (53%) | 110 g (55%) | 69 g (35%) | +4.5 [+2.4, +6.4] |
| side_A | 409 | 7% | 117 g (57%) | 177 g (86%) | 175 g (85%) | 113 g (54%) | -2.2 [-6.5, +1.1] |
| side_B | 464 | 55% | 116 g (59%) | 107 g (54%) | 127 g (65%) | 123 g (63%) | +20.4 [+14.5, +26.6] |
| side_C | 451 | 43% | 114 g (58%) | 103 g (53%) | 122 g (62%) | 128 g (65%) | +18.5 [+13.5, +24.1] |
| side_D | 474 | 18% | 115 g (59%) | 171 g (88%) | 161 g (83%) | 109 g (56%) | -9.9 [-14.5, -5.3] |
| side_plate_in_frame | 562 | 100% | 128 g (67%) | 107 g (56%) | 129 g (67%) | 129 g (67%) | +15.3 [+5.5, +25.2] |

MAE in grams (and as % of mean true mass). Negative plate - raw = the plate method is better.
