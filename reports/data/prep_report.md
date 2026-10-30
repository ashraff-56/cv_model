# Data preparation report

Seed 20261031; near-duplicate threshold pHash Hamming <= 6/64 (image or mirror); cap 600/class; min 80/class.

| class | raw files | unreadable/small | duplicates removed | in label conflicts | train | val | test |
|---|---:|---:|---:|---:|---:|---:|---:|
| biryani | 479 | 0 | 84 | 0 | 276 | 59 | 60 |
| samosa | 1462 | 0 | 121 | 0 | 420 | 90 | 90 |
| naan | 832 | 0 | 83 | 3 | 420 | 90 | 90 |
| roti_chapati | 663 | 0 | 159 | 0 | 353 | 76 | 75 |
| paratha | 100 | 0 | 15 | 0 | 59 | 13 | 13 |
| daal | 682 | 0 | 96 | 0 | 410 | 88 | 88 |
| curry_karahi | 1662 | 0 | 97 | 19 | 420 | 90 | 90 |
| haleem | 100 | 0 | 18 | 0 | 57 | 12 | 13 |
| chicken_tikka | 250 | 0 | 77 | 18 | 108 | 23 | 24 |
| pakora | 655 | 0 | 52 | 3 | 420 | 90 | 90 |
| chana_chole | 787 | 0 | 129 | 0 | 420 | 90 | 90 |
| jalebi | 653 | 0 | 161 | 0 | 344 | 74 | 74 |
| gulab_jamun | 150 | 0 | 26 | 0 | 87 | 19 | 18 |
| chai | 481 | 0 | 47 | 0 | 304 | 65 | 65 |
| paratha_roll | 833 | 0 | 228 | 0 | 420 | 90 | 90 |
| gol_gappay | 307 | 0 | 46 | 0 | 183 | 39 | 39 |
| halwa | 200 | 0 | 45 | 1 | 108 | 23 | 23 |
| kulfi | 236 | 0 | 18 | 0 | 153 | 33 | 32 |
| pizza | 2761 | 0 | 154 | 1 | 420 | 90 | 90 |
| burger | 2809 | 0 | 134 | 18 | 420 | 90 | 90 |
| french_fries | 2600 | 1 | 487 | 5 | 420 | 90 | 90 |
| fried_rice | 1455 | 0 | 56 | 0 | 420 | 90 | 90 |
| omelette | 1100 | 0 | 18 | 0 | 420 | 90 | 90 |
| fried_chicken | 2500 | 1 | 90 | 6 | 420 | 90 | 90 |
| sandwich | 2500 | 1 | 91 | 10 | 420 | 90 | 90 |
| hot_dog | 2500 | 0 | 361 | 2 | 420 | 90 | 90 |
| donut | 2500 | 0 | 104 | 1 | 420 | 90 | 90 |
| ice_cream | 1000 | 0 | 0 | 0 | 420 | 90 | 90 |
| chocolate_cake | 1000 | 0 | 2 | 0 | 420 | 90 | 90 |
| spaghetti | 2000 | 0 | 348 | 0 | 420 | 90 | 90 |
| salad | 2000 | 0 | 4 | 0 | 420 | 90 | 90 |
| not_food | 2500 | 0 | 25 | 1 | 420 | 90 | 90 |
| **total** | **39757** | **3** | **3376** | **88** | **10842** | **2324** | **2324** |

Label-conflict clusters dropped: 38
Classes dropped for having too few images: none

## Images kept per source

| class | dataset | raw | kept |
|---|---|---:|---:|
| biryani | anshul | 229 | 195 |
| biryani | food20 | 100 | 97 |
| biryani | pkdishes | 100 | 65 |
| biryani | sourav80 | 50 | 38 |
| burger | fastfood | 1500 | 328 |
| burger | food101 | 1000 | 211 |
| burger | l33t20 | 309 | 61 |
| chai | l33t20 | 381 | 364 |
| chai | pkdishes | 100 | 70 |
| chana_chole | anshul | 326 | 202 |
| chana_chole | l33t20 | 411 | 354 |
| chana_chole | sourav80 | 50 | 44 |
| chicken_tikka | food20 | 100 | 60 |
| chicken_tikka | pkdishes | 100 | 74 |
| chicken_tikka | sourav80 | 50 | 21 |
| chocolate_cake | food101 | 1000 | 600 |
| curry_karahi | food101 | 1000 | 398 |
| curry_karahi | l33t20 | 412 | 136 |
| curry_karahi | pkdishes | 100 | 27 |
| curry_karahi | sourav80 | 150 | 39 |
| daal | anshul | 261 | 229 |
| daal | l33t20 | 321 | 288 |
| daal | sourav80 | 100 | 69 |
| donut | fastfood | 1500 | 354 |
| donut | food101 | 1000 | 246 |
| french_fries | fastfood | 1500 | 310 |
| french_fries | food101 | 1000 | 272 |
| french_fries | pkdishes | 100 | 18 |
| fried_chicken | fastfood | 1500 | 355 |
| fried_chicken | food101 | 1000 | 245 |
| fried_rice | food101 | 1000 | 425 |
| fried_rice | l33t20 | 355 | 147 |
| fried_rice | pkdishes | 100 | 28 |
| gol_gappay | anshul | 177 | 134 |
| gol_gappay | l33t20 | 130 | 127 |
| gulab_jamun | food20 | 100 | 85 |
| gulab_jamun | sourav80 | 50 | 39 |
| haleem | pkdishes | 100 | 82 |
| halwa | food20 | 100 | 69 |
| halwa | sourav80 | 100 | 85 |
| hot_dog | fastfood | 1500 | 339 |
| hot_dog | food101 | 1000 | 261 |
| ice_cream | food101 | 1000 | 600 |
| jalebi | anshul | 306 | 198 |
| jalebi | l33t20 | 297 | 272 |
| jalebi | sourav80 | 50 | 22 |
| kulfi | l33t20 | 236 | 218 |
| naan | anshul | 353 | 247 |
| naan | food20 | 100 | 84 |
| naan | l33t20 | 329 | 251 |
| naan | sourav80 | 50 | 18 |
| not_food | food5k | 2500 | 600 |
| omelette | food101 | 1000 | 548 |
| omelette | pkdishes | 100 | 52 |
| pakora | anshul | 377 | 335 |
| pakora | l33t20 | 278 | 265 |
| paratha | pkdishes | 100 | 85 |
| paratha_roll | anshul | 340 | 226 |
| paratha_roll | food20 | 100 | 25 |
| paratha_roll | l33t20 | 293 | 265 |
| paratha_roll | pkdishes | 100 | 84 |
| pizza | fastfood | 1500 | 335 |
| pizza | food101 | 1000 | 217 |
| pizza | l33t20 | 261 | 48 |
| roti_chapati | food20 | 100 | 47 |
| roti_chapati | l33t20 | 413 | 387 |
| roti_chapati | pkdishes | 100 | 52 |
| roti_chapati | sourav80 | 50 | 18 |
| salad | food101 | 2000 | 600 |
| samosa | food101 | 1000 | 440 |
| samosa | food20 | 100 | 15 |
| samosa | l33t20 | 262 | 108 |
| samosa | pkdishes | 100 | 37 |
| sandwich | fastfood | 1500 | 357 |
| sandwich | food101 | 1000 | 243 |
| spaghetti | food101 | 2000 | 600 |
