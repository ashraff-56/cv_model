# Data preparation report

Seed 20261031; near-duplicate threshold pHash Hamming <= 6/64 (image or mirror); cap 600/class; min 80/class.

| class | raw files | unreadable/small | duplicates removed | in label conflicts | train | val | test |
|---|---:|---:|---:|---:|---:|---:|---:|
| biryani | 879 | 0 | 95 | 4 | 420 | 90 | 90 |
| naan | 1475 | 0 | 190 | 25 | 420 | 90 | 90 |
| roti_chapati | 1293 | 0 | 241 | 32 | 420 | 90 | 90 |
| paratha | 344 | 0 | 47 | 29 | 188 | 40 | 40 |
| aloo_paratha | 644 | 0 | 75 | 21 | 384 | 82 | 82 |
| anda_paratha | 244 | 0 | 64 | 3 | 124 | 26 | 27 |
| sheermal | 246 | 0 | 31 | 28 | 131 | 28 | 28 |
| puri | 340 | 0 | 19 | 41 | 196 | 42 | 42 |
| halwa_puri | 247 | 0 | 82 | 23 | 99 | 21 | 22 |
| paratha_roll | 833 | 0 | 228 | 1 | 420 | 90 | 90 |
| samosa | 2105 | 0 | 307 | 2 | 420 | 90 | 90 |
| pakora | 1194 | 0 | 221 | 8 | 420 | 90 | 90 |
| gol_gappay | 953 | 0 | 178 | 12 | 420 | 90 | 90 |
| chana_chaat | 244 | 0 | 68 | 0 | 126 | 25 | 25 |
| karahi | 488 | 69 | 112 | 5 | 211 | 45 | 46 |
| korma | 487 | 146 | 145 | 0 | 137 | 29 | 30 |
| chicken_curry | 2062 | 0 | 103 | 25 | 420 | 90 | 90 |
| nihari | 244 | 40 | 65 | 1 | 97 | 20 | 21 |
| kunna | 243 | 65 | 100 | 0 | 0 | 0 | 0 |
| paye | 247 | 55 | 110 | 0 | 57 | 12 | 13 |
| haleem | 346 | 0 | 102 | 7 | 166 | 35 | 36 |
| aloo_gosht | 139 | 60 | 57 | 0 | 0 | 0 | 0 |
| palak_gosht | 244 | 0 | 81 | 0 | 114 | 24 | 25 |
| keema | 242 | 24 | 70 | 2 | 102 | 22 | 22 |
| kofta | 906 | 26 | 452 | 1 | 299 | 64 | 64 |
| fish_curry | 646 | 33 | 125 | 3 | 340 | 72 | 73 |
| fried_fish | 239 | 0 | 61 | 19 | 111 | 24 | 24 |
| chicken_tikka | 494 | 0 | 172 | 23 | 209 | 45 | 45 |
| malai_boti | 241 | 0 | 97 | 4 | 98 | 21 | 21 |
| seekh_kebab | 644 | 0 | 124 | 5 | 360 | 77 | 78 |
| chapli_kebab | 244 | 0 | 101 | 1 | 99 | 21 | 22 |
| lamb_chops | 241 | 0 | 83 | 5 | 107 | 23 | 23 |
| chicken_roast | 247 | 0 | 75 | 18 | 114 | 20 | 20 |
| daal | 682 | 0 | 96 | 0 | 410 | 88 | 88 |
| chana_chole | 1587 | 0 | 291 | 9 | 420 | 90 | 90 |
| kadhi_pakora | 237 | 0 | 69 | 4 | 115 | 24 | 25 |
| aloo_baingan | 243 | 0 | 110 | 3 | 91 | 19 | 20 |
| aloo_gobi | 643 | 0 | 146 | 5 | 344 | 74 | 74 |
| aloo_methi | 369 | 1 | 150 | 6 | 148 | 32 | 32 |
| aloo_palak | 242 | 1 | 132 | 8 | 71 | 15 | 15 |
| aloo_shimla_mirch | 245 | 2 | 78 | 3 | 113 | 24 | 25 |
| bhindi | 650 | 1 | 191 | 14 | 311 | 66 | 67 |
| ghiya | 244 | 1 | 87 | 0 | 109 | 23 | 24 |
| karela | 244 | 0 | 66 | 0 | 125 | 26 | 27 |
| mixed_sabzi | 241 | 0 | 82 | 4 | 108 | 23 | 24 |
| saag | 246 | 0 | 61 | 0 | 130 | 27 | 28 |
| raita | 246 | 0 | 108 | 8 | 91 | 19 | 20 |
| kachumber_salad | 247 | 0 | 106 | 20 | 85 | 18 | 18 |
| achar | 244 | 0 | 84 | 0 | 112 | 24 | 24 |
| pudina_chutney | 244 | 0 | 114 | 4 | 93 | 16 | 17 |
| gulab_jamun | 742 | 0 | 91 | 8 | 420 | 90 | 90 |
| jalebi | 1053 | 0 | 237 | 6 | 420 | 90 | 90 |
| gajar_halwa | 650 | 0 | 125 | 13 | 358 | 77 | 77 |
| suji_halwa | 297 | 0 | 66 | 8 | 156 | 33 | 34 |
| barfi | 642 | 0 | 85 | 12 | 382 | 81 | 82 |
| rabri | 241 | 0 | 50 | 16 | 122 | 26 | 27 |
| ras_malai | 243 | 0 | 58 | 5 | 126 | 27 | 27 |
| kheer | 753 | 0 | 35 | 10 | 420 | 90 | 90 |
| kulfi | 636 | 0 | 31 | 3 | 420 | 90 | 90 |
| falooda | 400 | 0 | 12 | 3 | 270 | 57 | 58 |
| chai | 722 | 0 | 148 | 1 | 401 | 86 | 86 |
| coffee | 244 | 0 | 72 | 0 | 120 | 26 | 26 |
| lassi | 644 | 0 | 204 | 28 | 288 | 62 | 62 |
| lemonade | 238 | 0 | 76 | 1 | 113 | 24 | 24 |
| milk_soda | 197 | 0 | 134 | 35 | 0 | 0 | 0 |
| sattu | 243 | 104 | 89 | 0 | 0 | 0 | 0 |
| pizza | 4282 | 0 | 330 | 17 | 420 | 90 | 90 |
| burger | 2809 | 0 | 134 | 18 | 420 | 90 | 90 |
| french_fries | 2850 | 1 | 667 | 6 | 420 | 90 | 90 |
| fried_rice | 1455 | 0 | 56 | 1 | 420 | 90 | 90 |
| omelette | 1100 | 0 | 18 | 0 | 420 | 90 | 90 |
| fried_chicken | 2900 | 1 | 110 | 9 | 420 | 90 | 90 |
| sandwich | 2900 | 1 | 121 | 14 | 420 | 90 | 90 |
| hot_dog | 2500 | 0 | 361 | 3 | 420 | 90 | 90 |
| donut | 2500 | 0 | 100 | 7 | 420 | 90 | 90 |
| ice_cream | 1000 | 0 | 0 | 0 | 420 | 90 | 90 |
| chocolate_cake | 1000 | 0 | 2 | 0 | 420 | 90 | 90 |
| spaghetti | 2000 | 0 | 348 | 2 | 420 | 90 | 90 |
| salad | 2241 | 0 | 87 | 11 | 420 | 90 | 90 |
| not_food | 2500 | 0 | 25 | 1 | 420 | 90 | 90 |
| **total** | **66521** | **631** | **9924** | **674** | **20001** | **4265** | **4290** |

Label-conflict clusters dropped: 225
Classes dropped for having too few images: {'kunna': 78, 'aloo_gosht': 22, 'milk_soda': 28, 'sattu': 50}

## Images kept per source

| class | dataset | raw | kept |
|---|---|---:|---:|
| achar | smf | 244 | 160 |
| aloo_baingan | smf | 243 | 130 |
| aloo_gobi | khana | 400 | 335 |
| aloo_gobi | smf | 243 | 157 |
| aloo_gosht | smf | 139 | 0 |
| aloo_methi | khana | 125 | 84 |
| aloo_methi | smf | 244 | 128 |
| aloo_palak | smf | 242 | 101 |
| aloo_paratha | khana | 400 | 394 |
| aloo_paratha | smf | 244 | 154 |
| aloo_shimla_mirch | smf | 245 | 162 |
| anda_paratha | smf | 244 | 177 |
| barfi | khana | 400 | 349 |
| barfi | smf | 242 | 196 |
| bhindi | khana | 400 | 276 |
| bhindi | smf | 250 | 168 |
| biryani | anshul | 229 | 144 |
| biryani | food20 | 100 | 76 |
| biryani | khana | 400 | 305 |
| biryani | pkdishes | 100 | 50 |
| biryani | sourav80 | 50 | 25 |
| burger | fastfood | 1500 | 328 |
| burger | food101 | 1000 | 211 |
| burger | l33t20 | 309 | 61 |
| chai | l33t20 | 381 | 363 |
| chai | pkdishes | 100 | 70 |
| chai | smf | 241 | 140 |
| chana_chaat | smf | 244 | 176 |
| chana_chole | anshul | 326 | 99 |
| chana_chole | khana | 800 | 292 |
| chana_chole | l33t20 | 411 | 181 |
| chana_chole | sourav80 | 50 | 28 |
| chapli_kebab | smf | 244 | 142 |
| chicken_curry | food101 | 1000 | 322 |
| chicken_curry | khana | 400 | 118 |
| chicken_curry | l33t20 | 412 | 104 |
| chicken_curry | pkdishes | 100 | 25 |
| chicken_curry | sourav80 | 150 | 31 |
| chicken_roast | smf | 247 | 154 |
| chicken_tikka | food20 | 100 | 60 |
| chicken_tikka | pkdishes | 100 | 74 |
| chicken_tikka | smf | 244 | 144 |
| chicken_tikka | sourav80 | 50 | 21 |
| chocolate_cake | food101 | 1000 | 600 |
| coffee | smf | 244 | 172 |
| daal | anshul | 261 | 229 |
| daal | l33t20 | 321 | 288 |
| daal | sourav80 | 100 | 69 |
| donut | fastfood | 1500 | 354 |
| donut | food101 | 1000 | 246 |
| falooda | khana | 400 | 385 |
| fish_curry | khana | 400 | 349 |
| fish_curry | smf | 246 | 136 |
| french_fries | fastfood | 1500 | 310 |
| french_fries | food101 | 1000 | 272 |
| french_fries | pkdishes | 100 | 18 |
| french_fries | smf | 250 | 0 |
| fried_chicken | fastfood | 1500 | 301 |
| fried_chicken | food101 | 1000 | 207 |
| fried_chicken | khana | 400 | 92 |
| fried_fish | smf | 239 | 159 |
| fried_rice | food101 | 1000 | 425 |
| fried_rice | l33t20 | 355 | 147 |
| fried_rice | pkdishes | 100 | 28 |
| gajar_halwa | khana | 357 | 294 |
| gajar_halwa | smf | 243 | 178 |
| gajar_halwa | sourav80 | 50 | 40 |
| ghiya | smf | 244 | 156 |
| gol_gappay | anshul | 177 | 113 |
| gol_gappay | khana | 400 | 331 |
| gol_gappay | l33t20 | 130 | 108 |
| gol_gappay | smf | 246 | 48 |
| gulab_jamun | food20 | 100 | 75 |
| gulab_jamun | khana | 400 | 361 |
| gulab_jamun | smf | 192 | 129 |
| gulab_jamun | sourav80 | 50 | 35 |
| haleem | pkdishes | 100 | 76 |
| haleem | smf | 246 | 161 |
| halwa_puri | smf | 247 | 142 |
| hot_dog | fastfood | 1500 | 339 |
| hot_dog | food101 | 1000 | 261 |
| ice_cream | food101 | 1000 | 600 |
| jalebi | anshul | 306 | 142 |
| jalebi | khana | 400 | 241 |
| jalebi | l33t20 | 297 | 200 |
| jalebi | sourav80 | 50 | 17 |
| kachumber_salad | smf | 247 | 121 |
| kadhi_pakora | smf | 237 | 164 |
| karahi | smf | 488 | 302 |
| karela | smf | 244 | 178 |
| keema | smf | 242 | 146 |
| kheer | khana | 753 | 600 |
| kofta | anshul | 667 | 303 |
| kofta | smf | 239 | 124 |
| korma | smf | 487 | 196 |
| kulfi | khana | 400 | 385 |
| kulfi | l33t20 | 236 | 215 |
| kunna | smf | 243 | 0 |
| lamb_chops | smf | 241 | 153 |
| lassi | khana | 400 | 332 |
| lassi | smf | 244 | 80 |
| lemonade | smf | 238 | 161 |
| malai_boti | smf | 241 | 140 |
| milk_soda | smf | 197 | 0 |
| mixed_sabzi | smf | 241 | 155 |
| naan | anshul | 353 | 152 |
| naan | food20 | 100 | 44 |
| naan | khana | 400 | 186 |
| naan | l33t20 | 329 | 157 |
| naan | smf | 243 | 51 |
| naan | sourav80 | 50 | 10 |
| nihari | smf | 244 | 138 |
| not_food | food5k | 2500 | 600 |
| omelette | food101 | 1000 | 548 |
| omelette | pkdishes | 100 | 52 |
| pakora | anshul | 377 | 223 |
| pakora | khana | 292 | 162 |
| pakora | l33t20 | 278 | 179 |
| pakora | smf | 247 | 36 |
| palak_gosht | smf | 244 | 163 |
| paratha | pkdishes | 100 | 77 |
| paratha | smf | 244 | 191 |
| paratha_roll | anshul | 340 | 226 |
| paratha_roll | food20 | 100 | 25 |
| paratha_roll | l33t20 | 293 | 265 |
| paratha_roll | pkdishes | 100 | 84 |
| paye | smf | 247 | 82 |
| pizza | fastfood | 1500 | 225 |
| pizza | food101 | 1000 | 142 |
| pizza | khana | 1521 | 202 |
| pizza | l33t20 | 261 | 31 |
| pudina_chutney | smf | 244 | 126 |
| puri | food20 | 100 | 85 |
| puri | smf | 240 | 195 |
| rabri | smf | 241 | 175 |
| raita | smf | 246 | 130 |
| ras_malai | smf | 243 | 180 |
| roti_chapati | food20 | 100 | 26 |
| roti_chapati | khana | 400 | 188 |
| roti_chapati | l33t20 | 413 | 227 |
| roti_chapati | pkdishes | 100 | 23 |
| roti_chapati | smf | 230 | 125 |
| roti_chapati | sourav80 | 50 | 11 |
| saag | smf | 246 | 185 |
| salad | food101 | 2000 | 584 |
| salad | smf | 241 | 16 |
| samosa | food101 | 1000 | 356 |
| samosa | food20 | 100 | 12 |
| samosa | khana | 400 | 122 |
| samosa | l33t20 | 262 | 85 |
| samosa | pkdishes | 100 | 25 |
| samosa | smf | 243 | 0 |
| sandwich | fastfood | 1500 | 309 |
| sandwich | food101 | 1000 | 208 |
| sandwich | khana | 400 | 83 |
| sattu | smf | 243 | 0 |
| seekh_kebab | khana | 400 | 374 |
| seekh_kebab | smf | 244 | 141 |
| sheermal | smf | 246 | 187 |
| spaghetti | food101 | 2000 | 600 |
| suji_halwa | smf | 247 | 180 |
| suji_halwa | sourav80 | 50 | 43 |
