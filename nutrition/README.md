# Nutrition table

`nutrition.csv` has per-100 g energy (kcal), protein, carbohydrate and fat, and a default serving, for
every food class. `tools/build_nutrition.py` generates it from the databases below using the choices
in `sources.yaml`. No value is typed by hand. `calories(label, grams) = kcal_per_100g x grams / 100`.

## Sources

| Database | Version used | Download | Licence (as found) |
|---|---|---|---|
| USDA FoodData Central, FNDDS (survey foods) | FNDDS 2021-2023, FDC release 2024-10-31 (`FoodData_Central_survey_food_csv_2024-10-31.zip`) | https://fdc.nal.usda.gov/download-datasets | **Not verified in this session**: fdc.nal.usda.gov renders its terms client-side and they could not be read. Check the FDC site before citing a licence. |
| UK CoFID (McCance and Widdowson's Composition of Foods Integrated Dataset) | 2021 | https://www.gov.uk/government/publications/composition-of-foods-integrated-dataset-cofid | The GOV.UK page states: "All content is available under the Open Government Licence v3.0, except where otherwise stated © Crown copyright". |
| Indian Nutrient Databank (INDB) | GitHub commit `fbdb62bec519` (2025-04-08) | https://github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB- | The repository has no licence file (GitHub reports none). Cite the repository; ask the authors before redistribution. |

FNDDS energy/protein/fat/carbohydrate come from `food_nutrient.csv` nutrient numbers 208/203/204/205.

## Rules

1. Use USDA FNDDS when it has the dish. Otherwise use UK CoFID, then INDB.
2. `approximate = true` when the row is not a direct USDA match: a different database, a stand-in food,
   or a merged visual class whose variants differ.
3. `crosscheck_*` holds the same dish from a second database. If the two differ by more than 30% in
   kcal per 100 g, the row is automatically flagged `approximate`.
4. Default serving = an FNDDS portion weight where possible, otherwise an INDB serving. One INDB
   serving (kulfi, 266 g) was implausible and was replaced by the FNDDS ice cream bar portion.

## Rows that need a dietitian's review before production

The build log and the `notes` column list these:

* **Stand-ins (no source has the dish):** haleem (lentil soup with meat, likely too low), jalebi
  (funnel cake with sugar), gol gappay (fried puri shell only).
* **Large disagreements between databases:** pakora, gulab jamun, kulfi, biryani, chana/chole, halwa,
  chicken tikka. See `kcal_disagreement_pct`.
* INDB deep-fried recipes (samosa, pakora) come out at roughly 550-700 kcal/100 g, which suggests all
  frying oil is counted as absorbed. INDB was therefore not used as the primary source for fried foods.

In Phase 2 this file is replaced by the admin-managed Firestore `foods` collection. Its values
should be reviewed with the list above in mind.
