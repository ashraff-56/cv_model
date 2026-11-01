"""Build nutrition/nutrition.csv from the downloaded food-composition databases.

Every number comes from a database row named in nutrition/sources.yaml; nothing is typed by hand.

Inputs (download first; see nutrition/README.md):
  data/nutrition_src/FoodData_Central_survey_food_csv_2024-10-31/...   USDA FNDDS 2021-2023
  data/nutrition_src/cofid_2021.xlsx                                   UK CoFID 2021
  data/nutrition_src/indb/INDB.xlsx, recipes_servingsize.xlsx          Indian Nutrient Databank

    python tools/build_nutrition.py                                     # v1: 31 classes -> nutrition/nutrition.csv
    python tools/build_nutrition.py --classes configs/classes_v2.yaml         --sources nutrition/sources_v2.yaml --out nutrition/nutrition_v2.csv   # v2: 79 foods
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from smartbite_cv.config import load_classes  # noqa: E402

SRC = ROOT / "data" / "nutrition_src"
FNDDS = SRC / "FoodData_Central_survey_food_csv_2024-10-31" / "FoodData_Central_survey_food_csv_2024-10-31"
# FNDDS food_nutrient.nutrient_id holds the legacy nutrient number: 208 energy kcal, 203 protein,
# 204 total fat, 205 carbohydrate by difference.
FNDDS_NUTRIENTS = {"kcal": 208, "protein": 203, "fat": 204, "carbs": 205}
FIELDS = ["label", "display_name", "kcal_per_100g", "protein_g_per_100g", "carbs_g_per_100g", "fat_g_per_100g",
          "default_serving_g", "serving_description", "source", "source_id", "source_description", "source_url",
          "approximate", "crosscheck_kcal_per_100g", "crosscheck_source", "kcal_disagreement_pct", "notes"]
DISAGREEMENT_FLAG_PCT = 30.0


class Databases:
    def __init__(self):
        self.f_food = pd.read_csv(FNDDS / "food.csv").set_index("fdc_id")
        nut = pd.read_csv(FNDDS / "food_nutrient.csv")
        self.f_nut = nut[nut.nutrient_id.isin(FNDDS_NUTRIENTS.values())].pivot_table(
            index="fdc_id", columns="nutrient_id", values="amount")
        self.f_portion = pd.read_csv(FNDDS / "food_portion.csv")
        cofid = pd.read_excel(SRC / "cofid_2021.xlsx", sheet_name="1.3 Proximates").iloc[2:]
        self.cofid = cofid.set_index("Food Code")
        self.cofid_kcal = [c for c in cofid.columns if str(c).startswith("Energy (kcal)")][0]
        self.indb = pd.read_excel(SRC / "indb" / "INDB.xlsx").set_index("food_code")
        self.indb_serv = pd.read_excel(SRC / "indb" / "recipes_servingsize.xlsx").set_index("recipe_code")

    # ------------------------------------------------------------------ per-100 g values
    def fndds(self, fdc_id: int) -> dict:
        n = self.f_nut.loc[fdc_id]
        return {"kcal": n[208], "protein": n[203], "fat": n[204], "carbs": n[205],
                "description": self.f_food.loc[fdc_id, "description"],
                "source": "USDA FoodData Central, FNDDS 2021-2023 (survey foods, 2024-10-31 release)",
                "url": f"https://fdc.nal.usda.gov/food-details/{fdc_id}/nutrients"}

    def cofid_row(self, code: str) -> dict:
        r = self.cofid.loc[code]

        def num(v):
            if isinstance(v, str) and v.strip() == "Tr":  # trace amount
                return 0.0
            try:
                return float(v)
            except (TypeError, ValueError):  # 'N' = present but not measured
                raise ValueError(f"CoFID {code}: non-numeric value {v!r}") from None
        return {"kcal": num(r[self.cofid_kcal]), "protein": num(r["Protein (g)"]), "fat": num(r["Fat (g)"]),
                "carbs": num(r["Carbohydrate (g)"]), "description": r["Food Name"],
                "source": "UK CoFID 2021 (McCance and Widdowson's Composition of Foods Integrated Dataset)",
                "url": "https://www.gov.uk/government/publications/composition-of-foods-integrated-dataset-cofid"}

    def indb_row(self, code: str) -> dict:
        r = self.indb.loc[code]
        return {"kcal": float(r["energy_kcal"]), "protein": float(r["protein_g"]), "fat": float(r["fat_g"]),
                "carbs": float(r["carb_g"]), "description": r["food_name"],
                "source": "Indian Nutrient Databank (INDB), recipe-calculated",
                "url": "https://github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB-"}

    # ------------------------------------------------------------------ serving sizes
    def fndds_portion(self, fdc_id: int, description: str) -> tuple[float, str]:
        p = self.f_portion[(self.f_portion.fdc_id == fdc_id) & (self.f_portion.portion_description == description)]
        if p.empty:
            raise KeyError(f"FNDDS {fdc_id}: no portion {description!r}")
        return float(p.iloc[0].gram_weight), f"{description} (FNDDS {fdc_id})"

    def indb_portion(self, code: str) -> tuple[float, str]:
        """INDB per-serving grams = per-serving energy / per-100 g energy x 100 (both from INDB.xlsx)."""
        r = self.indb.loc[code]
        per_serving = [c for c in self.indb.columns if c.startswith("unit_serving_energy_kcal")]
        if not per_serving:
            raise KeyError("INDB.xlsx has no per-serving energy column")
        grams = float(r[per_serving[0]]) / float(r["energy_kcal"]) * 100.0
        s = self.indb_serv.loc[code]
        return round(grams, 1), f"1 {s['servings_unit']} (INDB {code})"


def lookup(db: Databases, name: str, food_id) -> dict:
    if name == "fndds":
        return db.fndds(int(food_id))
    if name == "cofid":
        return db.cofid_row(str(food_id))
    if name == "indb":
        return db.indb_row(str(food_id))
    raise ValueError(f"unknown database {name!r}")


def serving_of(db: Databases, s: dict) -> tuple[float, str]:
    """Default serving grams from a database portion named in the spec (never typed by hand)."""
    if "serving" in s:
        return db.fndds_portion(int(s["id"]), s["serving"])
    if "serving_fndds" in s:
        return db.fndds_portion(int(s["serving_fndds"]["id"]), s["serving_fndds"]["portion"])
    return db.indb_portion(s["serving_indb"]["code"])


def composite(db: Databases, parts: list[dict]) -> tuple[dict, float, str]:
    """A plated dish with no database entry (e.g. halwa puri): components with database values per 100 g and
    database portions; per-100 g values are the gram-weighted mean, the serving is the sum of the portions."""
    total, acc, desc, serv = 0.0, {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}, [], []
    for part in parts:
        v = lookup(db, part["db"], part["id"])
        grams, label = serving_of(db, part)
        grams *= part.get("count", 1)
        total += grams
        for k in acc:
            acc[k] += float(v[k]) * grams
        desc.append(f"{part.get('count', 1)} x {v['description']}")
        serv.append(f"{part.get('count', 1)} x {label}")
    values = {k: acc[k] / total for k in acc}
    values.update({"description": "Composite: " + " + ".join(desc), "source": "Composite of database rows (see notes)",
                   "url": ""})
    return values, round(total, 1), " + ".join(serv)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--classes", default=str(ROOT / "configs" / "classes.yaml"))
    ap.add_argument("--sources", default=str(ROOT / "nutrition" / "sources.yaml"))
    ap.add_argument("--out", default=str(ROOT / "nutrition" / "nutrition.csv"))
    args = ap.parse_args()
    spec = yaml.safe_load(Path(args.sources).read_text(encoding="utf-8"))["foods"]
    classes = [c for c in load_classes(args.classes) if c["slug"] != "not_food"]
    missing = [c["slug"] for c in classes if c["slug"] not in spec]
    if missing:
        raise SystemExit(f"{args.sources} lacks: {missing}")
    db = Databases()
    rows = []
    for c in classes:
        s = spec[c["slug"]]
        if "composite" in s:
            values, grams, serving = composite(db, s["composite"])
            s = {**s, "id": "composite"}
        else:
            values = lookup(db, s["db"], s["id"])
            grams, serving = serving_of(db, s)
        cross = lookup(db, s["crosscheck"]["db"], s["crosscheck"]["id"]) if "crosscheck" in s else None
        disagreement = abs(cross["kcal"] - values["kcal"]) / values["kcal"] * 100 if cross else None
        approximate = bool(s["approximate"]) or (disagreement is not None and disagreement > DISAGREEMENT_FLAG_PCT)
        rows.append({
            "label": c["slug"], "display_name": c["display"],
            "kcal_per_100g": round(float(values["kcal"]), 1), "protein_g_per_100g": round(float(values["protein"]), 2),
            "carbs_g_per_100g": round(float(values["carbs"]), 2), "fat_g_per_100g": round(float(values["fat"]), 2),
            "default_serving_g": grams, "serving_description": serving,
            "source": values["source"], "source_id": s["id"], "source_description": values["description"],
            "source_url": values["url"], "approximate": str(approximate).lower(),
            "crosscheck_kcal_per_100g": round(float(cross["kcal"]), 1) if cross else "",
            "crosscheck_source": f"{s['crosscheck']['db'].upper()} {s['crosscheck']['id']} {cross['description']}"
            if cross else "",
            "kcal_disagreement_pct": round(disagreement, 1) if cross else "",
            "notes": s.get("note", ""),
        })
    out = Path(args.out)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows, {sum(r['approximate'] == 'true' for r in rows)} approximate)")
    for r in rows:
        flag = " ~" if r["approximate"] == "true" else ""
        cc = f"  [x-check {r['crosscheck_kcal_per_100g']} ({r['kcal_disagreement_pct']}%)]" if r["crosscheck_source"] else ""
        print(f"  {r['label']:<15} {r['kcal_per_100g']:>6} kcal/100g  serving {r['default_serving_g']:>6} g  "
              f"<- {r['source_id']} {r['source_description'][:45]}{flag}{cc}")


if __name__ == "__main__":
    main()
