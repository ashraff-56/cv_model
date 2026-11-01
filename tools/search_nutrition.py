"""Search the local food-composition databases for candidate rows per class (helper for nutrition/sources.yaml).

Prints FNDDS, UK CoFID and INDB rows whose names match a regex, with per-100 g kcal/protein/carbs/fat and
(INDB) the serving size, so curated choices in sources.yaml can be made from real rows.

    python tools/search_nutrition.py --terms "nihari=nihari|beef stew" "bhindi=bhindi|okra"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from build_nutrition import Databases  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", nargs="+", required=True, help="label=regex")
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()
    db = Databases()
    fn = db.f_food["description"].astype(str)
    co = db.cofid["Food Name"].astype(str)
    ind = db.indb["food_name"].astype(str)
    per_serving = [c for c in db.indb.columns if c.startswith("unit_serving_energy_kcal")]
    for item in args.terms:
        label, pattern = item.split("=", 1)
        rx = re.compile(pattern, re.I)
        print(f"\n### {label}   /{pattern}/")
        for fid in fn[fn.str.contains(rx)].index[:args.limit]:
            try:
                r = db.fndds(int(fid))
                print(f"  fndds {fid}: {r['description'][:70]} | {r['kcal']:.0f} kcal P{r['protein']:.1f} C{r['carbs']:.1f} F{r['fat']:.1f}")
            except KeyError:
                pass
        for code in co[co.str.contains(rx)].index[:args.limit]:
            try:
                r = db.cofid_row(str(code))
                print(f"  cofid {code}: {r['description'][:70]} | {r['kcal']:.0f} kcal P{r['protein']:.1f} C{r['carbs']:.1f} F{r['fat']:.1f}")
            except (KeyError, ValueError) as exc:
                print(f"  cofid {code}: {db.cofid.loc[code, 'Food Name'][:70]} | (unusable: {exc})")
        for code in ind[ind.str.contains(rx)].index[:args.limit]:
            r = db.indb_row(str(code))
            serving = ""
            if per_serving and code in db.indb_serv.index and float(r["kcal"]) > 0:
                g = float(db.indb.loc[code, per_serving[0]]) / float(r["kcal"]) * 100
                serving = f" | serving {g:.0f} g ({db.indb_serv.loc[code, 'servings_unit']})"
            print(f"  indb {code}: {r['description'][:70]} | {r['kcal']:.0f} kcal P{r['protein']:.1f} C{r['carbs']:.1f} F{r['fat']:.1f}{serving}")


if __name__ == "__main__":
    main()
