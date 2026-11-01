"""Collect a small external, real-world test set from photos embedded in Wikipedia dish articles.

Wikimedia Commons itself is unreachable from the development network (connection reset at the TLS
handshake), so photos are taken from Wikipedia articles, which embed Commons files and serve them through the
normal Wikipedia API and upload.wikimedia.org. For each class: find the English article, follow its
language links (Urdu, Hindi, Bengali, Punjabi, Marathi, Tamil; regional editions often carry locally taken
photos), and take the photos embedded in those articles. Every file's licence, author and source page are
recorded for attribution. Never used for training. Next steps: tools/dedupe_external.py (drop images that
duplicate any training/val/test image), then a visual label check (review.csv).

    python tools/collect_wikipedia.py --per-class 15
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from smartbite_cv.config import load_classes  # noqa: E402

UA = "SmartBite-FYP-eval/0.1 (academic food-recognition evaluation; non-commercial)"
OUT = ROOT / "data" / "external" / "wikipedia"
LANGS = ["ur", "hi", "bn", "pa", "mr", "ta"]
FIELDS = ["label", "file", "title", "article", "lang", "license", "license_url", "artist", "source_url", "width", "height"]
JUNK = re.compile(r"logo|icon|map|flag|symbol|emblem|wiki|question_book|edit-clear|padlock|portal|coat_of_arms|"
                  r"signature|disambig|stub|ambox|crystal|nuvola|folder|\.svg$|\.gif$|\.tif", re.I)

# Article search terms per class (English Wikipedia).
QUERIES = {
    "roti_chapati": "Chapati", "naan": "Naan", "paratha": "Paratha", "aloo_paratha": "Aloo paratha",
    "anda_paratha": "Egg paratha", "sheermal": "Sheermal", "puri": "Puri (food)", "halwa_puri": "Halwa puri",
    "paratha_roll": "Kathi roll", "samosa": "Samosa", "pakora": "Pakora", "gol_gappay": "Panipuri",
    "chana_chaat": "Chaat", "karahi": "Karahi chicken", "korma": "Korma", "chicken_curry": "Chicken curry",
    "nihari": "Nihari", "kunna": "Kunna gosht", "paye": "Paya (food)", "haleem": "Haleem", "aloo_gosht": "Aloo gosht",
    "palak_gosht": "Saag gosht", "keema": "Keema", "kofta": "Kofta", "fish_curry": "Fish curry",
    "fried_fish": "Fried fish", "chicken_tikka": "Chicken tikka", "malai_boti": "Malai tikka",
    "seekh_kebab": "Seekh kebab", "chapli_kebab": "Chapli kebab", "lamb_chops": "Lamb chop",
    "chicken_roast": "Chargha", "daal": "Dal", "chana_chole": "Chana masala", "kadhi_pakora": "Kadhi",
    "aloo_baingan": "Aloo baingan", "aloo_gobi": "Aloo gobi", "aloo_methi": "Aloo methi", "aloo_palak": "Saag aloo",
    "aloo_shimla_mirch": "Aloo shimla mirch", "bhindi": "Bhindi masala", "ghiya": "Calabash", "karela": "Momordica charantia",
    "mixed_sabzi": "Mixed vegetable curry", "saag": "Saag", "raita": "Raita", "kachumber_salad": "Kachumber",
    "achar": "South Asian pickle", "pudina_chutney": "Mint chutney", "gulab_jamun": "Gulab jamun", "jalebi": "Jalebi",
    "gajar_halwa": "Gajar ka halwa", "suji_halwa": "Sooji halwa", "barfi": "Barfi", "rabri": "Rabri",
    "ras_malai": "Ras malai", "kheer": "Kheer", "kulfi": "Kulfi", "falooda": "Falooda", "chai": "Masala chai",
    "coffee": "Coffee", "lassi": "Lassi", "lemonade": "Lemonade", "milk_soda": "Doodh soda", "sattu": "Sattu",
    "pizza": "Pizza", "burger": "Hamburger", "french_fries": "French fries", "fried_rice": "Fried rice",
    "omelette": "Omelette", "fried_chicken": "Fried chicken", "sandwich": "Sandwich", "hot_dog": "Hot dog",
    "donut": "Doughnut", "ice_cream": "Ice cream", "chocolate_cake": "Chocolate cake", "spaghetti": "Spaghetti",
    "salad": "Salad", "biryani": "Biryani",
}
SKIP = {"not_food"}


def api(lang: str, params: dict) -> dict:
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    err = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as exc:  # noqa: BLE001 - transient network errors / rate limits: back off and retry
            err = exc
            time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(f"Wikipedia API failed ({err}): {lang} {params.get('titles') or params.get('srsearch')}")


def resolve_article(term: str) -> str | None:
    res = api("en", {"action": "query", "titles": term, "redirects": 1})
    page = res["query"]["pages"][0]
    if not page.get("missing"):
        return page["title"]
    hits = api("en", {"action": "query", "list": "search", "srsearch": term, "srlimit": 1})["query"]["search"]
    return hits[0]["title"] if hits else None


def language_titles(title: str) -> list[tuple[str, str]]:
    res = api("en", {"action": "query", "titles": title, "prop": "langlinks", "lllimit": 500})
    links = res["query"]["pages"][0].get("langlinks", [])
    return [("en", title)] + [(l["lang"], l["title"]) for l in links if l["lang"] in LANGS]


def article_images(lang: str, title: str) -> list[str]:
    res = api(lang, {"action": "query", "titles": title, "prop": "images", "imlimit": 100})
    files = [i["title"] for p in res["query"]["pages"] for i in p.get("images", [])]
    return [f for f in files if re.search(r"\.(jpe?g|png|webp)$", f, re.I) and not JUNK.search(f)]


def file_info(lang: str, titles: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(titles), 40):
        res = api(lang, {"action": "query", "titles": "|".join(titles[i:i + 40]), "prop": "imageinfo",
                         "iiprop": "url|size|extmetadata", "iiurlwidth": 640})
        for p in res["query"]["pages"]:
            ii = (p.get("imageinfo") or [{}])[0]
            if not ii.get("thumburl"):
                continue
            meta = ii.get("extmetadata", {})
            val = lambda k: re.sub(r"<[^>]+>", "", meta.get(k, {}).get("value", "")).strip()  # noqa: E731
            out.append({"title": p["title"], "thumb": ii["thumburl"], "source_url": ii.get("descriptionurl", ""),
                        "width": ii.get("width"), "height": ii.get("height"), "license": val("LicenseShortName"),
                        "license_url": val("LicenseUrl"), "artist": val("Artist")[:200]})
    return out


def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                dest.write_bytes(r.read())
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--classes", type=Path, default=ROOT / "configs" / "classes_v2.yaml")
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--min-side", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, summary, seen = [], {}, set()
    for cls in load_classes(args.classes):
        slug = cls["slug"]
        if slug in SKIP:
            continue
        try:
            collected = collect_class(slug, QUERIES.get(slug, cls["display"]), args, seen)
        except RuntimeError as exc:
            print(f"{slug:18s}   skipped: {exc}", flush=True)
            summary[slug] = {"article": None, "images": 0, "error": str(exc)}
            continue
        rows += collected["rows"]
        summary[slug] = {"article": collected["article"], "images": len(collected["rows"])}
        print(f"{slug:18s} {len(collected['rows']):3d}  ({collected['article']})", flush=True)
    with open(OUT / "wikipedia_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    (OUT / "collection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} images; {sum(1 for v in summary.values() if v['images'])} classes with at least one image")


def collect_class(slug: str, query: str, args, seen: set) -> dict:
    rows = []
    article = resolve_article(query)
    if article is None:
        return {"article": None, "rows": []}
    candidates: dict[str, tuple[str, str]] = {}
    for lang, title in language_titles(article):
        try:
            files = article_images(lang, title)
        except RuntimeError:   # an unreachable language edition is skipped, not fatal
            print(f"  skipped {lang}:{title}", flush=True)
            continue
        for f in files:
            key = f.split(":", 1)[1].replace(" ", "_")   # same file appears under language-specific prefixes
            if key not in seen:
                candidates.setdefault(key, (lang, title, f))
    # file details through the English API with the canonical "File:" prefix: works for every photo stored
    # on Commons (files local to one language edition are skipped)
    infos = []
    for info in file_info("en", ["File:" + k for k in candidates]):
        k = info["title"].split(":", 1)[1].replace(" ", "_")
        lang, art, _ = candidates.get(k, ("en", article, ""))
        infos.append({**info, "lang": lang, "article": art, "key": k})
    infos = [i for i in infos if min(i["width"] or 0, i["height"] or 0) >= args.min_side][:args.per_class]
    (OUT / slug).mkdir(exist_ok=True)
    for info in infos:
        name = re.sub(r"[^\w.-]+", "_", info["key"])[:120]
        dest = OUT / slug / (Path(name).stem + ".jpg")
        if dest.exists() or download(info["thumb"], dest):
            seen.add(info["key"])
            rows.append({"label": slug, "file": dest.relative_to(OUT).as_posix(), "title": info["title"],
                         "article": info["article"], "lang": info["lang"],
                         **{k: info[k] for k in ("license", "license_url", "artist", "source_url", "width", "height")}})
        time.sleep(0.1)
    return {"article": article, "rows": rows}


if __name__ == "__main__":
    main()
