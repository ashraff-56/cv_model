"""Collect an external, real-world test set from Wikimedia Commons (never used for training).

Commons photos are taken by many different people with ordinary cameras and phones, and every file carries
its own licence and author, recorded here for attribution. Pipeline:

  1. for each class, find Commons categories whose name matches the dish (category search), and take the
     image files directly in them (no recursion into sub-categories, which drift off-topic quickly);
  2. download a 640 px rendition of each file and record title, licence, author and source URL;
  3. (tools/dedupe_external.py) drop any image that duplicates a training/val/test image;
  4. labels are then checked by eye on contact sheets (tools/commons_sheets.py); rejected files are listed
     with a reason in data/external/commons/review.yaml.

    python tools/collect_commons.py --per-class 40
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import socket
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from smartbite_cv.config import load_classes  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
UA = "SmartBite-FYP-eval/0.1 (academic food-recognition evaluation; non-commercial)"
OUT = ROOT / "data" / "external" / "commons"
FIELDS = ["label", "file", "title", "category", "license", "license_url", "artist", "source_url", "width", "height"]

# Search terms per class when the display name alone is a poor Commons query.
QUERIES = {
    "roti_chapati": ["chapati", "roti"], "gol_gappay": ["panipuri", "golgappa"], "pakora": ["pakora", "pakoda"],
    "chana_chole": ["chana masala", "chole"], "chicken_curry": ["chicken curry", "butter chicken"],
    "karahi": ["karahi", "kadai chicken"], "fried_fish": ["fried fish", "machli"], "ghiya": ["lauki", "bottle gourd curry"],
    "karela": ["karela", "bitter gourd curry"], "achar": ["achar", "Indian pickles"], "pudina_chutney": ["mint chutney"],
    "suji_halwa": ["suji halwa", "sooji halwa"], "gajar_halwa": ["gajar halwa"], "chai": ["masala chai", "chai"],
    "milk_soda": ["doodh soda", "glass of milk"], "sattu": ["sattu"], "chicken_roast": ["chargha", "roast chicken"],
    "paye": ["paya (food)", "paya"], "kunna": ["kunna"], "lamb_chops": ["lamb chops"], "kachumber_salad": ["kachumber"],
    "fried_chicken": ["fried chicken"], "spaghetti": ["spaghetti"], "salad": ["salad"], "ice_cream": ["ice cream"],
    "french_fries": ["french fries"], "hot_dog": ["hot dogs"], "omelette": ["omelettes"],
}
SKIP = {"not_food"}


def pin_host_via_google_dns(host: str) -> None:
    """Resolve one host through Google public DNS when the local resolver refuses it.

    Some home routers filter commons.wikimedia.org at DNS level; the user approved looking up this one
    host via 8.8.8.8 for this script only. Only name resolution changes: TLS still verifies the certificate
    for `host`, and nothing system-wide is modified.
    """
    try:
        socket.gethostbyname(host)
        return  # local resolver works: nothing to do
    except OSError:
        pass
    with urllib.request.urlopen(f"https://dns.google/resolve?name={host}&type=A", timeout=15) as r:
        ips = [a["data"] for a in json.load(r).get("Answer", []) if a.get("type") == 1]
    if not ips:
        raise RuntimeError(f"could not resolve {host}")
    original = socket.getaddrinfo

    def getaddrinfo(name, *args, **kwargs):
        return original(ips[0] if name == host else name, *args, **kwargs)

    socket.getaddrinfo = getaddrinfo
    print(f"{host}: local DNS refused; using {ips[0]} from Google DNS", flush=True)


def api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001 - transient network errors: back off and retry
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Commons API failed: {params}")


def find_categories(term: str, limit: int = 3) -> list[str]:
    words = [w for w in re.split(r"\W+", term.lower()) if len(w) > 2]
    res = api({"action": "query", "list": "search", "srsearch": term, "srnamespace": 14, "srlimit": 10})
    cats = [h["title"] for h in res.get("query", {}).get("search", [])]
    # keep categories whose own name mentions the dish; drop obvious off-topic families
    bad = ("recipe", "restaurant", "logo", "people", "festival", "shop", "stall", "book", "map", "seller")
    cats = [c for c in cats if all(w in c.lower() for w in words) and not any(b in c.lower() for b in bad)]
    return cats[:limit]


def category_files(cat: str, limit: int) -> list[str]:
    res = api({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmtype": "file", "cmlimit": 500})
    titles = [m["title"] for m in res.get("query", {}).get("categorymembers", [])]
    return [t for t in titles if re.search(r"\.(jpe?g|png|webp)$", t, re.I)][:limit]


def file_info(titles: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(titles), 40):
        res = api({"action": "query", "titles": "|".join(titles[i:i + 40]), "prop": "imageinfo",
                   "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 640})
        for p in res.get("query", {}).get("pages", []):
            ii = (p.get("imageinfo") or [{}])[0]
            meta = ii.get("extmetadata", {})
            val = lambda k: re.sub(r"<[^>]+>", "", meta.get(k, {}).get("value", "")).strip()  # noqa: E731
            if not ii.get("thumburl"):
                continue
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
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--min-side", type=int, default=200, help="skip originals smaller than this")
    ap.add_argument("--google-dns", action="store_true", help="resolve commons.wikimedia.org via 8.8.8.8 if refused locally")
    args = ap.parse_args()
    if args.google_dns:
        pin_host_via_google_dns("commons.wikimedia.org")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {}
    for cls in load_classes(args.classes):
        slug = cls["slug"]
        if slug in SKIP:
            continue
        terms = QUERIES.get(slug, [cls["display"].split("/")[0].strip()])
        cats = []
        for t in terms:
            cats += [c for c in find_categories(t) if c not in cats]
        titles: dict[str, str] = {}
        for c in cats:
            for t in category_files(c, args.per_class * 3):
                titles.setdefault(t, c)
        infos = [i for i in file_info(list(titles)) if min(i["width"] or 0, i["height"] or 0) >= args.min_side]
        infos = infos[:args.per_class]
        (OUT / slug).mkdir(exist_ok=True)
        got = 0
        for info in infos:
            name = re.sub(r"[^\w.-]+", "_", info["title"].split(":", 1)[1])[:120]
            dest = OUT / slug / (Path(name).stem + ".jpg")
            if dest.exists() or download(info["thumb"], dest):
                got += 1
                rows.append({"label": slug, "file": dest.relative_to(OUT).as_posix(), "title": info["title"],
                             "category": titles[info["title"]], **{k: info[k] for k in
                             ("license", "license_url", "artist", "source_url", "width", "height")}})
            time.sleep(0.2)
        summary[slug] = {"categories": cats, "images": got}
        print(f"{slug:18s} {got:3d} images from {cats}", flush=True)
    with open(OUT / "commons_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    (OUT / "collection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(rows)} images, {sum(1 for v in summary.values() if v['images'])} classes with at least one image")


if __name__ == "__main__":
    main()
