"""Copy the files a manifest uses from one non-Kaggle source into a folder for a private Kaggle dataset.

Khana is distributed on Google Drive, not Kaggle, so only the images that survived de-duplication and the
per-class cap are uploaded (same relative paths, so the manifest's MD5 check on Kaggle still applies).

    python tools/make_upload_subset.py --dataset khana --manifest data/splits/manifest_v2.csv --user <kaggle user>
    kaggle datasets create -p kaggle_out/upload_khana --dir-mode zip
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from smartbite_cv.config import load_config  # noqa: E402
from smartbite_cv.prepare import dataset_roots  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-config", default="configs/data_v2.yaml")
    ap.add_argument("--user", required=True)
    args = ap.parse_args()
    cfg = load_config(ROOT / args.data_config)
    root = dataset_roots(cfg, "local")[args.dataset]
    out = ROOT / "kaggle_out" / f"upload_{args.dataset}"
    if out.exists():
        shutil.rmtree(out)
    rows = [r for r in csv.DictReader(open(args.manifest, encoding="utf-8")) if r["dataset"] == args.dataset]
    for r in rows:
        dst = out / r["relpath"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / r["relpath"], dst)
    spec = cfg["datasets"][args.dataset]
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": f"smartbite-{args.dataset}-subset", "id": f"{args.user}/smartbite-{args.dataset}-subset",
        "licenses": [{"name": "other"}],
        "description": f"Private working copy of {len(rows)} images used by the SmartBite FYP from: {spec['title']} "
                       f"({spec.get('url', '')}). Not redistributed; academic, non-commercial use only."}, indent=2))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"{len(rows)} files, {size:.0f} MB in {out}")


if __name__ == "__main__":
    main()
