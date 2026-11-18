"""Audit v3 dataset source folders without changing or training on any data.

Examples:
  python tools/audit_v3_sources.py --profile local
  python tools/audit_v3_sources.py --profile kaggle

This is a raw-file inventory only. Image decoding, duplicate removal, label review and split
creation are performed later by smartbite_cv.prepare after source paths are verified.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
import glob

import yaml

from smartbite_cv.config import REPO_ROOT, load_classes, load_config, resolve_path
from smartbite_cv.prepare import dataset_roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_v3.yaml")
    parser.add_argument("--classes", default="configs/classes_v3.yaml")
    parser.add_argument("--taxonomy", default="configs/taxonomy_v3.yaml")
    parser.add_argument("--profile", choices=["local", "kaggle"], default="local")
    parser.add_argument("--out-dir", default="reports/data_v3")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    classes = load_classes(resolve_path(args.classes))
    taxonomy = yaml.safe_load(resolve_path(args.taxonomy).read_text(encoding="utf-8"))
    metadata = {c["slug"]: c for c in taxonomy.get("classes", [])}
    roots = dataset_roots(cfg, args.profile)
    extensions = {x.lower() for x in cfg.get("image_exts", [])}
    rows: list[dict] = []
    totals: Counter[str] = Counter()
    source_totals: Counter[str] = Counter()
    missing_roots = sorted(set(cfg["datasets"]) - set(roots))

    for cls in classes:
        slug = cls["slug"]
        meta = metadata.get(slug, {})
        sources = cls.get("sources") or []
        class_total = 0
        if not sources:
            rows.append({
                "slug": slug, "display": cls.get("display", slug), "category": meta.get("category", "special"),
                "origin": meta.get("origin", "legacy"), "dataset": "", "pattern": "",
                "matched_folders": 0, "raw_images": 0, "target_images": meta.get("target_images", ""),
                "status": "NO_SOURCE_MAPPING",
            })
            continue

        for source in sources:
            dataset_id, pattern = next(iter(source.items()))
            if dataset_id not in roots:
                rows.append({
                    "slug": slug, "display": cls.get("display", slug), "category": meta.get("category", "legacy"),
                    "origin": meta.get("origin", "legacy"), "dataset": dataset_id, "pattern": pattern,
                    "matched_folders": 0, "raw_images": 0, "target_images": meta.get("target_images", ""),
                    "status": "DATASET_ROOT_MISSING",
                })
                continue

            spec = cfg["datasets"].get(dataset_id, {})
            matches = [Path(p) for p in sorted(glob.glob(str(roots[dataset_id] / pattern))) if Path(p).is_dir()]
            count = 0
            for folder in matches:
                for path in folder.iterdir():
                    if path.is_file() and (path.suffix.lower() in extensions or spec.get("any_extension", False)):
                        count += 1
            class_total += count
            source_totals[dataset_id] += count
            rows.append({
                "slug": slug, "display": cls.get("display", slug), "category": meta.get("category", "legacy"),
                "origin": meta.get("origin", "legacy"), "dataset": dataset_id, "pattern": pattern,
                "matched_folders": len(matches), "raw_images": count,
                "target_images": meta.get("target_images", ""),
                "status": "OK" if count else "FOLDER_OR_IMAGES_NOT_FOUND",
            })
        totals[slug] = class_total

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "source_audit.csv"
    fields = ["slug", "display", "category", "origin", "dataset", "pattern", "matched_folders",
              "raw_images", "target_images", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# SmartBite v3 raw dataset source audit", "",
        f"Profile: `{args.profile}`", "",
        "**Important:** these are raw folder/file counts only. They are not verified unique images, usable images,",
        "or final train counts. No image has been decoded, deduplicated, visually reviewed or split by this script.", "",
        f"- Configured food classes plus special labels: {len(classes)}",
        f"- Dataset roots found: {len(roots)}/{len(cfg['datasets'])}",
        f"- Dataset roots missing: {', '.join(missing_roots) if missing_roots else 'none'}",
        f"- Classes with at least one raw image: {sum(1 for n in totals.values() if n > 0)}",
        f"- Classes with no raw images: {sum(1 for n in totals.values() if n == 0)}", "",
        "## Per-class raw image totals", "",
        "| class | category | raw images | target | status |",
        "|---|---|---:|---:|---|",
    ]
    for cls in classes:
        slug = cls["slug"]
        meta = metadata.get(slug, {})
        n = totals.get(slug, 0)
        status = "HAS_RAW_FILES" if n else "NO_RAW_FILES"
        target = meta.get("target_images", "")
        report.append(f"| {slug} | {meta.get('category', 'special')} | {n} | {target} | {status} |")
    report += ["", "## Raw image totals by source", "", "| dataset | raw files | root status |", "|---|---:|---|"]
    for dataset_id in cfg["datasets"]:
        report.append(f"| {dataset_id} | {source_totals.get(dataset_id, 0)} | "
                      f"{'FOUND' if dataset_id in roots else 'MISSING'} |")
    report += ["", "## Next gate", "",
        "1. Resolve missing dataset roots and folder mappings.",
        "2. Inspect contact sheets for every newly mapped class.",
        "3. Run the existing preparation pipeline for image decoding, duplicate detection and leakage-safe splits.",
        "4. Keep classes below the minimum usable-image threshold out of the training label map.",
        "5. Record per-source license and attribution evidence before the data is approved.", ""]
    (out_dir / "source_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / 'source_audit.md'}")
    print(f"Found {sum(1 for n in totals.values() if n > 0)}/{len(classes)} classes with raw files.")
    print("Missing dataset roots:", ", ".join(missing_roots) if missing_roots else "none")


if __name__ == "__main__":
    main()
