"""Build the leakage-safe training manifest from the raw source datasets.

Steps (all deterministic for a given seed and identical input bytes):
 1. collect image files per class from configs/classes.yaml
 2. decode every image (EXIF-corrected); drop unreadable or tiny ones
 3. MD5 (exact duplicates) + pHash of image and its mirror (near-duplicates) -> clusters
 4. drop clusters whose members carry different class labels (label conflict)
 5. keep one representative per cluster (largest image), so no near-duplicate pair can
    straddle train/val/test
 6. cap each class (seeded random subset), drop classes below the minimum
 7. split each class 70/15/15, ordered by a seeded hash of the image content (not file path)
 8. optional: write resized copies for faster training

    python -m smartbite_cv.prepare                   # local paths from configs/data.yaml
    python -m smartbite_cv.prepare --profile kaggle --verify-only   # check files match the manifest
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import REPO_ROOT, load_classes, load_config, resolve_path
from .imageio import load_rgb
from .phash import cluster, near_duplicate_pairs, phash_pair

MANIFEST_FIELDS = ["md5", "label", "split", "dataset", "relpath", "width", "height", "phash", "cluster_size"]


# ---------------------------------------------------------------- dataset roots
def find_kaggle_root(marker: str, search_root: Path = Path("/kaggle/input"), max_depth: int = 4) -> Path | None:
    """Return the directory under /kaggle/input that contains `marker` (mount layout varies)."""
    frontier = [search_root]
    for _ in range(max_depth + 1):
        nxt = []
        for d in frontier:
            if (d / marker).exists():
                return d
            try:
                nxt.extend(p for p in d.iterdir() if p.is_dir())
            except OSError:
                continue
        frontier = nxt
    return None


def dataset_roots(cfg: dict, profile: str) -> dict[str, Path]:
    roots = {}
    for ds_id, spec in cfg["datasets"].items():
        if profile == "local":
            root = resolve_path(spec["local"])
        elif profile == "kaggle":
            root = find_kaggle_root(spec["marker"])
        else:
            raise ValueError(f"unknown profile {profile!r}")
        if root is None or not (root / spec["marker"]).exists():
            print(f"WARNING: dataset {ds_id!r} not found (marker {spec['marker']!r})", file=sys.stderr)
            continue
        roots[ds_id] = root
    return roots


# ---------------------------------------------------------------- collection
def collect(classes: list[dict], roots: dict[str, Path], exts: set[str]) -> list[dict]:
    records = []
    for cls in classes:
        for source in cls["sources"]:
            (ds_id, pattern), = source.items()
            if ds_id not in roots:
                continue
            root = roots[ds_id]
            for folder in sorted(glob.glob(str(root / pattern))):
                for name in sorted(os.listdir(folder)):
                    if Path(name).suffix.lower() in exts:
                        rel = Path(folder, name).relative_to(root).as_posix()
                        records.append({"label": cls["slug"], "dataset": ds_id, "relpath": rel})
    return records


def _inspect(args: tuple[str, int]) -> dict:
    path, min_side = args
    try:
        data = Path(path).read_bytes()
        img = load_rgb(data)
        w, h = img.size
        if min(w, h) < min_side:
            return {"error": f"too small ({w}x{h})"}
        ph, ph_flip = phash_pair(img)
        return {"md5": hashlib.md5(data).hexdigest(), "width": w, "height": h, "phash": ph, "phash_flip": ph_flip}
    except MemoryError:  # transient under parallel load (very large photos); retried serially
        return {"retry": True}
    except Exception as exc:  # corrupt/unsupported file: record and skip
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def seeded_key(seed: int, md5: str) -> str:
    return hashlib.sha1(f"{seed}:{md5}".encode()).hexdigest()


def _ref(item: dict) -> dict:
    return {k: item[k] for k in ("label", "dataset", "relpath")}


def _reduce_clusters(items: list[dict], cluster_ids: list[int]) -> tuple[list[dict], list[list[dict]], Counter]:
    """One representative (largest image) per duplicate cluster; clusters with mixed labels are dropped."""
    members: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        members[cid].append(idx)
    kept, conflicts, removed = [], [], Counter()
    for idxs in members.values():
        if len({items[i]["label"] for i in idxs}) > 1:  # same picture filed under different dishes
            conflicts.append([_ref(items[i]) for i in idxs])
            continue
        # largest image wins; ties broken by content hash so the choice never depends on file order
        rep = max(idxs, key=lambda i: (items[i]["width"] * items[i]["height"], items[i]["md5"]))
        removed[items[rep]["label"]] += len(idxs) - 1
        kept.append({**items[rep], "cluster_size": sum(items[i]["cluster_size"] for i in idxs)})
    return kept, conflicts, removed


# ---------------------------------------------------------------- main pipeline
def build_manifest(cfg: dict, classes: list[dict], roots: dict[str, Path], workers: int) -> tuple[list[dict], dict]:
    seed = cfg["seed"]
    records = collect(classes, roots, {e.lower() for e in cfg["image_exts"]})
    stats: dict = {"collected": Counter(r["label"] for r in records)}
    by_source = Counter((r["label"], r["dataset"]) for r in records)
    print(f"collected {len(records)} files from {len(roots)} datasets")

    jobs = [(str(roots[r["dataset"]] / r["relpath"]), cfg["min_side"]) for r in records]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_inspect, jobs, chunksize=64))
    for i, res in enumerate(results):
        if res.get("retry"):
            results[i] = _inspect(jobs[i])
            if results[i].get("retry"):
                results[i] = {"error": "MemoryError (image too large to decode)"}

    good, errors = [], []
    for rec, res in zip(records, results):
        if "error" in res:
            errors.append({**rec, "error": res["error"]})
        else:
            good.append({**rec, **res})
    stats["unreadable_or_small"] = Counter(e["label"] for e in errors)
    print(f"decoded {len(good)} images, dropped {len(errors)} unreadable/small")

    # pass 1: exact + near duplicates by pHash (an exact copy has distance 0, so one pass covers both)
    for g in good:
        g["cluster_size"] = 1
    pairs = near_duplicate_pairs([g["phash"] for g in good], [g["phash_flip"] for g in good],
                                 cfg["dedupe"]["max_hamming"])
    kept, conflicts, dup_removed = _reduce_clusters(good, cluster(len(good), pairs))
    stats["phash_duplicates_removed"] = dup_removed
    print(f"pHash pass: {len(kept)} unique images; {len(conflicts)} label-conflict clusters dropped")

    # pass 2: re-cropped / re-framed copies by CNN-embedding cosine similarity
    emb_cfg = cfg["dedupe"].get("embedding", {})
    if emb_cfg.get("enabled", False):
        from .embed_dedupe import embed, similar_pairs

        emb = embed([roots[k["dataset"]] / k["relpath"] for k in kept], emb_cfg["model"], workers=min(workers, 4),
                    cache_file=resolve_path(emb_cfg["cache_file"]), keys=[k["md5"] for k in kept])
        audit_min = emb_cfg.get("audit_min_cosine", emb_cfg["min_cosine"])
        sim = list(similar_pairs(emb, min(audit_min, emb_cfg["min_cosine"])))
        stats["embedding_pair_audit"] = [
            {"cosine": round(c, 4), "a": _ref(kept[i]), "b": _ref(kept[j])} for i, j, c in sorted(sim, key=lambda t: -t[2])]
        dup_pairs = [(i, j) for i, j, c in sim if c >= emb_cfg["min_cosine"]]
        kept, conflicts2, dup_removed2 = _reduce_clusters(kept, cluster(len(kept), dup_pairs))
        conflicts += conflicts2
        stats["embedding_duplicates_removed"] = dup_removed2
        print(f"embedding pass (cosine >= {emb_cfg['min_cosine']}): {len(kept)} unique images; "
              f"{len(conflicts2)} more label-conflict clusters dropped")
        dup_removed = dup_removed + dup_removed2
    stats["duplicates_removed"] = dup_removed
    stats["label_conflict_clusters"] = len(conflicts)
    stats["label_conflict_images"] = Counter(m["label"] for c in conflicts for m in c)

    per_class: dict[str, list[dict]] = defaultdict(list)
    for row in kept:
        per_class[row["label"]].append(row)

    manifest, dropped_classes = [], {}
    fr = cfg["split"]
    for cls in classes:
        rows = sorted(per_class.get(cls["slug"], []), key=lambda r: seeded_key(seed, r["md5"]))
        if len(rows) < cfg["min_per_class"]:
            dropped_classes[cls["slug"]] = len(rows)
            continue
        rows = rows[: cfg["cap_per_class"]]
        n = len(rows)
        n_train = round(n * fr["train"])
        n_val = round(n * fr["val"])
        for i, row in enumerate(rows):
            split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            manifest.append({**row, "split": split, "phash": f"{row['phash']:016x}"})
    stats["dropped_classes"] = dropped_classes

    stats["by_source_raw"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(by_source.items())}
    stats["kept_by_source"] = {f"{k[0]}|{k[1]}": v for k, v in
                               sorted(Counter((m["label"], m["dataset"]) for m in manifest).items())}
    stats["final"] = {s: Counter(m["label"] for m in manifest if m["split"] == s) for s in ("train", "val", "test")}
    stats["errors_sample"] = errors[:50]
    stats["conflicts_sample"] = conflicts[:50]
    return manifest, stats


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["label"], r["split"], r["md5"])))


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cache_one(args: tuple[str, str, int, int]) -> str | None:
    src, dst, short_side, quality = args
    if os.path.exists(dst):
        return None
    try:
        img = load_rgb(src)
        w, h = img.size
        scale = short_side / min(w, h)
        if scale < 1:
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), resample=3)  # bicubic
        tmp = dst + ".tmp"
        img.save(tmp, "JPEG", quality=quality)
        os.replace(tmp, dst)
        return None
    except Exception as exc:
        return f"{src}: {exc}"


def build_cache(rows: list[dict], roots: dict[str, Path], cache_dir: Path, short_side: int, quality: int,
                workers: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(roots[r["dataset"]] / r["relpath"]), str(cache_dir / f"{r['md5']}.jpg"), short_side, quality)
            for r in rows if r["dataset"] in roots]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        failures = [f for f in pool.map(_cache_one, jobs, chunksize=64) if f]
    print(f"cache: {len(jobs)} images in {cache_dir} ({len(failures)} failures)")
    for f in failures[:10]:
        print("  ", f)


def verify(rows: list[dict], roots: dict[str, Path], workers: int) -> list[dict]:
    """Rows whose file is missing or whose bytes differ from the manifest MD5."""
    def check(r):
        p = roots.get(r["dataset"])
        if p is None or not (p / r["relpath"]).exists():
            return r
        return None if hashlib.md5((p / r["relpath"]).read_bytes()).hexdigest() == r["md5"] else r
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for r in pool.map(check, rows) if r is not None]


def write_report(stats: dict, classes: list[dict], cfg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = stats.pop("embedding_pair_audit", None)
    if audit is not None:
        with open(out_dir / "embedding_pairs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["cosine", "label_a", "dataset_a", "relpath_a", "label_b", "dataset_b", "relpath_b"])
            for p in audit:
                w.writerow([p["cosine"], *p["a"].values(), *p["b"].values()])
    with open(out_dir / "prep_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=lambda o: dict(o))
    lines = ["# Data preparation report", "",
             f"Seed {cfg['seed']}; near-duplicate threshold pHash Hamming <= {cfg['dedupe']['max_hamming']}/64 "
             f"(image or mirror); cap {cfg['cap_per_class']}/class; min {cfg['min_per_class']}/class.", "",
             "| class | raw files | unreadable/small | duplicates removed | in label conflicts | train | val | test |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    tot = Counter()
    for cls in classes:
        s = cls["slug"]
        vals = [stats["collected"].get(s, 0), stats["unreadable_or_small"].get(s, 0),
                stats["duplicates_removed"].get(s, 0), stats["label_conflict_images"].get(s, 0),
                stats["final"]["train"].get(s, 0), stats["final"]["val"].get(s, 0), stats["final"]["test"].get(s, 0)]
        for k, v in zip(range(7), vals):
            tot[k] += v
        lines.append(f"| {s} | " + " | ".join(str(v) for v in vals) + " |")
    lines.append("| **total** | " + " | ".join(f"**{tot[k]}**" for k in range(7)) + " |")
    lines += ["", f"Label-conflict clusters dropped: {stats['label_conflict_clusters']}",
              f"Classes dropped for having too few images: {stats['dropped_classes'] or 'none'}", "",
              "## Images kept per source", "", "| class | dataset | raw | kept |", "|---|---|---:|---:|"]
    for key, raw in stats["by_source_raw"].items():
        lines.append(f"| {key.split('|')[0]} | {key.split('|')[1]} | {raw} | {stats['kept_by_source'].get(key, 0)} |")
    (out_dir / "prep_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build / verify the training manifest")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "data.yaml"))
    ap.add_argument("--classes", default=str(REPO_ROOT / "configs" / "classes.yaml"))
    ap.add_argument("--profile", choices=["local", "kaggle"], default="local")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--verify-only", action="store_true", help="check an existing manifest against the files")
    ap.add_argument("--cache", action="store_true", help="write resized training copies")
    ap.add_argument("--cache-dir", help="override cache.dir")
    ap.add_argument("--write-verified", help="with --verify-only: write the rows that passed to this CSV")
    ap.add_argument("--set", action="append", default=[], help="config override key.sub=value")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.set)
    classes = load_classes(args.classes)
    roots = dataset_roots(cfg, args.profile)
    manifest_path = resolve_path(cfg["manifest"])

    if args.verify_only:
        rows = read_manifest(manifest_path)
        bad = verify(rows, roots, args.workers)
        print(f"verified {len(rows) - len(bad)}/{len(rows)} manifest rows")
        for r in bad[:20]:
            print("  missing/mismatch:", r["dataset"], r["relpath"])
        bad_ids = {r["md5"] for r in bad}
        good = [r for r in rows if r["md5"] not in bad_ids]
        if args.write_verified:
            write_manifest(good, Path(args.write_verified))
            print(f"verified manifest: {args.write_verified} ({len(good)} rows)")
        if args.cache:
            build_cache(good, roots, Path(args.cache_dir or resolve_path(cfg["cache"]["dir"])),
                        cfg["cache"]["short_side"], cfg["cache"]["quality"], args.workers)
        sys.exit(1 if bad else 0)

    rows, stats = build_manifest(cfg, classes, roots, args.workers)
    write_manifest(rows, manifest_path)
    write_report(stats, classes, cfg, resolve_path(cfg["report_dir"]))
    print(f"manifest: {manifest_path} ({len(rows)} rows)")
    if args.cache:
        build_cache(rows, roots, Path(args.cache_dir or resolve_path(cfg["cache"]["dir"])),
                    cfg["cache"]["short_side"], cfg["cache"]["quality"], args.workers)


if __name__ == "__main__":
    main()
