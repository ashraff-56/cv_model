"""Drop external test images that duplicate any image in the training manifest (train, val or test).

Wikipedia photos are popular and get scraped into web datasets, so an "external" photo may already be in
the training data. Uses the same rules as smartbite_cv.prepare: pHash within `max_hamming` (also against
the mirror image) or ImageNet-embedding cosine >= `min_cosine`. Writes <external>/dedupe.csv and a manifest
of the survivors.

    python tools/dedupe_external.py --external data/external/wikipedia --csv wikipedia_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from smartbite_cv.config import load_config, resolve_path  # noqa: E402
from smartbite_cv.embed_dedupe import embed  # noqa: E402
from smartbite_cv.imageio import load_rgb  # noqa: E402
from smartbite_cv.phash import _popcount, phash_pair  # noqa: E402
from smartbite_cv.prepare import dataset_roots, read_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--external", type=Path, required=True)
    ap.add_argument("--csv", required=True, help="collection manifest inside --external")
    ap.add_argument("--data-config", default="configs/data_v2.yaml")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.data_config)
    rows = read_manifest(resolve_path(cfg["manifest"]))
    roots = dataset_roots(cfg, "local")
    ext = list(csv.DictReader(open(args.external / args.csv, encoding="utf-8")))

    # pHash against every manifest image (manifest stores the image hash; compare to both external variants)
    man_hash = np.array([int(r["phash"], 16) for r in rows], dtype=np.uint64)
    max_d = cfg["dedupe"]["max_hamming"]
    emb_cfg = cfg["dedupe"]["embedding"]
    cache_dir = resolve_path(cfg["cache"]["dir"])
    man_paths = [(cache_dir / f"{r['md5']}.jpg") if (cache_dir / f"{r['md5']}.jpg").exists()
                 else roots[r["dataset"]] / r["relpath"] for r in rows]
    man_emb = embed(man_paths, emb_cfg["model"], workers=0, batch_size=32,
                    cache_file=resolve_path(emb_cfg["cache_file"]), keys=[r["md5"] for r in rows])
    ext_paths = [args.external / e["file"] for e in ext]
    ext_emb = embed(ext_paths, emb_cfg["model"], workers=0, batch_size=32)
    sims = ext_emb @ man_emb.T
    out = []
    for i, e in enumerate(ext):
        h, hf = phash_pair(load_rgb(ext_paths[i]))
        d = np.minimum(_popcount(man_hash ^ np.uint64(h)), _popcount(man_hash ^ np.uint64(hf)))
        j_ph, j_cos = int(d.argmin()), int(sims[i].argmax())
        dup = d[j_ph] <= max_d or sims[i, j_cos] >= emb_cfg["min_cosine"]
        out.append({**e, "duplicate": int(dup), "min_hamming": int(d[j_ph]), "max_cosine": round(float(sims[i, j_cos]), 4),
                    "nearest_label": rows[j_cos]["label"], "nearest_split": rows[j_cos]["split"],
                    "nearest": f"{rows[j_cos]['dataset']}/{rows[j_cos]['relpath']}"})
    fields = list(out[0].keys()) if out else []
    with open(args.external / "dedupe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    n_dup = sum(o["duplicate"] for o in out)
    print(f"{len(out)} external images; {n_dup} duplicate an image in the training manifest and are excluded")


if __name__ == "__main__":
    main()
