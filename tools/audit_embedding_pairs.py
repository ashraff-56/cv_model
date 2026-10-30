"""Contact sheets of image pairs per cosine-similarity bin, to choose the embedding-dedupe threshold.

    python tools/audit_embedding_pairs.py --pairs reports/data/embedding_pairs.csv --out reports/data/audit
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from smartbite_cv.config import load_config  # noqa: E402
from smartbite_cv.imageio import load_rgb  # noqa: E402
from smartbite_cv.prepare import dataset_roots  # noqa: E402

BINS = [(0.85, 0.88), (0.88, 0.90), (0.90, 0.92), (0.92, 0.94), (0.94, 0.96), (0.96, 1.01)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-bin", type=int, default=16)
    args = ap.parse_args()
    roots = dataset_roots(load_config(ROOT / "configs" / "data.yaml"), "local")
    with open(args.pairs, newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for lo, hi in BINS:
        sel = [p for p in pairs if lo <= float(p["cosine"]) < hi]
        same = sum(p["label_a"] == p["label_b"] for p in sel)
        print(f"[{lo:.2f}, {hi:.2f}): {len(sel)} pairs ({same} same-label)")
        sample = rng.sample(sel, min(args.per_bin, len(sel)))
        s, cols = 120, 4
        sheet = Image.new("RGB", (cols * (2 * s + 12), ((len(sample) + cols - 1) // cols) * (s + 14)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, p in enumerate(sample):
            x, y = (k % cols) * (2 * s + 12), (k // cols) * (s + 14)
            for m, side in enumerate("ab"):
                img = load_rgb(roots[p[f"dataset_{side}"]] / p[f"relpath_{side}"])
                img.thumbnail((s, s))
                sheet.paste(img, (x + m * s, y))
            draw.text((x, y + s), f"{float(p['cosine']):.3f} {p['label_a'][:8]}/{p['label_b'][:8]}", fill="black")
        sheet.save(args.out / f"pairs_{lo:.2f}_{hi:.2f}.jpg", quality=85)


if __name__ == "__main__":
    main()
