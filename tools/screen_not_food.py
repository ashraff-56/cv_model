"""Flag source images that a trained model confidently calls not_food (maps, people, product boxes, ...).

Used once on the SMF dataset, whose class folders contain web junk. Flagged files are written to a CSV for
visual review; only reviewed rows go into the exclusion list used by prepare (configs/data_v2.yaml: exclude).

    python tools/screen_not_food.py --run runs/efficientnet_b0 --root <dataset root> --dataset smf --out data/splits/smf_not_food_flags.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from smartbite_cv.audit import PathDataset, _images  # noqa: E402
from smartbite_cv.data import build_transforms  # noqa: E402
from smartbite_cv.metrics import calibrate  # noqa: E402
from smartbite_cv.modeling import infer_logits, load_trained  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="dataset root (relpaths are relative to it)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    model, meta = load_trained(args.run / "best.pt")
    t = json.load(open(args.run / "eval" / "calibration.json", encoding="utf-8"))["temperature"]
    paths = _images(args.root)
    tf = build_transforms(meta["img_size"], train=False, crop_pct=meta["crop_pct"])
    logits, flag = infer_logits(model, DataLoader(PathDataset(paths, tf), batch_size=32, num_workers=0), "cpu")
    p_nf = calibrate(logits, t)[:, meta["classes"].index("not_food")]
    p_nf[flag == -2] = -1.0   # unreadable: prepare drops these anyway
    print(f"{int((flag == -2).sum())} unreadable files skipped")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "relpath", "p_not_food"])
        for p, s in sorted(zip(paths, p_nf), key=lambda x: -x[1]):
            if s >= args.threshold:
                w.writerow([args.dataset, p.relative_to(args.root).as_posix(), f"{s:.3f}"])
    print(f"{int((p_nf >= args.threshold).sum())} of {len(paths)} images flagged (p_not_food >= {args.threshold})")


if __name__ == "__main__":
    main()
