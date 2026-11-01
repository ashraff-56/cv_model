"""Benchmark: does using the plate as a ruler make photo grams robust to camera distance and angle?

The same weighed Nutrition5k test dishes are photographed in three conditions, and three estimators are scored:

conditions  overhead       the portion model's own fixed camera (what its 27.3 g test MAE was measured on)
            zoom_<z>       overhead photo zoomed in (z > 1, camera closer) or out (z < 1, camera farther)
            side           side-angle stills: Raspberry Pi cameras at 30/60 degrees, a different camera,
                           angle and distance from training (tools/fetch_side_frames.py)
estimators  constant       median training mass for every dish (the floor any model must beat)
            raw            portion model on the photo as it is
            plate          plate found -> tilt undone -> plate drawn at the reference size -> portion model;
                           falls back to raw when no usable plate is found, so every photo is scored
            plate_any      same, but uses the plate even when it is mostly out of frame (diagnostic)

The reference plate size is the median plate measured on Nutrition5k *training* overhead photos, so nothing
is tuned on test dishes. Nutrition5k plates come in two sizes, so even a perfect detector pays for assuming one.

    python -m smartbite_cv.plate_bench --n5k <mirror root> --side <stills dir> --out reports/plate_bench
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .imageio import load_rgb
from .plate import DetectionCache, PlateDetector, draw, portion_view
from .portion import PortionEstimator, mass_metrics, read_dish_metadata

METHODS = ("constant", "raw", "plate", "plate_any")
ZOOMS = (0.6, 0.8, 1.25)
SEED = 20261101


def zoom(arr: np.ndarray, z: float) -> np.ndarray:
    """Simulate moving the camera: z > 1 crops the centre and enlarges it; z < 1 shrinks the photo onto a
    canvas of its own median border colour (same output size either way)."""
    h, w = arr.shape[:2]
    if z >= 1:
        ch, cw = round(h / z), round(w / z)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return cv2.resize(arr[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_CUBIC)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    out = np.empty_like(arr)
    out[:] = np.median(border, axis=0).astype(arr.dtype)
    sh, sw = round(h * z), round(w * z)
    y0, x0 = (h - sh) // 2, (w - sw) // 2
    out[y0:y0 + sh, x0:x0 + sw] = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
    return out


def bootstrap_ci(values: np.ndarray, stat=np.mean, n: int = 2000, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    stats = np.array([stat(values[i]) for i in idx])
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def summarise(rows: list[dict], constant_g: float, methods: tuple[str, ...] = METHODS,
              pair: tuple[str, str] = ("plate", "raw")) -> dict:
    """Metrics per condition and estimator, with bootstrap CIs, and the paired difference pair[0] - pair[1].
    Side stills of one dish share a dish, so CIs resample dishes, not photos."""
    out = {}
    for cond in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == cond]
        true = np.array([r["true_g"] for r in sub])
        dishes = np.array([r["dish_id"] for r in sub])
        uniq = np.unique(dishes)
        groups = [np.flatnonzero(dishes == d) for d in uniq]
        res = {"photos": len(sub), "dishes": len(uniq),
               "plate_found": float(np.mean([r["found"] for r in sub])),
               "plate_usable": float(np.mean([r["reliable"] for r in sub]))}
        for m in methods:
            pred = np.full(len(sub), constant_g) if m == "constant" else np.array([r[f"{m}_g"] for r in sub])
            met = mass_metrics(pred, true)
            err = np.abs(pred - true)
            rng = np.random.default_rng(SEED)
            boots = []
            for _ in range(2000):
                pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                boots.append(err[pick].mean() / true[pick].mean() * 100)
            met["mae_pct_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            res[m] = met
        a, b = pair
        diff = np.array([abs(r[f"{a}_g"] - r["true_g"]) - abs(r[f"{b}_g"] - r["true_g"]) for r in sub])
        per_dish = np.array([diff[g].mean() for g in groups])
        res[f"{a}_minus_{b}_mae_g"] = {"mean": float(per_dish.mean()), "ci95": bootstrap_ci(per_dish)}
        out[cond] = res
    return out


class Bench:
    def __init__(self, detector: PlateDetector, estimator: PortionEstimator, ref_sigma: dict[str, float],
                 cache: DetectionCache):
        self.det, self.est, self.ref, self.cache = detector, estimator, ref_sigma, cache

    def score(self, img: Image.Image, key: str) -> tuple[dict, object, np.ndarray | None]:
        mu_raw, _ = self.est.log_grams(img)
        row = {"raw_g": math.exp(mu_raw), "found": 0, "reliable": 0, "kind": "", "score": "", "sigma_px": "",
               "tilt_deg": "", "border": ""}
        plate = self.cache.detect(self.det, key, img)
        view = None
        if plate is None:
            row["plate_g"] = row["plate_any_g"] = row["raw_g"]
            return row, None, None
        g = plate.geometry
        view = portion_view(np.asarray(img), plate, self.ref)
        mu_p, _ = self.est.log_grams(Image.fromarray(view))
        row.update({"found": 1, "reliable": int(plate.reliable), "kind": plate.kind, "score": round(plate.score, 3),
                    "sigma_px": round(g.sigma_major, 1), "tilt_deg": round(g.tilt_deg, 1),
                    "border": round(plate.touches_border, 3), "plate_any_g": math.exp(mu_p)})
        row["plate_g"] = row["plate_any_g"] if plate.reliable else row["raw_g"]
        return row, plate, view


def reference_sizes(det: PlateDetector, cache: DetectionCache, n5k: Path, train_ids: list[str], n: int, out: Path,
                    min_count: int = 10) -> dict:
    """Median plate size (long-axis sigma, px in the 640x480 overhead frame) per container kind, measured on
    a seeded sample of training dishes; plates mostly out of frame are skipped."""
    sample = random.Random(SEED).sample(train_ids, min(n, len(train_ids)))
    rows, sizes = [], {}
    for d in sample:
        p = n5k / "imagery" / "realsense_overhead" / d / "rgb.png"
        if not p.exists():
            continue
        plate = cache.detect(det, f"overhead/{d}", load_rgb(p))
        if plate is None:
            rows.append({"dish_id": d, "found": 0})
            continue
        rows.append({"dish_id": d, "found": 1, "kind": plate.kind, "sigma_px": round(plate.geometry.sigma_major, 1),
                     "tilt_deg": round(plate.geometry.tilt_deg, 1), "reliable": int(plate.reliable)})
        if plate.reliable:
            sizes.setdefault(plate.kind, []).append(plate.geometry.sigma_major)
    _write_csv(out / "reference_train_plates.csv", rows)
    ref = {k: float(np.median(v)) for k, v in sizes.items() if len(v) >= min_count}
    if "plate" not in ref:
        raise RuntimeError(f"too few usable training plates to set a reference: { {k: len(v) for k, v in sizes.items()} }")
    return {"ref_sigma_px": ref, "counts": {k: len(v) for k, v in sizes.items()}, "sampled": len(sample),
            "quartiles": {k: [float(q) for q in np.percentile(v, [25, 50, 75])] for k, v in sizes.items() if v}}


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _viz(path: Path, img: Image.Image, plate, view) -> None:
    left = draw(np.asarray(img), plate)
    left = cv2.resize(left, (round(left.shape[1] * 480 / left.shape[0]), 480))
    right = cv2.cvtColor(view, cv2.COLOR_RGB2BGR) if view is not None else np.zeros((480, 640, 3), np.uint8)
    cv2.imwrite(str(path), np.hstack([left, right]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def write_summary(path: Path, summary: dict, ref: dict) -> None:
    lines = ["# Plate-as-ruler benchmark (Nutrition5k test dishes, weighed)", "",
             f"Reference plate size (training photos): {ref['ref_sigma_px']} px long-axis sigma; usable training "
             f"plates {ref['counts']} of {ref['sampled']} sampled.", "",
             "| condition | photos | plate usable | constant | raw | plate | plate_any | plate - raw (g, 95% CI) |",
             "|---|---|---|---|---|---|---|---|"]
    for cond, r in summary.items():
        cell = lambda m: f"{r[m]['mae_g']:.0f} g ({r[m]['mae_pct_of_mean']:.0f}%)"  # noqa: E731
        d = r["plate_minus_raw_mae_g"]
        lines.append(f"| {cond} | {r['photos']} | {r['plate_usable']:.0%} | {cell('constant')} | {cell('raw')} | "
                     f"{cell('plate')} | {cell('plate_any')} | {d['mean']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}] |")
    lines += ["", "MAE in grams (and as % of mean true mass). Negative plate - raw = the plate method is better."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n5k", type=Path, required=True, help="Nutrition5k root with imagery/realsense_overhead")
    ap.add_argument("--side", type=Path, help="directory from tools/fetch_side_frames.py (with frames.csv)")
    ap.add_argument("--meta", type=Path, default=Path("data/raw/nutrition5k"))
    ap.add_argument("--portion", type=Path, default=Path("models/food_weight_finder_no_plate.onnx"),
                    help="first portion model (.onnx file or export folder)")
    ap.add_argument("--out", type=Path, default=Path("reports/plate_bench"))
    ap.add_argument("--ref-train", type=int, default=600, help="training dishes sampled for the reference plate size")
    ap.add_argument("--limit", type=int, help="test dishes (smoke runs only)")
    ap.add_argument("--min-ref", type=int, default=10, help="usable training plates needed per container kind")
    ap.add_argument("--viz", type=int, default=24, help="side stills to save with the detection drawn")
    ap.add_argument("--cache", type=Path, help="detections cache (JSON lines), shared with plate_portion views")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "viz").mkdir(exist_ok=True)
    t0 = time.time()

    meta = read_dish_metadata(sorted((args.meta / "metadata").glob("dish_metadata_cafe*.csv")))
    read_ids = lambda n: [l.strip() for l in open(args.meta / "dish_ids" / "splits" / n, encoding="utf-8") if l.strip()]  # noqa: E731
    train_ids = [d for d in read_ids("depth_train_ids.txt") if d in meta and meta[d]["mass"] > 0]
    test_ids = [d for d in read_ids("depth_test_ids.txt") if d in meta and meta[d]["mass"] > 0][:args.limit]
    constant_g = float(np.median([meta[d]["mass"] for d in train_ids]))

    det, cache = PlateDetector(), DetectionCache(args.cache)
    ref = reference_sizes(det, cache, args.n5k, train_ids, args.ref_train, args.out, args.min_ref)
    print(f"reference plate sizes {ref['ref_sigma_px']} from {ref['counts']} ({time.time() - t0:.0f}s)", flush=True)
    bench = Bench(det, PortionEstimator(args.portion), ref["ref_sigma_px"], cache)

    rows = []
    for i, d in enumerate(test_ids):
        p = args.n5k / "imagery" / "realsense_overhead" / d / "rgb.png"
        if not p.exists():
            continue
        img = load_rgb(p)
        arr = np.asarray(img)
        for cond, im in [("overhead", img)] + [(f"zoom_{z}", Image.fromarray(zoom(arr, z))) for z in ZOOMS]:
            row, _, _ = bench.score(im, f"{cond}/{d}")
            rows.append({"condition": cond, "dish_id": d, "camera": "overhead", "true_g": meta[d]["mass"], **row})
        if (i + 1) % 100 == 0:
            print(f"overhead {i + 1}/{len(test_ids)} ({time.time() - t0:.0f}s)", flush=True)

    if args.side and (args.side / "frames.csv").exists():
        wanted = set(test_ids)
        frames = [r for r in csv.DictReader(open(args.side / "frames.csv", encoding="utf-8")) if r["dish_id"] in wanted]
        viz_pick = set(random.Random(SEED).sample(range(len(frames)), min(args.viz, len(frames))))
        for i, fr in enumerate(frames):
            img = load_rgb(args.side / fr["file"])
            row, plate, view = bench.score(img, f"side/{fr['dish_id']}_{fr['camera']}")
            rows.append({"condition": "side", "dish_id": fr["dish_id"], "camera": fr["camera"],
                         "true_g": meta[fr["dish_id"]]["mass"], **row})
            if i in viz_pick:
                _viz(args.out / "viz" / f"{fr['dish_id']}_{fr['camera']}.jpg", img, plate, view)
            if (i + 1) % 200 == 0:
                print(f"side {i + 1}/{len(frames)} ({time.time() - t0:.0f}s)", flush=True)

    _write_csv(args.out / "predictions.csv", rows)
    summary = summarise(rows, constant_g)
    side = [r for r in rows if r["condition"] == "side"]
    if side:   # per camera (30 vs 60 degrees) and on stills where the whole plate is in frame
        for cam in sorted({r["camera"] for r in side}):
            summary.update({f"side_{k}": v for k, v in summarise(
                [{**r, "condition": cam} for r in side if r["camera"] == cam], constant_g).items()})
        usable = [{**r, "condition": "side_plate_in_frame"} for r in side if r["reliable"]]
        if usable:
            summary.update(summarise(usable, constant_g))
    result = {"reference": ref, "constant_g": constant_g, "summary": summary, "minutes": (time.time() - t0) / 60}
    (args.out / "plate_bench.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_summary(args.out / "summary.md", summary, ref)
    print((args.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
