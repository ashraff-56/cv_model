"""Plate-relative portion model: grams from a top-down, plate-only view, for any camera angle or distance.

The first portion model (portion.py) learned grams from Nutrition5k's fixed overhead camera, so it only knows the
scale of that camera; on side-angle photos it overestimated 2-3x even after the tilt was undone (plate_bench).
This model is trained on what any photo with a plate can be turned into: plate found, tilt undone, plate drawn
at one fixed size, everything off the plate grey (plate.plate_only_view). Training mixes the overhead photos with
side-angle stills of the same training dishes (30 and 60 degrees, another camera), so it also sees foreshortened
food. It still has to infer the plate's real size from what is on it (Nutrition5k uses roughly 21 and 28 cm
plates): that is the error an unknown plate size causes in the app, and the test numbers include it.

Splits: official Nutrition5k train/test dish lists; validation = the same 10% of training dishes as portion.py.
Plates cut off by the frame are left out of training (their scale is wrong) but every test photo is scored;
with no plate at all the estimate falls back to the median training mass (the app falls back to a typical serving).

    python -m smartbite_cv.plate_portion prefill --n5k <root> --side-train <dir> --side-test <dir> --cache c0.jsonl --shard 0/2
    python -m smartbite_cv.plate_portion views --n5k <root> --side-train <dir> --side-test <dir> --cache all.jsonl --out <views dir>
    python -m smartbite_cv.plate_portion train --views <views dir> --config configs/train/plate_portion.yaml --out runs/plate_portion
    python -m smartbite_cv.plate_portion evaluate --run runs/plate_portion --views <views dir> [--old-preds predictions.csv]
    python -m smartbite_cv.plate_portion export --run runs/plate_portion --out artifacts/plate_portion
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import load_config, resolve_path
from .data import IMAGENET_MEAN, IMAGENET_STD
from .imageio import load_rgb
from .plate import GRAY, PLATE_VIEW_SIGMA, PLATE_VIEW_SIZE, DetectionCache, PlateDetector, plate_only_view
from .plate_bench import ZOOMS, summarise, zoom
from .portion import (PortionNet, export_onnx, fit, fit_sigma_scale, interval_coverage, load_portion,
                      mass_metrics, predict, read_dish_metadata, split_ids)

VIEW_FIELDS = ["key", "dish_id", "split", "condition", "camera", "true_g", "found", "reliable", "kind",
               "tilt_deg", "sigma_px", "border", "file"]


# ---------------------------------------------------------------- views
def view_jobs(cfg: dict, n5k: Path, side_train: Path | None, side_test: Path | None) -> tuple[list, dict]:
    """Every photo the views (and plate_bench) need: (key, dish, split, condition, camera, loader)."""
    meta_dir = resolve_path(cfg["data"]["metadata_dir"])
    meta = read_dish_metadata(sorted((meta_dir / "metadata").glob("dish_metadata_cafe*.csv")))
    split_of = {d: s for s, ids in split_ids(cfg).items() for d in ids}
    jobs = []   # (key, dish, split, condition, camera, loader)
    for d, s in split_of.items():
        if d not in meta or meta[d]["mass"] <= 0:
            continue
        p = n5k / "imagery" / "realsense_overhead" / d / "rgb.png"
        if p.exists():
            jobs.append((f"overhead/{d}", d, s, "overhead", "overhead", lambda p=p: load_rgb(p)))
            if s == "test":
                for z in ZOOMS:
                    jobs.append((f"zoom_{z}/{d}", d, s, f"zoom_{z}", "overhead",
                                 lambda p=p, z=z: Image.fromarray(zoom(np.asarray(load_rgb(p)), z))))
    for side in (side_train, side_test):
        if side and (side / "frames.csv").exists():
            for fr in csv.DictReader(open(side / "frames.csv", encoding="utf-8")):
                d = fr["dish_id"]
                if d in split_of and d in meta and meta[d]["mass"] > 0:
                    jobs.append((f"side/{d}_{fr['camera']}", d, split_of[d], "side", fr["camera"],
                                 lambda p=side / fr["file"]: load_rgb(p)))
    return jobs, meta


def prefill(jobs: list, cache: DetectionCache, shard: int, shards: int) -> int:
    """Detect plates for this shard's uncached photos (one process per GPU; caches are merged afterwards)."""
    todo = [j for j in jobs if j[0] not in cache.records][shard::shards]
    det, t0 = PlateDetector(), time.time()
    for i, (key, *_, load) in enumerate(todo):
        cache.detect(det, key, load())
        if (i + 1) % 500 == 0:
            print(f"shard {shard}: {i + 1}/{len(todo)} ({(i + 1) / (time.time() - t0):.1f}/s)", flush=True)
    return len(todo)


def build_views(cfg: dict, n5k: Path, side_train: Path | None, side_test: Path | None, out: Path,
                cache: DetectionCache, det: PlateDetector | None) -> list[dict]:
    jobs, meta = view_jobs(cfg, n5k, side_train, side_test)
    (out / "img").mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, (key, d, s, cond, cam, load) in enumerate(jobs):
        img = load()
        plate = cache.detect(det, key, img)
        row = {"key": key, "dish_id": d, "split": s, "condition": cond, "camera": cam, "true_g": meta[d]["mass"],
               "found": int(plate is not None), "reliable": int(plate is not None and plate.reliable),
               "kind": "", "tilt_deg": "", "sigma_px": "", "border": "", "file": ""}
        if plate is not None:
            name = key.replace("/", "__") + ".jpg"
            view = plate_only_view(np.asarray(img), plate)
            cv2.imwrite(str(out / "img" / name), cv2.cvtColor(view, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            row.update({"kind": plate.kind, "tilt_deg": round(plate.geometry.tilt_deg, 1),
                        "sigma_px": round(plate.geometry.sigma_major, 1), "border": round(plate.touches_border, 3),
                        "file": f"img/{name}"})
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"views {i + 1}/{len(jobs)} ({(i + 1) / (time.time() - t0):.1f}/s)", flush=True)
    with open(out / "views.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VIEW_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def read_views(views: Path) -> list[dict]:
    rows = list(csv.DictReader(open(views / "views.csv", encoding="utf-8")))
    for r in rows:
        r["true_g"], r["found"], r["reliable"] = float(r["true_g"]), int(r["found"]), int(r["reliable"])
    return rows


# ---------------------------------------------------------------- data
class ViewDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path, transform):
        self.rows, self.root, self.transform = rows, root, transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        return self.transform(load_rgb(self.root / r["file"])), torch.tensor(math.log(r["true_g"]), dtype=torch.float32)


def view_transforms(train: bool, aug: dict | None = None) -> T.Compose:
    ops = []
    if train:
        aug = aug or {}
        ops += [T.RandomAffine(degrees=aug.get("rotate", 180), scale=tuple(aug.get("scale", [0.95, 1.05])), fill=GRAY,
                               interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(), T.ColorJitter(*aug.get("color_jitter", [0.2, 0.2, 0.15, 0.02]))]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def loaders_for(cfg: dict, rows: list[dict], views: Path, workers: int | None = None) -> dict:
    """train/val: views with a whole plate in frame; test: every photo with a plate found (scored separately)."""
    pick = {"train": lambda r: r["split"] == "train" and r["reliable"],
            "val": lambda r: r["split"] == "val" and r["reliable"],
            "test": lambda r: r["split"] == "test" and r["found"]}
    workers = cfg["num_workers"] if workers is None else workers
    out = {}
    for s, keep in pick.items():
        sub = [r for r in rows if keep(r)]
        out[s] = (sub, DataLoader(ViewDataset(sub, views, view_transforms(s == "train", cfg.get("aug"))),
                                  batch_size=cfg["batch_size"], shuffle=(s == "train"), num_workers=workers,
                                  drop_last=(s == "train" and len(sub) > cfg["batch_size"]),
                                  pin_memory=torch.cuda.is_available()))
    return out


# ---------------------------------------------------------------- train / evaluate
def train(cfg: dict, views: Path, out: Path) -> None:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_views(views)
    data = loaders_for(cfg, rows, views)
    tr = data["train"][0]
    model_cfg = {**cfg["model"], "init_log_mass": float(np.mean([math.log(r["true_g"]) for r in tr]))}
    model = PortionNet(model_cfg).to(device)
    by = lambda s, c: sum(1 for r in data[s][0] if r["condition"] == c)  # noqa: E731
    print(f"device={device} train={len(tr)} (overhead {by('train', 'overhead')}, side {by('train', 'side')}) "
          f"val={len(data['val'][0])} dishes={len({r['dish_id'] for r in tr})}", flush=True)
    fit(cfg, model, model_cfg, {s: data[s][1] for s in ("train", "val")}, out, device)
    cfg_out = json.loads((out / "config_resolved.json").read_text(encoding="utf-8"))
    cfg_out["views"] = {"size": PLATE_VIEW_SIZE, "sigma": PLATE_VIEW_SIGMA}
    (out / "config_resolved.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")


def evaluate(run: Path, views: Path, old_preds: Path | None = None) -> dict:
    """Every test photo scored: plate-relative model where a plate was found, else the median training mass.
    With plate_bench's predictions.csv, the first portion model (raw photo) is scored on the same photos."""
    cfg = json.loads((run / "config_resolved.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_portion(run / "best.pt", device)
    rows = read_views(views)
    data = loaders_for(cfg, rows, views)
    mu_v, lv_v, y_v = predict(model, data["val"][1], device)
    mu_t, lv_t, y_t = predict(model, data["test"][1], device)
    s = fit_sigma_scale(mu_v, lv_v, y_v)
    sig_t = np.exp(0.5 * lv_t) * s
    train_g = [r["true_g"] for r in rows if r["split"] == "train" and r["condition"] == "overhead"]
    constant_g = float(np.median(train_g))
    pred_of = {r["key"]: float(np.exp(m)) for r, m in zip(data["test"][0], mu_t)}
    old = {}
    if old_preds and old_preds.exists():
        for r in csv.DictReader(open(old_preds, encoding="utf-8")):
            cam = r["camera"]
            key = f"side/{r['dish_id']}_{cam}" if r["condition"] == "side" else f"{r['condition']}/{r['dish_id']}"
            old[key] = (float(r["raw_g"]), float(r["plate_g"]))
    test = []
    for r in rows:
        if r["split"] != "test":
            continue
        row = {**r, "new_g": pred_of.get(r["key"], constant_g)}
        if old:
            if r["key"] not in old:
                continue   # keep both estimators on exactly the same photos
            row["raw_g"], row["oldplate_g"] = old[r["key"]]
        test.append(row)
    methods = ("constant", "new") + (("raw", "oldplate") if old else ())
    pair = ("new", "raw") if old else ("new", "constant")
    summary = summarise(test, constant_g, methods, pair)
    side = [r for r in test if r["condition"] == "side"]
    for cam in sorted({r["camera"] for r in side}):
        summary.update({f"side_{k}": v for k, v in summarise(
            [{**r, "condition": cam} for r in side if r["camera"] == cam], constant_g, methods, pair).items()})
    in_frame = [{**r, "condition": "side_plate_in_frame"} for r in side if r["reliable"]]
    if in_frame:
        summary.update(summarise(in_frame, constant_g, methods, pair))
    by_cond = {}
    for c in sorted({r["condition"] for r in data["test"][0]}):
        idx = [i for i, r in enumerate(data["test"][0]) if r["condition"] == c]
        by_cond[c] = {"coverage_68": interval_coverage(mu_t[idx], sig_t[idx], y_t[idx], 1.0),
                      "coverage_95": interval_coverage(mu_t[idx], sig_t[idx], y_t[idx], 1.96)}
    res = {"constant_g": constant_g, "summary": summary,
           "uncertainty": {"sigma_scale_fitted_on_val": s, "by_condition": by_cond,
                           "median_sigma_log": float(np.median(sig_t))},
           "val": mass_metrics(np.exp(mu_v), np.exp(y_v)),
           "prior_used_for_fusion_eval": {"typical_g": float(np.exp(np.mean(np.log(train_g)))),
                                          "sigma_log": float(np.std(np.log(train_g)))},
           "model": {"testMassMAE": summary.get("overhead", {}).get("new", {})}}
    (run / "eval").mkdir(exist_ok=True)
    (run / "eval" / "plate_portion_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    # export_onnx (portion.py) reads these two files
    (run / "eval" / "portion_metrics.json").write_text(json.dumps({
        "uncertainty": {"sigma_scale_fitted_on_val": s}, "prior_used_for_fusion_eval": res["prior_used_for_fusion_eval"],
        "model": {k: summary["overhead"]["new"][k] for k in ("mae_g", "mae_pct_of_mean", "median_ape_pct")}
        if "overhead" in summary else {"mae_g": None, "mae_pct_of_mean": None, "median_ape_pct": None}},
        indent=2), encoding="utf-8")
    with open(run / "eval" / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["key", "condition", "camera", "dish_id", "true_g", "found", "reliable", "new_g"] + \
                 (["raw_g", "oldplate_g"] if old else [])
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(test)
    write_table(run / "eval" / "summary.md", summary, methods, pair)
    print((run / "eval" / "summary.md").read_text(encoding="utf-8"))
    return res


def write_table(path: Path, summary: dict, methods: tuple, pair: tuple) -> None:
    names = {"constant": "same guess for all", "new": "plate model (new)", "raw": "first model, raw photo",
             "oldplate": "first model + plate scaling"}
    key = f"{pair[0]}_minus_{pair[1]}_mae_g"
    lines = ["# Plate-relative portion model: Nutrition5k test dishes (weighed)", "",
             "| condition | photos | plate found | plate in frame | " + " | ".join(names[m] for m in methods)
             + f" | {pair[0]} - {pair[1]} (g, 95% CI) |", "|---" * (5 + len(methods)) + "|"]
    for cond, r in summary.items():
        d = r[key]
        cells = " | ".join(f"{r[m]['mae_g']:.0f} g ({r[m]['mae_pct_of_mean']:.0f}%)" for m in methods)
        lines.append(f"| {cond} | {r['photos']} | {r['plate_found']:.0%} | {r['plate_usable']:.0%} | {cells} | "
                     f"{d['mean']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}] |")
    lines += ["", "Mean absolute error in grams (and as % of mean true mass). CIs resample dishes. Negative difference = "
              f"{names[pair[0]]} is better."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PlatePortionEstimator:
    """Server-side grams from a photo: plate found -> plate-only view -> plate-relative ONNX model.

    The detector needs PyTorch + transformers (OWLv2 + SAM 2.1); the portion model itself runs on onnxruntime.
    Returns None when no plate is found, so the caller can fall back to another estimate."""

    PARTLY_OUT = "Part of the plate is outside the photo. Retake it with the whole plate in view for a better estimate."

    def __init__(self, model_dir: str | Path, detector: PlateDetector | None = None, threads: int | None = None):
        from .portion import PortionEstimator
        self.portion = PortionEstimator(model_dir, threads)
        self.detector = detector or PlateDetector()

    def estimate(self, img: Image.Image, typical_serving_g: float | None = None) -> dict | None:
        plate = self.detector.detect(img)
        if plate is None:
            return None
        q = self.portion.estimate(Image.fromarray(plate_only_view(np.asarray(img), plate)), typical_serving_g)
        q["plate"] = {"kind": plate.kind, "tiltDeg": round(plate.geometry.tilt_deg), "inFrame": plate.reliable}
        if not plate.reliable:
            q["warning"] = self.PARTLY_OUT
        return q


def export(run: Path, out_dir: Path) -> Path:
    path = export_onnx(run, out_dir)
    meta = json.loads((out_dir / "portion_meta.json").read_text(encoding="utf-8"))
    meta.update({"input": "plate_only_view", "viewSize": PLATE_VIEW_SIZE, "viewSigma": PLATE_VIEW_SIGMA,
                 "trainedOn": "Nutrition5k overhead + side-angle stills as plate-only views (official train split)"})
    (out_dir / "portion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("views", "prefill"):
        v = sub.add_parser(name)
        v.add_argument("--config", action="append")
        v.add_argument("--set", action="append", default=[])
        v.add_argument("--n5k", type=Path, required=True)
        v.add_argument("--side-train", type=Path)
        v.add_argument("--side-test", type=Path)
        v.add_argument("--cache", type=Path)
        v.add_argument("--out", type=Path, required=(name == "views"))
        v.add_argument("--shard", default="0/1", help="prefill only: i/n, e.g. one process per GPU")
    t = sub.add_parser("train")
    t.add_argument("--config", action="append", required=True)
    t.add_argument("--set", action="append", default=[])
    t.add_argument("--views", type=Path, required=True)
    t.add_argument("--out", type=Path, default=Path("runs/plate_portion"))
    e = sub.add_parser("evaluate")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--views", type=Path, required=True)
    e.add_argument("--old-preds", type=Path)
    x = sub.add_parser("export")
    x.add_argument("--run", type=Path, required=True)
    x.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd in ("views", "prefill"):
        cache = DetectionCache(args.cache)
        cfg = load_config(*(args.config or ["configs/train/plate_portion.yaml"]), overrides=args.set)
        jobs, _ = view_jobs(cfg, args.n5k, args.side_train, args.side_test)
        if args.cmd == "prefill":
            i, n = (int(x) for x in args.shard.split("/"))
            print(f"shard {i}/{n}: detected {prefill(jobs, cache, i, n)} photos")
            return
        det = PlateDetector() if any(j[0] not in cache.records for j in jobs) else None
        rows = build_views(cfg, args.n5k, args.side_train, args.side_test, args.out, cache, det)
        print(f"{len(rows)} photos, plate found in {sum(r['found'] for r in rows)}, "
              f"whole plate in frame in {sum(r['reliable'] for r in rows)}")
    elif args.cmd == "train":
        train(load_config(*args.config, overrides=args.set), args.views, args.out)
    elif args.cmd == "evaluate":
        evaluate(args.run, args.views, args.old_preds)
    else:
        print(f"exported {export(args.run, args.out)}")


if __name__ == "__main__":
    main()
