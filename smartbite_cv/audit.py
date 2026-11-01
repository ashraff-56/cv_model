"""Integrity audit of a trained run: checks that a good test score is not an artefact.

Each check targets a known way a model can look good on paper and still be bad in use:

  overfitting  clean train accuracy (eval transform, no augmentation) vs val vs test; learning curves
  leakage      test accuracy vs similarity to the nearest *training* image, measured with an embedding
               that was not used for de-duplication; 1-nearest-neighbour baseline on generic features
  shortcuts    per-(class, source) test accuracy; Food-5K *food* photos (never trained on) must not be
               called not_food, otherwise not_food learned "Food-5K style"; Grad-CAM heatmaps
  robustness   test accuracy under phone-like corruptions (blur, noise, low light, JPEG, crops, ...)
  open set     dishes outside the 31 classes: how often they would be pre-selected (high band)
  comparison   McNemar test of the test-set errors against a second run (--compare)

Official test numbers are the ones `evaluate` computed on Kaggle (eval/logits.npz); this script re-runs
the checkpoint locally on CPU and first checks that it reproduces those logits.

    python -m smartbite_cv.audit --run runs/efficientnet_b0 --compare runs/mobilenetv2
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageEnhance, ImageFilter  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from .bands import NOT_FOOD, BandThresholds, assign_band  # noqa: E402
from .config import REPO_ROOT, load_classes, resolve_path  # noqa: E402
from .data import IMAGENET_MEAN, IMAGENET_STD, ManifestDataset, build_transforms  # noqa: E402
from .embed_dedupe import embed  # noqa: E402
from .imageio import load_rgb  # noqa: E402
from .metrics import bootstrap_ci, calibrate, nll, summarize, topk_correct  # noqa: E402
from .modeling import infer_logits, load_trained, split_rows  # noqa: E402

SIM_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 1.0001]
INDEPENDENT_EMBEDDING = "mobilenetv2_100.ra_in1k"   # de-duplication used efficientnet_b0.ra_in1k features
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------- corruptions (module-level: picklable)
def _rng(img: Image.Image) -> np.random.Generator:
    """Seed from the pixels so a corrupted image is identical across runs and DataLoader workers."""
    return np.random.default_rng(zlib.crc32(np.asarray(img)[::9, ::9].tobytes()))


def clean(img):
    return img


def gaussian_blur(img):
    return img.filter(ImageFilter.GaussianBlur(2))


_MOTION = ImageFilter.Kernel((5, 5), [0] * 10 + [1] * 5 + [0] * 10, scale=5)


def motion_blur(img):
    return img.filter(_MOTION)


def jpeg_q15(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=15)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_noise(img):
    a = np.asarray(img, dtype=np.float32)
    a = a + _rng(img).normal(0.0, 0.08 * 255, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def low_light(img):
    return ImageEnhance.Brightness(img).enhance(0.4)


def overexposed(img):
    return ImageEnhance.Brightness(img).enhance(1.6)


def low_resolution(img):
    w, h = img.size
    small = img.resize((max(1, w // 4), max(1, h // 4)), Image.BILINEAR)
    return small.resize((w, h), Image.BICUBIC)


def tight_crop(img):
    w, h = img.size
    cw, ch = int(w * 0.6), int(h * 0.6)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def rotated_90(img):
    return img.rotate(90, expand=True)


CORRUPTIONS = {f.__name__: f for f in (clean, gaussian_blur, motion_blur, jpeg_q15, gaussian_noise, low_light,
                                        overexposed, low_resolution, tight_crop, rotated_90)}


# ---------------------------------------------------------------- data helpers
class PathDataset(Dataset):
    """Unlabelled image files (out-of-scope probes)."""

    def __init__(self, paths: list[Path], transform):
        self.paths, self.transform = paths, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            return self.transform(load_rgb(self.paths[i])), -1
        except Exception:  # noqa: BLE001 - corrupt file: grey placeholder, label -2 marks it unreadable
            return self.transform(Image.new("RGB", (256, 256), (128, 128, 128))), -2


BATCH = 32   # CPU inference batch; small to keep memory low on a 16 GB laptop


CACHE_DIR: Path | None = None   # set in main(): per-step logits are saved so a crash loses no finished work


def _logits(model, ds, workers: int, key: str | None = None) -> np.ndarray:
    path = CACHE_DIR / f"logits_{key}.npy" if CACHE_DIR is not None and key else None
    if path is not None and path.exists():
        cached = np.load(path)
        if len(cached) == len(ds):
            return cached
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=workers)
    logits = infer_logits(model, loader, "cpu")[0]
    if path is not None:
        np.save(path, logits)
    return logits


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


# ---------------------------------------------------------------- pure analysis functions (unit-tested)
def similarity_bins(max_sim: np.ndarray, correct: np.ndarray, nn_correct: np.ndarray,
                    edges=SIM_BINS) -> list[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (max_sim >= lo) & (max_sim < hi)
        out.append({"range": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": int(m.sum()),
                    "model_top1": float(correct[m].mean()) if m.any() else None,
                    "nn1_top1": float(nn_correct[m].mean()) if m.any() else None})
    return out


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on paired 0/1 outcomes (Dietterich 1998)."""
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    n = a_only + b_only
    p = float(binomtest(min(a_only, b_only), n, 0.5).pvalue) if n else 1.0
    return {"a_right_b_wrong": a_only, "a_wrong_b_right": b_only, "p_value": p}


def paired_bootstrap_diff(correct_a: np.ndarray, correct_b: np.ndarray, n_boot: int = 2000,
                          seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct_a), size=(n_boot, len(correct_a)))
    diffs = correct_a.astype(float)[idx].mean(1) - correct_b.astype(float)[idx].mean(1)
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def band_counts(probs: np.ndarray, classes: list[str], t: BandThresholds) -> dict:
    bands = [assign_band(float(p.max()), classes[int(p.argmax())], t)[0] for p in probs]
    c = Counter(bands)
    return {b: c.get(b, 0) / max(len(bands), 1) for b in ("high", "medium", "low")}


# ---------------------------------------------------------------- checks
def check_overfitting(run: Path, train_probs, train_y, train_logits, val, test, temperature, classes) -> dict:
    hist = list(csv.DictReader(open(run / "history.csv", encoding="utf-8")))
    s2 = [h for h in hist if h["stage"] == "2"]
    vloss = [float(h["val_loss"]) for h in s2]
    vf1 = [float(h["val_macro_f1"]) for h in s2]
    best = int(np.argmax(vf1))
    tr = summarize(train_probs, train_y, classes)
    rows = {"train_clean": {"top1": tr["top1"], "top3": tr["top3"], "macro_f1": tr["macro_f1"],
                            "nll": nll(train_logits, train_y, temperature)}}
    for name, (lg, y) in (("val", val), ("test", test)):
        s = summarize(calibrate(lg, temperature), y, classes)
        rows[name] = {"top1": s["top1"], "top3": s["top3"], "macro_f1": s["macro_f1"], "nll": nll(lg, y, temperature)}
    return {
        "splits": rows,
        "gap_train_test_top1": rows["train_clean"]["top1"] - rows["test"]["top1"],
        "gap_val_test_top1": rows["val"]["top1"] - rows["test"]["top1"],
        "stage2_epochs_run": len(s2),
        "best_epoch_stage2": best + 1,
        "val_loss_min_epoch": int(np.argmin(vloss)) + 1,
        "val_loss_min": min(vloss),
        "val_loss_final": vloss[-1],
        "val_loss_rise_after_min": vloss[-1] - min(vloss),
        "history": hist,
    }


def check_leakage(test_rows, train_rows, correct, test_y, train_y, paths_by_md5, workers) -> dict:
    out = {}
    train_y = np.asarray(train_y)
    sources = {
        "efficientnet_b0 ImageNet (used for de-duplication)": None,
        f"{INDEPENDENT_EMBEDDING} ImageNet (independent)": INDEPENDENT_EMBEDDING,
    }
    for name, model_name in sources.items():
        if model_name is None:
            data = np.load(REPO_ROOT / "data" / "cache" / "embeddings_effb0.npz")
            emb = dict(zip(data["keys"].tolist(), data["emb"]))
            te = np.stack([emb[r["md5"]] for r in test_rows])
            tr = np.stack([emb[r["md5"]] for r in train_rows])
        else:
            md5s = [r["md5"] for r in train_rows] + [r["md5"] for r in test_rows]
            e = embed([paths_by_md5[m] for m in md5s], model_name=model_name, workers=workers, batch_size=BATCH,
                      cache_file=REPO_ROOT / "data" / "cache" / f"embeddings_{model_name.split('.')[0]}.npz",
                      keys=md5s)
            tr, te = e[:len(train_rows)], e[len(train_rows):]
        sims = te @ tr.T
        j = sims.argmax(1)
        max_sim = sims[np.arange(len(te)), j]
        nn_correct = train_y[j] == test_y
        far = max_sim < 0.6
        out[name] = {
            "max_sim_quantiles": {q: float(np.quantile(max_sim, q)) for q in (0.5, 0.9, 0.99, 1.0)},
            "nn1_top1_all": float(nn_correct.mean()),
            "bins": similarity_bins(max_sim, correct, nn_correct),
            "model_top1_far_lt_0.6": float(correct[far].mean()) if far.any() else None,
            "n_far_lt_0.6": int(far.sum()),
            "closest_pairs": [{"test": test_rows[i]["md5"], "train": train_rows[j[i]]["md5"],
                               "cosine": float(max_sim[i])} for i in np.argsort(-max_sim)[:12]],
        }
    return out


def check_sources(test_rows, correct) -> dict:
    by_ds, by_cls_ds, cls_sources = defaultdict(list), defaultdict(list), defaultdict(set)
    for r, c in zip(test_rows, correct):
        by_ds[r["dataset"]].append(c)
        by_cls_ds[(r["label"], r["dataset"])].append(c)
        cls_sources[r["label"]].add(r["dataset"])
    return {
        "per_source": {d: {"n": len(v), "top1": float(np.mean(v))} for d, v in sorted(by_ds.items())},
        "single_source_classes": sorted(c for c, s in cls_sources.items() if len(s) == 1),
        "per_class_source": [{"class": c, "source": d, "n": len(v), "top1": float(np.mean(v))}
                             for (c, d), v in sorted(by_cls_ds.items()) if len(cls_sources[c]) > 1],
    }


def gradcam(model, x: torch.Tensor, cls: int) -> np.ndarray:
    """Grad-CAM (Selvaraju et al. 2017) on the last convolutional feature map."""
    feats = model.forward_features(x)
    feats.retain_grad()
    logits = model.forward_head(feats)
    model.zero_grad(set_to_none=True)
    logits[0, cls].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats).sum(1))[0].detach().numpy()
    return cam / (cam.max() + 1e-8)


def gradcam_figure(model, ds: ManifestDataset, probs, y, classes, path: Path, title: str, n_each: int = 6) -> list:
    pred, conf = probs.argmax(1), probs.max(1)
    rng = np.random.default_rng(0)
    right_idx = np.where(pred == y)[0]
    right = rng.choice(right_idx, size=min(n_each, len(right_idx)), replace=False)
    wrong_idx = np.where(pred != y)[0]
    wrong = wrong_idx[np.argsort(-conf[wrong_idx])][:n_each]
    picks = [(int(i), "correct") for i in right] + [(int(i), "error") for i in wrong]
    mean, std = np.array(IMAGENET_MEAN)[:, None, None], np.array(IMAGENET_STD)[:, None, None]
    fig, axes = plt.subplots(3, 4, figsize=(13, 10.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (i, kind) in zip(axes.flat, picks):
        x, _ = ds[i]
        cam = gradcam(model, x.unsqueeze(0), int(pred[i]))
        img = np.clip((x.numpy() * std + mean).transpose(1, 2, 0), 0, 1)
        cam_up = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(img.shape[:2][::-1], Image.BILINEAR)) / 255
        ax.imshow(img)
        ax.imshow(cam_up, cmap="jet", alpha=0.4)
        ax.set_title(f"{kind}: true {classes[y[i]]}\npred {classes[pred[i]]} ({conf[i]:.2f})", fontsize=9,
                     color="black" if kind == "correct" else "firebrick")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [{"index": i, "kind": k, "md5": ds.rows[i]["md5"]} for i, k in picks]


# ---------------------------------------------------------------- figures
def plot_learning_curves(hist: list[dict], path: Path, title: str) -> None:
    x = np.arange(1, len(hist) + 1)
    s1 = sum(h["stage"] == "1" for h in hist)
    f = {k: [float(h[k]) for h in hist] for k in ("train_loss", "val_loss", "train_top1", "val_top1", "val_macro_f1")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    a.plot(x, f["train_loss"], label="train loss (augmented, label-smoothed)")
    a.plot(x, f["val_loss"], label="val loss")
    b.plot(x, f["train_top1"], label="train top-1 (augmented)")
    b.plot(x, f["val_top1"], label="val top-1")
    b.plot(x, f["val_macro_f1"], label="val macro-F1")
    best = s1 + int(np.argmax(f["val_macro_f1"][s1:]))
    for ax in (a, b):
        ax.axvline(s1 + 0.5, color="grey", ls=":", lw=1)
        ax.axvline(best + 1, color="green", ls="--", lw=1, label="best checkpoint")
        ax.set_xlabel("epoch (stage 1 | stage 2)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    a.set_title("loss")
    b.set_title("accuracy")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_robustness(rob: dict, path: Path, title: str) -> None:
    names = list(rob)
    fig, ax = plt.subplots(figsize=(11, 4))
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [rob[n]["top1"] for n in names], 0.4, label="top-1")
    ax.bar(xs + 0.2, [rob[n]["top3"] for n in names], 0.4, label="top-3")
    ax.set_xticks(xs, names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Integrity audit (overfitting, leakage, shortcuts, robustness)")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--compare", type=Path, help="second run for a paired McNemar test")
    ap.add_argument("--out", type=Path, help="default: reports/integrity/<run name>")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, help="debug: cap images per probe set")
    args = ap.parse_args(argv)

    run = args.run
    out = args.out or REPO_ROOT / "reports" / "integrity" / run.name
    out.mkdir(parents=True, exist_ok=True)
    global CACHE_DIR
    CACHE_DIR = out / "cache"
    CACHE_DIR.mkdir(exist_ok=True)
    with open(run / "config_resolved.json", encoding="utf-8") as f:
        cfg = json.load(f)
    local = copy.deepcopy(cfg)
    local["data"]["manifest"] = None          # Kaggle paths -> the local manifest (MD5-identical, checked on Kaggle)
    model, meta = load_trained(run / "best.pt", "cpu")
    splits, classes, roots, data_cfg = split_rows(local, "local")
    assert classes == meta["classes"]
    cache_dir = resolve_path(data_cfg["cache"]["dir"])
    temperature = json.load(open(run / "eval" / "calibration.json", encoding="utf-8"))["temperature"]
    bands = BandThresholds.load(run / "eval" / "bands.yaml")
    official = np.load(run / "eval" / "logits.npz")
    lim = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)
    eval_tf = build_transforms(cfg["img_size"], train=False, crop_pct=cfg.get("crop_pct", 0.875))
    cidx = {c: i for i, c in enumerate(classes)}
    report: dict = {"run": run.name, "model": meta["model_cfg"]["name"], "temperature": temperature}

    def manifest_ds(rows, corruption=clean):
        return ManifestDataset(rows, cidx, roots, T.Compose([corruption, eval_tf]), cache_dir=cache_dir)

    test_rows, train_rows = splits["test"], splits["train"]
    test_y = np.array([cidx[r["label"]] for r in test_rows])
    assert np.array_equal(test_y, official["test_y"]), "local test split order differs from the Kaggle run"
    test_probs = calibrate(official["test_logits"], temperature)
    correct = test_probs.argmax(1) == test_y

    # 0. reproduction: the downloaded checkpoint, run locally on CPU, must give the Kaggle logits
    print("reproduction check (test split, CPU)", flush=True)
    rob_logits = {"clean": _logits(model, manifest_ds(lim(test_rows)), args.workers, "test_clean")}
    ref = official["test_logits"][:len(rob_logits["clean"])]
    report["reproduction"] = {"max_abs_logit_diff": float(np.abs(rob_logits["clean"] - ref).max()),
                              "argmax_agreement": float((rob_logits["clean"].argmax(1) == ref.argmax(1)).mean())}

    # 1. overfitting
    print("clean train-split evaluation", flush=True)
    tr_rows = lim(train_rows)
    tr_logits = _logits(model, manifest_ds(tr_rows), args.workers, "train_clean")
    tr_y = np.array([cidx[r["label"]] for r in tr_rows])
    report["overfitting"] = check_overfitting(
        run, calibrate(tr_logits, temperature), tr_y, tr_logits,
        (official["val_logits"], official["val_y"]), (official["test_logits"], test_y), temperature, classes)
    plot_learning_curves(report["overfitting"]["history"], out / "learning_curves.png",
                         f"{meta['model_cfg']['name']}: learning curves")

    # 2. leakage
    print("near-duplicate / leakage analysis", flush=True)
    def path_of(r):
        p = cache_dir / f"{r['md5']}.jpg"
        return p if p.exists() else roots[r["dataset"]] / r["relpath"]
    paths_by_md5 = {r["md5"]: path_of(r) for r in train_rows + test_rows}
    report["leakage"] = check_leakage(test_rows, train_rows, correct, test_y,
                                      [cidx[r["label"]] for r in train_rows], paths_by_md5, args.workers)

    # 3. shortcuts: sources, Food-5K food photos, Grad-CAM
    report["sources"] = check_sources(test_rows, correct)
    food5k = lim([p for d in sorted((roots["food5k"]).glob("*/food")) for p in _images(d)])
    print(f"Food-5K food photos (never trained on): {len(food5k)}", flush=True)
    f5 = calibrate(_logits(model, PathDataset(food5k, eval_tf), args.workers, "food5k_food"), temperature)
    nf = classes.index(NOT_FOOD)
    report["food5k_food_probe"] = {"n": len(food5k), "predicted_not_food": float((f5.argmax(1) == nf).mean()),
                                   "mean_p_not_food": float(f5[:, nf].mean()), "bands": band_counts(f5, classes, bands)}
    print("Grad-CAM", flush=True)
    report["gradcam"] = gradcam_figure(model, manifest_ds(test_rows), test_probs, test_y, classes,
                                       out / "gradcam.png", f"{meta['model_cfg']['name']}: Grad-CAM on test images")

    # 4. robustness
    for name, fn in CORRUPTIONS.items():
        if name == "clean":
            continue
        print(f"corruption: {name}", flush=True)
        rob_logits[name] = _logits(model, manifest_ds(lim(test_rows), fn), args.workers, f"test_{name}")
    y_lim = test_y[:len(rob_logits["clean"])]
    report["robustness"] = {}
    for name, lg in rob_logits.items():
        p = calibrate(lg, temperature)
        report["robustness"][name] = {"top1": float((p.argmax(1) == y_lim).mean()),
                                      "top3": float(topk_correct(p, y_lim, 3).mean()),
                                      "bands": band_counts(p, classes, bands),
                                      "high_band_accuracy": float((p.argmax(1) == y_lim)[p.max(1) >= bands.high].mean())
                                      if (p.max(1) >= bands.high).any() else None}
    plot_robustness(report["robustness"], out / "robustness.png",
                    f"{meta['model_cfg']['name']}: test accuracy under phone-like corruptions")

    # 5. open set: dishes from the same Indian dataset that are not among the 31 classes
    parent = roots["sourav80"] / "Indian Food Images" / "Indian Food Images"
    used = {Path(list(s.values())[0]).name for c in load_classes(resolve_path(cfg["data"]["classes"]))
            for s in c["sources"] if "sourav80" in s}
    ood_dirs = sorted(d for d in parent.iterdir() if d.is_dir() and d.name not in used)
    ood = [(d.name, p) for d in ood_dirs for p in _images(d)]
    ood = lim(ood)
    print(f"open-set probe: {len(ood)} images from {len(ood_dirs)} unseen dishes", flush=True)
    po = calibrate(_logits(model, PathDataset([p for _, p in ood], eval_tf), args.workers, "open_set"), temperature)
    per_dish = defaultdict(list)
    for (dish, _), p in zip(ood, po):
        per_dish[dish].append(p)
    report["open_set"] = {
        "n": len(ood), "dishes": len(ood_dirs), "bands": band_counts(po, classes, bands),
        "in_distribution_test_bands": band_counts(test_probs, classes, bands),
        "per_dish": sorted(({"dish": d, "n": len(ps), "high": band_counts(np.stack(ps), classes, bands)["high"],
                             "top_prediction": Counter(classes[int(p.argmax())] for p in ps).most_common(1)[0][0]}
                            for d, ps in per_dish.items()), key=lambda r: -r["high"]),
    }

    # 6. paired comparison
    if args.compare:
        other = np.load(args.compare / "eval" / "logits.npz")
        assert np.array_equal(other["test_y"], test_y)
        t2 = json.load(open(args.compare / "eval" / "calibration.json", encoding="utf-8"))["temperature"]
        correct_b = calibrate(other["test_logits"], t2).argmax(1) == test_y
        report["comparison"] = {"other": args.compare.name, "top1_this": float(correct.mean()),
                                "top1_other": float(correct_b.mean()),
                                "diff_ci95": paired_bootstrap_diff(correct, correct_b),
                                "mcnemar": mcnemar(correct, correct_b),
                                "top1_ci95_this": bootstrap_ci(correct)}

    report["overfitting"].pop("history")
    (out / "integrity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("reproduction", "food5k_food_probe")}, indent=2))
    print(f"wrote {out / 'integrity.json'}")


if __name__ == "__main__":
    main()
