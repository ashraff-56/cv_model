"""Portion (food mass) estimation from one overhead photo, trained on Nutrition5k.

SmartBite's users don't type quantities, so grams come from the photo. Following Nutrition5k's
decomposition [Thames et al., CVPR 2021], nutrients = (nutrients per gram of the recognised dish) x (mass):
the dish classifier + nutrition table give the first factor, this model gives the second.

Model: ImageNet EfficientNet-B0 regressing log(mass) with a predicted log-variance (Gaussian NLL), so every
estimate carries its own uncertainty. That uncertainty is then used to combine the photo estimate with a
typical serving of the recognised dish (`fuse_with_serving`): a confident photo estimate dominates, an
unsure one falls back toward the usual portion.

Data: Nutrition5k overhead RealSense RGB (depth split: dishes with overhead captures), official train/test
dish lists; 10% of train (seeded by dish id) is held out for early stopping and uncertainty calibration.
Augmentation never rescales the image: apparent size is the signal, so random-resized crops would destroy it.

    python -m smartbite_cv.portion train --config configs/train/portion.yaml --profile kaggle --out runs/portion
    python -m smartbite_cv.portion evaluate --run runs/portion --profile kaggle
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import load_config, resolve_path
from .data import IMAGENET_MEAN, IMAGENET_STD
from .imageio import load_rgb
from .prepare import find_kaggle_root

LOGVAR_MIN, LOGVAR_MAX = -7.0, 3.0


# ---------------------------------------------------------------- data
def read_dish_metadata(paths: list[Path]) -> dict[str, dict]:
    """dish_id -> {mass, kcal, fat, carb, protein} from Nutrition5k's dish_metadata_cafe*.csv (no header)."""
    out = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 6:
                    continue
                kcal, mass, fat, carb, protein = (float(x) for x in row[1:6])
                out[row[0]] = {"mass": mass, "kcal": kcal, "fat": fat, "carb": carb, "protein": protein}
    return out


def split_ids(cfg: dict) -> dict[str, list[str]]:
    """Official train/test dish lists (depth split = dishes with an overhead capture); val carved from train."""
    meta_dir = resolve_path(cfg["data"]["metadata_dir"])
    read = lambda name: [l.strip() for l in open(meta_dir / "dish_ids" / "splits" / name, encoding="utf-8") if l.strip()]  # noqa: E731
    train, test = read(cfg["data"]["train_ids"]), read(cfg["data"]["test_ids"])
    seed, frac = cfg["seed"], cfg["data"]["val_fraction"]
    key = lambda d: hashlib.sha1(f"{seed}:{d}".encode()).hexdigest()  # noqa: E731
    ordered = sorted(train, key=key)
    n_val = round(len(ordered) * frac)
    return {"train": sorted(ordered[n_val:]), "val": sorted(ordered[:n_val]), "test": sorted(test)}


def image_root(cfg: dict, profile: str) -> Path:
    if profile == "kaggle":
        root = find_kaggle_root(cfg["data"]["marker"])
        if root is None:
            raise FileNotFoundError(f"Nutrition5k images not found under /kaggle/input (marker {cfg['data']['marker']})")
        return root
    return resolve_path(cfg["data"]["local_root"])


def usable(ids: list[str], meta: dict, root: Path) -> list[str]:
    """Dishes with metadata, a positive mass and an overhead RGB image."""
    return [d for d in ids if d in meta and meta[d]["mass"] > 0
            and (root / "imagery" / "realsense_overhead" / d / "rgb.png").exists()]


class PortionDataset(Dataset):
    def __init__(self, ids: list[str], meta: dict, root: Path, transform):
        self.ids, self.meta, self.root, self.transform = ids, meta, root, transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        d = self.ids[i]
        img = load_rgb(self.root / "imagery" / "realsense_overhead" / d / "rgb.png")
        return self.transform(img), torch.tensor(math.log(self.meta[d]["mass"]), dtype=torch.float32)


def build_transforms(img_hw: tuple[int, int], train: bool, aug: dict | None = None) -> T.Compose:
    """Resize to a fixed size (no crop, no rescale augmentation: size on screen is the label signal)."""
    ops = [T.Resize(tuple(img_hw), interpolation=T.InterpolationMode.BICUBIC)]
    if train:
        aug = aug or {}
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                T.ColorJitter(*aug.get("color_jitter", [0.15, 0.15, 0.1, 0.02]))]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


# ---------------------------------------------------------------- model
class PortionNet(torch.nn.Module):
    """Backbone -> (mean, log-variance) of log(mass in grams)."""

    def __init__(self, model_cfg: dict, pretrained: bool = True):
        super().__init__()
        extra = {}
        if pretrained and model_cfg.get("pretrained_file"):
            extra["pretrained_cfg_overlay"] = {"file": str(resolve_path(model_cfg["pretrained_file"]))}
        self.backbone = timm.create_model(model_cfg["name"], pretrained=pretrained, num_classes=0,
                                          drop_rate=model_cfg.get("drop_rate", 0.0),
                                          drop_path_rate=model_cfg.get("drop_path_rate", 0.0), **extra)
        self.head = torch.nn.Linear(self.backbone.num_features, 2)
        torch.nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([model_cfg.get("init_log_mass", 5.3), 0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.backbone(x))
        return torch.stack([out[:, 0], out[:, 1].clamp(LOGVAR_MIN, LOGVAR_MAX)], dim=1)


def gaussian_nll(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mu, logvar = out[:, 0], out[:, 1]
    return 0.5 * (logvar + (y - mu) ** 2 / logvar.exp()).mean()


@torch.no_grad()
def predict(model: PortionNet, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    mus, logvars, ys = [], [], []
    for x, y in loader:
        out = model(x.to(device)).float().cpu()
        mus.append(out[:, 0].numpy())
        logvars.append(out[:, 1].numpy())
        ys.append(y.numpy())
    return np.concatenate(mus), np.concatenate(logvars), np.concatenate(ys)


# ---------------------------------------------------------------- metrics & fusion (unit-tested)
def mass_metrics(pred_g: np.ndarray, true_g: np.ndarray) -> dict:
    """Nutrition5k-style MAE (grams and % of mean truth) plus per-dish relative errors."""
    err = np.abs(pred_g - true_g)
    rel = err / true_g
    return {"n": int(len(true_g)), "mae_g": float(err.mean()), "mae_pct_of_mean": float(err.mean() / true_g.mean() * 100),
            "median_ape_pct": float(np.median(rel) * 100), "within_20pct": float((rel <= 0.2).mean()),
            "within_30pct": float((rel <= 0.3).mean()), "mean_signed_pct": float(((pred_g - true_g) / true_g).mean() * 100)}


def fit_sigma_scale(mu: np.ndarray, logvar: np.ndarray, y: np.ndarray) -> float:
    """Scale s for predicted std (sigma * s) minimising Gaussian NLL on held-out data (closed form)."""
    var = np.exp(logvar)
    return float(np.sqrt(np.mean((y - mu) ** 2 / var)))


def interval_coverage(mu, sigma, y, z: float) -> float:
    return float((np.abs(y - mu) <= z * sigma).mean())


def fuse_with_serving(mu_log: float | np.ndarray, sigma_log: float | np.ndarray, serving_g: float | np.ndarray,
                      prior_sigma_log: float) -> tuple[np.ndarray, np.ndarray]:
    """Precision-weighted combination in log-grams of the photo estimate and a typical serving.

    Both are treated as Gaussian in log space (multiplicative errors). Returns (mean, std) of log grams.
    """
    w_model, w_prior = 1.0 / np.square(sigma_log), 1.0 / prior_sigma_log ** 2
    mean = (w_model * np.asarray(mu_log) + w_prior * np.log(serving_g)) / (w_model + w_prior)
    return mean, np.sqrt(1.0 / (w_model + w_prior))


# ---------------------------------------------------------------- train / evaluate
def _loaders(cfg, profile, splits_used=("train", "val", "test"), workers=None):
    meta_dir = resolve_path(cfg["data"]["metadata_dir"])
    meta = read_dish_metadata(sorted((meta_dir / "metadata").glob("dish_metadata_cafe*.csv")))
    root = image_root(cfg, profile)
    ids = {k: usable(v, meta, root) for k, v in split_ids(cfg).items()}
    hw, bs = cfg["img_hw"], cfg["batch_size"]
    workers = cfg["num_workers"] if workers is None else workers
    loaders = {}
    for s in splits_used:
        ds = PortionDataset(ids[s], meta, root, build_transforms(hw, train=(s == "train"), aug=cfg.get("aug")))
        loaders[s] = DataLoader(ds, batch_size=bs, shuffle=(s == "train"), num_workers=workers,
                                drop_last=(s == "train"), pin_memory=torch.cuda.is_available())
    return loaders, ids, meta


def train(cfg: dict, profile: str, out: Path) -> None:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders, ids, meta = _loaders(cfg, profile, ("train", "val"))
    train_mass = np.array([meta[d]["mass"] for d in ids["train"]])
    model_cfg = {**cfg["model"], "init_log_mass": float(np.log(train_mass).mean())}
    model = PortionNet(model_cfg).to(device)
    print(f"device={device} train={len(ids['train'])} val={len(ids['val'])} "
          f"train mass mean={train_mass.mean():.1f} g", flush=True)
    fit(cfg, model, model_cfg, loaders, out, device)


def fit(cfg: dict, model: PortionNet, model_cfg: dict, loaders: dict, out: Path, device: str) -> float:
    """AdamW + warmup/cosine, Gaussian NLL, early stopping on validation MAE (grams); writes best.pt."""
    out.mkdir(parents=True, exist_ok=True)
    epochs, lr = cfg["epochs"], cfg["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    steps = epochs * len(loaders["train"])
    warm = cfg.get("warmup_epochs", 1) * len(loaders["train"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * (
        cfg["min_lr"] / lr + (1 - cfg["min_lr"] / lr) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best, bad, hist = float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        t0, model_loss, n = time.time(), 0.0, 0
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                out_ = model(x)
            loss = gaussian_nll(out_.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            scaler.step(opt)
            scaler.update()
            sched.step()
            model_loss += loss.item() * len(y)
            n += len(y)
        mu, logvar, yv = predict(model, loaders["val"], device)
        m = mass_metrics(np.exp(mu), np.exp(yv))
        row = {"epoch": epoch, "train_nll": model_loss / n, "val_mae_g": m["mae_g"], "val_mae_pct": m["mae_pct_of_mean"],
               "val_median_ape": m["median_ape_pct"], "seconds": round(time.time() - t0, 1)}
        hist.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if m["mae_g"] < best:
            best, bad = m["mae_g"], 0
            torch.save({"model": model.state_dict(), "model_cfg": model_cfg, "img_hw": cfg["img_hw"]}, out / "best.pt")
        else:
            bad += 1
            if bad >= cfg["early_stop_patience"]:
                print(f"early stop: no val MAE improvement for {bad} epochs", flush=True)
                break
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    (out / "config_resolved.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"best val MAE {best:.1f} g; checkpoint {out / 'best.pt'}", flush=True)
    return best


def load_portion(path: Path, device: str = "cpu") -> tuple[PortionNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PortionNet(ckpt["model_cfg"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def evaluate(run: Path, profile: str) -> dict:
    cfg = json.loads((run / "config_resolved.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_portion(run / "best.pt", device)
    loaders, ids, meta = _loaders(cfg, profile, ("train", "val", "test"))
    mu_v, lv_v, y_v = predict(model, loaders["val"], device)
    mu_t, lv_t, y_t = predict(model, loaders["test"], device)
    s = fit_sigma_scale(mu_v, lv_v, y_v)
    sig_t = np.exp(0.5 * lv_t) * s
    true_g = np.exp(y_t)
    train_log = np.log([meta[d]["mass"] for d in ids["train"]])
    prior_mu, prior_sigma = float(train_log.mean()), float(train_log.std())
    fused_mu, fused_sig = fuse_with_serving(mu_t, sig_t, np.exp(prior_mu), prior_sigma)
    # nutrient error caused by the portion estimate alone (true nutrients per gram x estimated grams)
    per_g = {k: np.array([meta[d][k] / meta[d]["mass"] for d in ids["test"]]) for k in ("kcal", "protein", "carb", "fat")}
    truth = {k: np.array([meta[d][k] for d in ids["test"]]) for k in per_g}

    def nutrient_err(grams):
        return {k: float(np.abs(per_g[k] * grams - truth[k]).mean() / truth[k].mean() * 100) for k in per_g}

    res = {
        "test_n": int(len(y_t)), "val_n": int(len(y_v)),
        "baseline_train_mean": mass_metrics(np.full_like(true_g, np.exp(train_log).mean()), true_g),
        "baseline_train_geometric_mean": mass_metrics(np.full_like(true_g, np.exp(prior_mu)), true_g),
        "model": mass_metrics(np.exp(mu_t), true_g),
        "model_fused_with_typical_serving": mass_metrics(np.exp(fused_mu), true_g),
        "nutrient_mae_pct_of_mean_from_portion_error": {"model": nutrient_err(np.exp(mu_t)),
                                                        "fused": nutrient_err(np.exp(fused_mu))},
        "uncertainty": {"sigma_scale_fitted_on_val": s, "coverage_68": interval_coverage(mu_t, sig_t, y_t, 1.0),
                        "coverage_95": interval_coverage(mu_t, sig_t, y_t, 1.96),
                        "median_sigma_log": float(np.median(sig_t))},
        "prior_used_for_fusion_eval": {"typical_g": float(np.exp(prior_mu)), "sigma_log": prior_sigma,
                                       "note": "Nutrition5k has no dish-specific typical servings; the train-set "
                                               "geometric mean stands in for the per-dish serving used in the app"},
        "paper_reference_overhead_image_only_mass_mae": "38.1 g / 29.5% (Nutrition5k Table 4)",
    }
    (run / "eval").mkdir(exist_ok=True)
    (run / "eval" / "portion_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (run / "eval" / "calibration.json").write_text(json.dumps({"sigma_scale": s}, indent=2), encoding="utf-8")
    np.savez_compressed(run / "eval" / "test_predictions.npz", dish=np.array(ids["test"]), mu=mu_t,
                        sigma=sig_t, y=y_t)
    print(json.dumps({k: res[k] for k in ("baseline_train_mean", "model", "model_fused_with_typical_serving",
                                          "uncertainty")}, indent=2))
    return res


# ---------------------------------------------------------------- deployment (ONNX, no PyTorch at inference)
def export_onnx(run: Path, out_dir: Path) -> Path:
    """model.onnx (input image -> [mean, log-variance] of log grams) + portion_meta.json with calibration."""
    model, ckpt = load_portion(run / "best.pt")
    metrics = json.loads((run / "eval" / "portion_metrics.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = ckpt["img_hw"]
    # Parity inputs look like photos (smooth colour fields in [0, 1], then normalised): white Gaussian noise
    # drives activations far outside anything the model sees and exaggerates float rounding differences.
    gen = torch.Generator().manual_seed(0)
    smooth = torch.nn.functional.interpolate(torch.rand(4, 3, h // 10, w // 10, generator=gen), size=(h, w),
                                             mode="bicubic", align_corners=False).clamp(0, 1)
    dummy = (smooth - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    torch.onnx.export(model, (dummy[:1],), str(out_dir / "model.onnx"), input_names=["input"], output_names=["out"],
                      dynamic_axes={"input": {0: "batch"}, "out": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    diff = float(np.abs(sess.run(["out"], {"input": dummy.numpy()})[0] - ref).max())
    if diff > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {diff}")
    meta = {"imgHW": [h, w], "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD), "interpolation": "bicubic",
            "sigmaScale": metrics["uncertainty"]["sigma_scale_fitted_on_val"],
            "priorSigmaLog": metrics["prior_used_for_fusion_eval"]["sigma_log"],
            "trainedOn": "Nutrition5k overhead RGB (Thames et al., CVPR 2021), official depth split",
            "testMassMAE": {k: metrics["model"][k] for k in ("mae_g", "mae_pct_of_mean", "median_ape_pct")},
            "onnxParityMaxAbsDiff": diff}
    (out_dir / "portion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir / "model.onnx"


class PortionEstimator:
    """ONNX portion model: photo -> (grams, uncertainty), optionally combined with a typical serving."""

    def __init__(self, model_dir: str | Path, threads: int | None = None):
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "portion_meta.json").read_text(encoding="utf-8"))
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(self.model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"])

    def preprocess(self, img) -> np.ndarray:
        from PIL import Image
        if img.height > img.width:   # trained on landscape 4:3 overhead frames; a top-down plate is rotation-free
            img = img.rotate(90, expand=True)
        h, w = self.meta["imgHW"]
        img = img.resize((w, h), Image.Resampling.BICUBIC)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - np.asarray(self.meta["mean"], dtype=np.float32)) / np.asarray(self.meta["std"], dtype=np.float32)
        return x.transpose(2, 0, 1)[None]

    def log_grams(self, img) -> tuple[float, float]:
        mu, logvar = self.session.run(["out"], {"input": self.preprocess(img)})[0][0]
        return float(mu), float(np.exp(0.5 * logvar) * self.meta["sigmaScale"])

    def estimate(self, img, typical_serving_g: float | None = None) -> dict:
        """Grams with a ~68% range. With a typical serving, the two are combined by confidence."""
        mu, sigma = self.log_grams(img)
        out = {"photoEstimateG": round(float(np.exp(mu)), 1), "photoRangeG": [round(float(np.exp(mu - sigma)), 1),
                                                                              round(float(np.exp(mu + sigma)), 1)]}
        if typical_serving_g:
            mu, sigma = fuse_with_serving(mu, sigma, typical_serving_g, self.meta["priorSigmaLog"])
            mu, sigma = float(mu), float(sigma)
            out["typicalServingG"] = typical_serving_g
        out.update({"grams": round(float(np.exp(mu)), 1),
                    "rangeG": [round(float(np.exp(mu - sigma)), 1), round(float(np.exp(mu + sigma)), 1)]})
        return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", action="append", required=True)
    t.add_argument("--set", action="append", default=[])
    t.add_argument("--profile", choices=["local", "kaggle"], default="local")
    t.add_argument("--out", type=Path, default=Path("runs/portion"))
    e = sub.add_parser("evaluate")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--profile", choices=["local", "kaggle"], default="local")
    x = sub.add_parser("export")
    x.add_argument("--run", type=Path, required=True)
    x.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "train":
        train(load_config(*args.config, overrides=args.set), args.profile, args.out)
    elif args.cmd == "evaluate":
        evaluate(args.run, args.profile)
    else:
        print(f"exported {export_onnx(args.run, args.out)}")


if __name__ == "__main__":
    main()
