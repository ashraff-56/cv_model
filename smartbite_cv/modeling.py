"""Model construction, checkpoint I/O and batched inference shared by train/evaluate/export."""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader

from .config import load_classes, load_config, resolve_path
from .data import ManifestDataset, build_transforms
from .prepare import dataset_roots, read_manifest


def build_model(model_cfg: dict, num_classes: int, pretrained: bool | None = None) -> torch.nn.Module:
    pretrained = model_cfg.get("pretrained", True) if pretrained is None else pretrained
    extra = {}
    if pretrained and model_cfg.get("pretrained_file"):
        # offline machines (Kaggle without internet): load the same timm weights from a local file
        extra["pretrained_cfg_overlay"] = {"file": str(resolve_path(model_cfg["pretrained_file"]))}
    model = timm.create_model(
        model_cfg["name"],
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=model_cfg.get("drop_rate", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        **extra,
    )
    if pretrained:
        # timm's default head init (uniform +-1/sqrt(num_classes)) gives very large initial logits on
        # ImageNet features; a zero-initialised head starts at loss = ln(C) and trains smoothly.
        head = model.get_classifier()
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    return model


def checkpoint_meta(cfg: dict, classes: list[str]) -> dict:
    """Store the exact ordered class definitions used by this run, not just their slugs."""
    class_path = resolve_path(cfg["data"]["classes"])
    definitions = load_classes(class_path)
    by_slug = {item["slug"]: item for item in definitions}
    class_metadata = []
    for slug in classes:
        item = by_slug.get(slug, {"slug": slug})
        class_metadata.append({
            "slug": slug,
            "display": item.get("display", slug),
            "variants": item.get("variants", []),
            "category": item.get("category"),
        })
    return {
        "model_cfg": cfg["model"],
        "classes": classes,
        "class_metadata": class_metadata,
        "class_config": str(class_path),
        "class_config_sha256": hashlib.sha256(class_path.read_bytes()).hexdigest(),
        "img_size": cfg["img_size"],
        "crop_pct": cfg.get("crop_pct", 0.875),
        "mean": list(cfg.get("mean", (0.485, 0.456, 0.406))),
        "std": list(cfg.get("std", (0.229, 0.224, 0.225))),
        "interpolation": "bicubic",
    }


def load_trained(path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = build_model(meta["model_cfg"], len(meta["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), meta


@torch.no_grad()
def infer_logits(model: torch.nn.Module, loader: DataLoader, device: str, amp: bool = False):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.startswith("cuda")):
            logits = model(x)
        all_logits.append(logits.float().cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_y)


def split_rows(cfg: dict, profile: str) -> tuple[dict[str, list[dict]], list[str], dict, dict]:
    """Manifest rows per split (after optional class filter / per-class limits), class list, roots, data cfg."""
    data_cfg = load_config(resolve_path(cfg["data"]["config"]))
    all_classes = [c["slug"] for c in load_classes(resolve_path(cfg["data"]["classes"]))]
    rows = read_manifest(resolve_path(cfg["data"].get("manifest") or data_cfg["manifest"]))
    only = cfg["data"].get("only_classes")
    present = {r["label"] for r in rows}
    classes = [c for c in all_classes if c in present and (not only or c in only)]
    limits = cfg["data"].get("limit_per_class") or {}
    out: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        sel = [r for r in rows if r["split"] == split and r["label"] in classes]
        if limits.get(split):
            kept, seen = [], {}
            for r in sel:
                seen[r["label"]] = seen.get(r["label"], 0) + 1
                if seen[r["label"]] <= limits[split]:
                    kept.append(r)
            sel = kept
        out[split] = sel
    roots = dataset_roots(data_cfg, profile)
    return out, classes, roots, data_cfg


def make_loader(rows, classes, roots, data_cfg, cfg, train: bool, sampler=None, cache_dir=None) -> DataLoader:
    tf = build_transforms(cfg["img_size"], train=train, aug=cfg.get("aug"), crop_pct=cfg.get("crop_pct", 0.875))
    ds = ManifestDataset(rows, {c: i for i, c in enumerate(classes)}, roots, tf,
                         cache_dir=cache_dir if cache_dir is not None else resolve_path(data_cfg["cache"]["dir"]))
    workers = cfg["data"].get("num_workers", 4)
    return DataLoader(ds, batch_size=cfg["batch_size"] if train else cfg.get("eval_batch_size", cfg["batch_size"]),
                      shuffle=train and sampler is None, sampler=sampler, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      drop_last=train and len(ds) > cfg["batch_size"])
