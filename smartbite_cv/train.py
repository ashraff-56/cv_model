"""Two-stage transfer learning: (1) train the new classifier head on a frozen ImageNet backbone,
(2) unfreeze everything and fine-tune with a lower learning rate, cosine schedule and warmup.

    python -m smartbite_cv.train --config configs/train/base.yaml --config configs/train/efficientnet_b0.yaml
    python -m smartbite_cv.train ... --resume          # continue from runs/<name>/last.pt

Best checkpoint = highest validation macro-F1. The test split is never touched here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
from scipy.special import softmax
from timm.optim import create_optimizer_v2

from .config import load_config, resolve_path
from .data import sqrt_balanced_sampler
from .metrics import summarize
from .modeling import build_model, checkpoint_meta, infer_logits, make_loader, split_rows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for p in model.parameters():
        p.requires_grad = trainable
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def freeze_batchnorm(model: torch.nn.Module) -> None:
    """Keep pretrained BatchNorm statistics while only the head is trained."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def warmup_cosine(total_steps: int, warmup_steps: int, min_ratio: float):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return fn


def make_stage(model, stage_cfg: dict, steps_per_epoch: int):
    opt = create_optimizer_v2(model, opt="adamw", lr=stage_cfg["lr"], weight_decay=stage_cfg["weight_decay"])
    total = stage_cfg["epochs"] * steps_per_epoch
    warm = int(stage_cfg.get("warmup_epochs", 0) * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(total, warm, stage_cfg.get("min_lr", 0.0) / stage_cfg["lr"]))
    return opt, sched


def train_one_epoch(model, loader, opt, sched, scaler, device, cfg, stage: int) -> dict:
    model.train()
    if stage == 1:
        freeze_batchnorm(model)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    use_amp = cfg.get("amp", True) and device.startswith("cuda")
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.get("grad_clip"):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
    return {"train_loss": total_loss / seen, "train_top1": correct / seen}


def validate(model, loader, device, cfg, classes) -> dict:
    logits, y = infer_logits(model, loader, device, amp=cfg.get("amp", True))
    probs = softmax(logits, axis=1)
    s = summarize(probs, y, classes)
    val_loss = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())
    return {"val_loss": val_loss, "val_top1": s["top1"], "val_top3": s["top3"], "val_macro_f1": s["macro_f1"]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Two-stage fine-tuning")
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--profile", choices=["local", "kaggle"], default="local")
    ap.add_argument("--out", help="run directory (default runs/<run_name>)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(*args.config, overrides=args.set)
    out = Path(args.out) if args.out else resolve_path(Path("runs") / cfg["run_name"])
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits, classes, roots, data_cfg = split_rows(cfg, args.profile)
    cache_dir = cfg["data"].get("cache_dir")
    cache_dir = resolve_path(cache_dir) if cache_dir else None
    train_rows = splits["train"]
    targets = [classes.index(r["label"]) for r in train_rows]
    sampler = sqrt_balanced_sampler(targets, cfg["seed"]) if cfg.get("balance") == "sqrt" else None
    train_loader = make_loader(train_rows, classes, roots, data_cfg, cfg, train=True, sampler=sampler,
                               cache_dir=cache_dir)
    val_loader = make_loader(splits["val"], classes, roots, data_cfg, cfg, train=False, cache_dir=cache_dir)
    print(f"device={device} classes={len(classes)} train={len(train_rows)} val={len(splits['val'])}")

    model = build_model(cfg["model"], len(classes)).to(device)
    meta = checkpoint_meta(cfg, classes)
    with open(out / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(out / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({"device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
                   "torch": torch.__version__, "timm": timm.__version__, "python": platform.python_version(),
                   "n_train": len(train_rows), "n_val": len(splits["val"]), "classes": classes}, f, indent=2)

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("amp", True) and device == "cuda")
    state = {"stage": 1, "epoch": 0, "best": -1.0, "bad_epochs": 0}
    resume = torch.load(out / "last.pt", map_location=device, weights_only=False) \
        if args.resume and (out / "last.pt").exists() else None
    if resume:
        model.load_state_dict(resume["model"])
        state = resume["state"]
        print(f"resuming at stage {state['stage']} epoch {state['epoch']}")

    history_path = out / "history.csv"
    fields = ["stage", "epoch", "lr", "train_loss", "train_top1", "val_loss", "val_top1", "val_top3",
              "val_macro_f1", "seconds"]
    if not (resume and history_path.exists()):
        with open(history_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    for stage in (1, 2):
        if stage < state["stage"]:
            continue
        stage_cfg = cfg[f"stage{stage}"]
        set_backbone_trainable(model, trainable=(stage == 2))
        opt, sched = make_stage(model, stage_cfg, len(train_loader))
        if resume and resume["state"]["stage"] == stage:
            opt.load_state_dict(resume["opt"])
            sched.load_state_dict(resume["sched"])
            scaler.load_state_dict(resume["scaler"])
        start_epoch = state["epoch"] if state["stage"] == stage else 0
        for epoch in range(start_epoch, stage_cfg["epochs"]):
            t0 = time.time()
            row = {"stage": stage, "epoch": epoch + 1, "lr": opt.param_groups[0]["lr"]}
            row.update(train_one_epoch(model, train_loader, opt, sched, scaler, device, cfg, stage))
            row.update(validate(model, val_loader, device, cfg, classes))
            row["seconds"] = round(time.time() - t0, 1)
            with open(history_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

            improved = row["val_macro_f1"] > state["best"]
            if improved:
                state["best"], state["bad_epochs"] = row["val_macro_f1"], 0
                torch.save({"model": model.state_dict(), "meta": meta, "stage": stage, "epoch": epoch + 1,
                            "val": row}, out / "best.pt")
            elif stage == 2:
                state["bad_epochs"] += 1
            state.update(stage=stage, epoch=epoch + 1)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "state": state, "meta": meta}, out / "last.pt")
            if stage == 2 and state["bad_epochs"] >= cfg.get("early_stop_patience", 1_000):
                print(f"early stop: no val macro-F1 improvement for {state['bad_epochs']} epochs")
                break
        state.update(stage=stage + 1, epoch=0)
        resume = None
    print(f"best val macro-F1 {state['best']:.4f}; checkpoint {out / 'best.pt'}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
