"""Export a trained run to a self-contained ONNX bundle and check it against PyTorch.

Bundle layout (what predict.py and the Phase 2 API load):
    <out>/model.onnx        logits for a [N,3,H,W] float32 normalised batch
    <out>/model_meta.json   classes, preprocessing, temperature, provenance
    <out>/bands.yaml        confidence-band thresholds (editable config)

    python -m smartbite_cv.export --run runs/efficientnet_b0 --out artifacts/efficientnet_b0
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .config import load_classes
from .data import eval_resize
from .modeling import load_trained


def export_onnx(model: torch.nn.Module, img_size: int, path: Path, opset: int = 17) -> None:
    dummy = torch.randn(1, 3, img_size, img_size)
    kwargs = dict(input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset)
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` argument
        torch.onnx.export(model, (dummy,), str(path), **kwargs)


def parity_check(model: torch.nn.Module, onnx_path: Path, img_size: int, n: int = 4) -> float:
    import onnxruntime as ort

    x = torch.randn(n, 3, img_size, img_size, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"input": x.numpy()})[0]
    return float(np.abs(ref - got).max())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Export best.pt to an ONNX bundle")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--eval-dir", type=Path, help="where calibration.json / bands.yaml live (default <run>/eval)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args(argv)

    eval_dir = args.eval_dir or args.run / "eval"
    args.out.mkdir(parents=True, exist_ok=True)
    model, meta = load_trained(args.run / "best.pt", "cpu")
    onnx_path = args.out / "model.onnx"
    export_onnx(model, meta["img_size"], onnx_path, args.opset)
    max_diff = parity_check(model, onnx_path, meta["img_size"])
    print(f"ONNX vs PyTorch max |logit diff| = {max_diff:.2e}")
    if max_diff > 1e-3:
        raise SystemExit("ONNX output does not match PyTorch; refusing to ship this export")

    with open(eval_dir / "calibration.json", encoding="utf-8") as f:
        temperature = json.load(f)["temperature"]
    shutil.copy(eval_dir / "bands.yaml", args.out / "bands.yaml")
    # New checkpoints carry the exact class definitions used during training. Keep the
    # fallback for older checkpoints that predate this metadata.
    display = {c["slug"]: c for c in meta.get("class_metadata", [])}
    if not display:
        display = {c["slug"]: c for c in load_classes()}
    ckpt = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    bundle = {
        "modelName": meta["model_cfg"]["name"],
        "classes": [{"slug": s, "displayName": display.get(s, {}).get("display", s),
                     "variants": display.get(s, {}).get("variants", []),
                     **({"category": display[s]["category"]} if display.get(s, {}).get("category") else {})}
                    for s in meta["classes"]],
        "classConfig": meta.get("class_config"),
        "classConfigSha256": meta.get("class_config_sha256"),
        "notFoodLabel": "not_food",
        "imgSize": meta["img_size"],
        "resizeShortSide": eval_resize(meta["img_size"], meta["crop_pct"]),
        "interpolation": meta["interpolation"],
        "mean": meta["mean"],
        "std": meta["std"],
        "temperature": temperature,
        "onnxOpset": args.opset,
        "onnxParityMaxAbsDiff": max_diff,
        "checkpoint": {"stage": ckpt.get("stage"), "epoch": ckpt.get("epoch"), "val": ckpt.get("val")},
        "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(args.out / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print(f"bundle written to {args.out} ({onnx_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
