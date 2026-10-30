"""CPU latency of the ONNX bundle, single image (batch 1): decode+preprocess and model separately.

    python -m smartbite_cv.latency --model-dir artifacts/efficientnet_b0 --image some.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .imageio import load_rgb
from .predict import FoodRecognizer, preprocess


def _stats(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"median_ms": float(np.median(a)), "p90_ms": float(np.percentile(a, 90)),
            "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean()), "runs": len(a)}


def benchmark(model_dir: Path, image: Path, runs: int = 100, warmup: int = 10, threads: int | None = None) -> dict:
    import onnxruntime as ort

    rec = FoodRecognizer(model_dir, threads=threads)
    m = rec.meta
    raw = Path(image).read_bytes()
    pre, model, total = [], [], []
    for i in range(warmup + runs):
        t0 = time.perf_counter()
        x = preprocess(load_rgb(raw), m["imgSize"], m["resizeShortSide"], m["mean"], m["std"], m["interpolation"])
        t1 = time.perf_counter()
        rec.session.run(["logits"], {"input": x})
        t2 = time.perf_counter()
        if i >= warmup:
            pre.append((t1 - t0) * 1e3)
            model.append((t2 - t1) * 1e3)
            total.append((t2 - t0) * 1e3)
    return {
        "model": m["modelName"], "onnx_mb": round((model_dir / "model.onnx").stat().st_size / 1e6, 2),
        "image": str(image), "image_size": list(load_rgb(raw).size),
        "decode_preprocess": _stats(pre), "inference": _stats(model), "total": _stats(total),
        "machine": {"processor": platform.processor(), "cpu_count": os.cpu_count(), "system": platform.system(),
                    "python": platform.python_version(), "onnxruntime": ort.__version__,
                    "intra_op_threads": threads or "default"},
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    res = benchmark(args.model_dir, args.image, args.runs, threads=args.threads)
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
