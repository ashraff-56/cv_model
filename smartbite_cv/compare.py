"""Side-by-side table of evaluated models (reads each run's eval/metrics.json + latency json).

    python -m smartbite_cv.compare --run runs/efficientnet_b0 --run runs/mobilenetv2 \
        --latency reports/metrics/latency_efficientnet_b0.json --latency reports/metrics/latency_mobilenetv2.json \
        --out reports/metrics/comparison.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import plots


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare evaluated runs")
    ap.add_argument("--run", action="append", required=True, type=Path)
    ap.add_argument("--latency", action="append", default=[], type=Path)
    ap.add_argument("--artifact", action="append", default=[], type=Path, help="bundle dirs for ONNX size")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    latency = {}
    for p in args.latency:
        d = json.loads(p.read_text(encoding="utf-8"))
        latency[d["model"]] = d
    rows = []
    for run in args.run:
        m = json.loads((run / "eval" / "metrics.json").read_text(encoding="utf-8"))
        t = m["test"]
        lat = latency.get(m["model"], {})
        rows.append({
            "name": m["model"], "top1": t["top1"], "top1_ci": t["top1_ci95"], "top3": t["top3"],
            "macro_f1": t["macro_f1"], "ece_before": m["calibration"]["test"]["ece_before"],
            "ece_after": m["calibration"]["test"]["ece_after"], "T": m["temperature"],
            "high": m["bands_on_test"]["high"], "low": m["bands_on_test"]["low"],
            "lat": lat.get("inference", {}).get("median_ms"), "lat_total": lat.get("total", {}).get("median_ms"),
            "onnx_mb": lat.get("onnx_mb"), "n": t["n"],
        })

    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}%"

    def num(v, unit):
        return "n/a" if v is None else f"{v:.1f} {unit}"

    lines = ["| model | test top-1 (95% CI) | top-3 | macro-F1 | ECE before -> after T | T | "
             "high band: share / top-1 | low band share | CPU inference median | end-to-end median | ONNX size |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['name']} | {pct(r['top1'])} ({pct(r['top1_ci'][0])}-{pct(r['top1_ci'][1])}) | {pct(r['top3'])} | "
            f"{r['macro_f1']:.3f} | {r['ece_before']:.3f} -> {r['ece_after']:.3f} | {r['T']:.2f} | "
            f"{pct(r['high']['fraction'])} / {pct(r['high']['top1'])} | {pct(r['low']['fraction'])} | "
            f"{num(r['lat'], 'ms')} | {num(r['lat_total'], 'ms')} | {num(r['onnx_mb'], 'MB')} |")
    lines.append(f"\nTest images: {rows[0]['n']} (same split for every model).")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    plots.comparison_bars(rows, args.out.with_suffix(".png"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
