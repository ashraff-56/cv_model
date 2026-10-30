"""Render the results section of the report straight from run outputs (no hand-copied numbers).

    python tools/make_results_section.py --results kaggle_out/results/outputs --out reports/results_section.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = {"efficientnet_b0": "EfficientNet-B0 (primary)", "mobilenetv2": "MobileNetV2 (baseline)"}


def pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True, help="folder with runs/<model>/ and artifacts/<model>/")
    ap.add_argument("--local-latency", type=Path, default=Path("reports/metrics"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    L = []
    rows = {}
    for key, name in MODELS.items():
        run = args.results / "runs" / key
        m = load(run / "eval" / "metrics.json")
        if m is None:
            continue
        info = load(run / "run_info.json") or {}
        hist = (run / "history.csv").read_text(encoding="utf-8").strip().splitlines()
        lat_local = load(args.local_latency / f"latency_{key}.json")
        lat_remote = load(run / "latency_colab_or_kaggle_cpu.json")
        meta = load(args.results / "artifacts" / key / "model_meta.json") or {}
        rows[key] = (name, m, info, hist, lat_local, lat_remote, meta)

    L += ["### Headline comparison (test split, never used for training, model selection or threshold choice)", "",
          "| | " + " | ".join(r[0] for r in rows.values()) + " |", "|---|" + "---|" * len(rows)]

    def line(label, fn):
        L.append(f"| {label} | " + " | ".join(fn(*r) for r in rows.values()) + " |")

    line("test images", lambda n, m, *_: str(m["test"]["n"]))
    line("top-1 accuracy (95% bootstrap CI)",
         lambda n, m, *_: f"{pct(m['test']['top1'])} ({pct(m['test']['top1_ci95'][0])} to {pct(m['test']['top1_ci95'][1])})")
    line("top-3 accuracy (95% CI)",
         lambda n, m, *_: f"{pct(m['test']['top3'])} ({pct(m['test']['top3_ci95'][0])} to {pct(m['test']['top3_ci95'][1])})")
    line("macro-F1", lambda n, m, *_: f"{m['test']['macro_f1']:.3f}")
    line("macro precision / recall", lambda n, m, *_: f"{m['test']['macro_precision']:.3f} / {m['test']['macro_recall']:.3f}")
    line("validation top-1 (model selection)", lambda n, m, *_: pct(m["val"]["top1"]))
    line("temperature T (fit on val)", lambda n, m, *_: f"{m['temperature']:.3f}")
    line("test ECE before -> after T", lambda n, m, *_: f"{m['calibration']['test']['ece_before']:.3f} -> {m['calibration']['test']['ece_after']:.3f}")
    line("test NLL before -> after T", lambda n, m, *_: f"{m['calibration']['test']['nll_before']:.3f} -> {m['calibration']['test']['nll_after']:.3f}")
    line("band thresholds high / low (val)", lambda n, m, *_: f"{m['bands']['high']:.3f} / {m['bands']['low']:.3f}")
    for b in ("high", "medium", "low"):
        line(f"test images in **{b}** band: share, top-1, top-3",
             lambda n, m, *_, b=b: f"{pct(m['bands_on_test'][b]['fraction'])}, {pct(m['bands_on_test'][b]['top1'])}, "
                                   f"{pct(m['bands_on_test'][b]['top3'])}")
    line("best epoch (stage)", lambda n, m, i, h, ll, lr, meta: f"{meta.get('checkpoint', {}).get('epoch')} "
                                                                 f"(stage {meta.get('checkpoint', {}).get('stage')})")
    line("epochs run", lambda n, m, i, h, *_: str(len(h) - 1))
    line("training GPU", lambda n, m, i, *_: str(i.get("device", "n/a")))
    line("ONNX size", lambda *r: f"{r[4]['onnx_mb']} MB" if r[4] else (f"{r[5]['onnx_mb']} MB" if r[5] else "n/a"))
    line("CPU latency, this PC: model / end-to-end median",
         lambda *r: f"{r[4]['inference']['median_ms']:.1f} / {r[4]['total']['median_ms']:.1f} ms" if r[4] else "n/a")
    line("CPU latency, Kaggle VM: model / end-to-end median",
         lambda *r: f"{r[5]['inference']['median_ms']:.1f} / {r[5]['total']['median_ms']:.1f} ms" if r[5] else "n/a")
    any_local = next((r[4] for r in rows.values() if r[4]), None)
    if any_local:
        mc = any_local["machine"]
        L += ["", f"Local latency machine: {mc['processor']}, {mc['cpu_count']} logical CPUs, {mc['system']}, "
                  f"onnxruntime {mc['onnxruntime']}, batch 1, {any_local['total']['runs']} timed runs after warm-up, "
                  f"test image {any_local['image_size'][0]}x{any_local['image_size'][1]} px."]

    for key, (name, m, *_rest) in rows.items():
        pc = m["test"]["per_class"]
        L += ["", f"### Per-class results: {name}", "", "| class | precision | recall | F1 | test images |",
              "|---|---:|---:|---:|---:|"]
        for c, v in sorted(pc.items(), key=lambda kv: kv[1]["f1"]):
            L.append(f"| {c} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['support']} |")
        L += ["", f"Worst confused pairs ({name}, both directions summed):", "",
              "| pair | a predicted as b | b predicted as a | total |", "|---|---:|---:|---:|"]
        for p in m["test"]["worst_confused_pairs"][:12]:
            L.append(f"| {p['a']} / {p['b']} | {p['a_as_b']} | {p['b_as_a']} | {p['total']} |")
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
