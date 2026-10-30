"""Package the code + manifest for Kaggle and generate the training notebook.

    python tools/build_kaggle.py --user <kaggle-username>
      -> kaggle_out/bundle/code/   private dataset "<user>/smartbite-cv-code" (one zip: code + manifest)
      -> kaggle_out/bundle/kernel/ private GPU notebook "<user>/smartbite-food-train"
      -> notebooks/smartbite_food_train.ipynb (same notebook, kept in the repo; also runs on Colab)

Then:
    kaggle datasets create -p kaggle_out/bundle/code        (first time; later: kaggle datasets version -p ... -m msg)
    kaggle kernels push -p kaggle_out/bundle/kernel
    kaggle kernels status <user>/smartbite-food-train
    kaggle kernels output <user>/smartbite-food-train -p kaggle_out/results
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ["smartbite_cv", "configs", "nutrition", "tools", "tests", "data/splits/manifest.csv", "requirements.txt"]
CODE_SLUG = "smartbite-cv-code"
KERNEL_SLUG = "smartbite-food-train"
MODELS = ["efficientnet_b0", "mobilenetv2"]
# Kaggle notebooks may run without internet (accounts without phone verification), so the ImageNet
# weights and a timm wheel travel inside the private code dataset: kaggle_out/offline/ -> offline/.
OFFLINE = ROOT / "kaggle_out" / "offline"
WEIGHTS = {"efficientnet_b0": "offline/efficientnet_b0.ra_in1k.safetensors",
           "mobilenetv2": "offline/mobilenetv2_100.ra_in1k.safetensors"}

SETUP = r'''# SmartBite food recognition - training run (Kaggle GPU or Colab)
import os, sys, glob, json, shutil, subprocess, zipfile, time
ON_KAGGLE = os.path.exists("/kaggle/input")
WORK = "/kaggle/working/SmartBite" if ON_KAGGLE else "/content/SmartBite"
OUT = "/kaggle/working/outputs" if ON_KAGGLE else "/content/outputs"
CACHE = "/tmp/smartbite_cache"

def run(cmd):
    """Run a shell command, stream its output, stop the notebook if it fails."""
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=WORK)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")

if ON_KAGGLE and not os.path.exists(f"{WORK}/smartbite_cv"):
    # The private code dataset is either auto-extracted or still a zip; handle both.
    inits = glob.glob("/kaggle/input/**/smartbite_cv/__init__.py", recursive=True)
    if inits:
        shutil.copytree(os.path.dirname(os.path.dirname(inits[0])), WORK, dirs_exist_ok=True)
    else:
        zips = glob.glob("/kaggle/input/**/smartbite_code.zip", recursive=True)
        assert zips, "attach the smartbite-cv-code dataset"
        zipfile.ZipFile(zips[0]).extractall(WORK)
# Colab: upload/clone the repo to /content/SmartBite and put the raw datasets under
# /content/SmartBite/data/raw/<same folder names as configs/data.yaml>, then set PROFILE = "local".
PROFILE = "kaggle" if ON_KAGGLE else "local"
os.makedirs(OUT, exist_ok=True)
# No internet needed: timm comes from the bundled wheel, ImageNet weights from offline/*.safetensors.
whl = glob.glob(f"{WORK}/offline/timm-*.whl")
if ON_KAGGLE and not whl:
    raise RuntimeError("offline/ missing: the attached smartbite-cv-code dataset version is stale; re-run")
run(f"pip install -q --no-index --no-deps {whl[0]}" if whl else "pip install -q timm onnx onnxruntime")
run("nvidia-smi || true")
run("python -c \"import torch, timm; print('torch', torch.__version__, 'timm', timm.__version__)\"")
import torch
assert torch.cuda.is_available(), "No GPU attached: enable the GPU accelerator (Kaggle needs a phone-verified account)"
try:
    import onnxruntime  # noqa: F401
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False  # ONNX export + latency are then done locally from best.pt
print("onnxruntime available:", HAVE_ORT)
'''

VERIFY = r'''# 1. Check every manifest file exists with the same bytes (MD5) and build resized copies.
t0 = time.time()
try:
    run(f"python -m smartbite_cv.prepare --profile {PROFILE} --verify-only --write-verified data/splits/manifest_verified.csv "
        f"--cache --cache-dir {CACHE} --workers 4")
except RuntimeError:
    print("Some manifest rows are missing or differ; training continues on the verified rows only.")
print(f"data step: {time.time() - t0:.0f}s")
'''

TRAIN = r'''# 2. Train, evaluate (test split used once), export to ONNX, CPU latency.
MODELS = {models}
WEIGHTS = {weights}
COMMON = f"--profile {{PROFILE}} --set data.cache_dir={{CACHE}} --set data.manifest=data/splits/manifest_verified.csv"
sample = sorted(glob.glob(f"{{CACHE}}/*.jpg"))[0]
for m in MODELS:
    t0 = time.time()
    weights = f"--set model.pretrained_file={{WEIGHTS[m]}}" if os.path.exists(f"{{WORK}}/{{WEIGHTS[m]}}") else ""
    run(f"python -m smartbite_cv.train --config configs/train/base.yaml --config configs/train/{{m}}.yaml {{COMMON}} "
        f"{{weights}} --out runs/{{m}}")
    run(f"python -m smartbite_cv.evaluate --run runs/{{m}} --profile {{PROFILE}}")
    if HAVE_ORT:
        run(f"python -m smartbite_cv.export --run runs/{{m}} --out artifacts/{{m}}")
        run(f"python -m smartbite_cv.latency --model-dir artifacts/{{m}} --image {{sample}} --out runs/{{m}}/latency_colab_or_kaggle_cpu.json")
    print(f"{{m}}: {{(time.time() - t0) / 60:.1f}} min")
lat = [f"runs/{{m}}/latency_colab_or_kaggle_cpu.json" for m in MODELS if os.path.exists(f"{{WORK}}/runs/{{m}}/latency_colab_or_kaggle_cpu.json")]
run("python -m smartbite_cv.compare " + " ".join(f"--run runs/{{m}}" for m in MODELS)
    + "".join(f" --latency {{p}}" for p in lat) + " --out runs/comparison.md")
'''

PACKAGE = r'''# 3. Collect outputs (small files + best checkpoints + ONNX bundles).
for m in MODELS:
    dst = f"{OUT}/runs/{m}"
    os.makedirs(dst, exist_ok=True)
    for f in ["history.csv", "config_resolved.json", "run_info.json", "best.pt", "latency_colab_or_kaggle_cpu.json"]:
        if os.path.exists(f"{WORK}/runs/{m}/{f}"):
            shutil.copy(f"{WORK}/runs/{m}/{f}", dst)
    shutil.copytree(f"{WORK}/runs/{m}/eval", f"{dst}/eval", dirs_exist_ok=True)
    if os.path.exists(f"{WORK}/artifacts/{m}"):
        shutil.copytree(f"{WORK}/artifacts/{m}", f"{OUT}/artifacts/{m}", dirs_exist_ok=True)
for f in ["runs/comparison.md", "runs/comparison.png", "data/splits/manifest_verified.csv"]:
    if os.path.exists(f"{WORK}/{f}"):
        shutil.copy(f"{WORK}/{f}", OUT)
if ON_KAGGLE:
    shutil.rmtree(WORK, ignore_errors=True)   # keep the notebook output small
print(open(f"{OUT}/comparison.md").read())
'''


def build_notebook() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# SmartBite food recognition: training run\n"
            "Trains EfficientNet-B0 (primary) and MobileNetV2 (baseline) on the leakage-safe split in "
            "`data/splits/manifest.csv`, then evaluates on the untouched test split, calibrates (temperature "
            "scaling), chooses confidence bands on the validation split, exports ONNX and measures CPU latency.\n\n"
            "**Kaggle:** attach the source datasets listed in `configs/data.yaml` and the private code dataset; "
            "GPU on; internet is not required (weights and timm are bundled). **Colab:** see the comment in the first cell."),
        nbformat.v4.new_code_cell(SETUP),
        nbformat.v4.new_code_cell(VERIFY),
        nbformat.v4.new_code_cell(TRAIN.format(models=json.dumps(MODELS), weights=json.dumps(WEIGHTS))),
        nbformat.v4.new_code_cell(PACKAGE),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", required=True, help="Kaggle username that will own the private dataset/notebook")
    args = ap.parse_args()

    bundle = ROOT / "kaggle_out" / "bundle"
    code_dir, kernel_dir = bundle / "code", bundle / "kernel"
    code_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(code_dir / "smartbite_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
            for f in files:
                z.write(f, f.relative_to(ROOT).as_posix())
        for f in sorted(OFFLINE.glob("*")):
            z.write(f, f"offline/{f.name}")
    (code_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": CODE_SLUG, "id": f"{args.user}/{CODE_SLUG}", "licenses": [{"name": "other"}]}, indent=2))

    nb = build_notebook()
    nbformat.write(nb, ROOT / "notebooks" / "smartbite_food_train.ipynb")
    nbformat.write(nb, kernel_dir / f"{KERNEL_SLUG}.ipynb")
    data_cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    sources = [d["kaggle_ref"] for d in data_cfg["datasets"].values()] + [f"{args.user}/{CODE_SLUG}"]
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": f"{KERNEL_SLUG}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "dataset_sources": sources, "competition_sources": [], "kernel_sources": []},
        indent=2))
    size = (code_dir / "smartbite_code.zip").stat().st_size / 1e6
    print(f"code zip {size:.1f} MB; notebook + metadata in {kernel_dir}; dataset sources: {sources}")


if __name__ == "__main__":
    main()
