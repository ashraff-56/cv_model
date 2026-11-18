"""Package the code + manifest for Kaggle and generate the training notebook.

    python tools/build_kaggle.py --user <kaggle-username> [--version v2]
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
INCLUDE = ["smartbite_cv", "configs", "nutrition", "tools", "tests", "requirements.txt"]
# per data version: data config, classes, manifest, extra train config, code dataset and notebook slugs
VERSIONS = {
    "v1": {"data": "configs/data.yaml", "classes": "configs/classes.yaml", "manifest": "data/splits/manifest.csv",
           "train_extra": "", "code_slug": "smartbite-cv-code", "kernel_slug": "smartbite-food-train"},
    "v2": {"data": "configs/data_v2.yaml", "classes": "configs/classes_v2.yaml", "manifest": "data/splits/manifest_v2.csv",
           "train_extra": "--config configs/train/v2.yaml", "code_slug": "smartbite-cv-code-v2",
           "kernel_slug": "smartbite-food-train-v2",
           # Nutrition5k metadata + official splits (small) travel with the code for the portion notebook
           "extra_files": ["data/raw/nutrition5k/metadata", "data/raw/nutrition5k/dish_ids"]},
    # v3 data audit only: no GPU training; inventories attached source folders and writes CSV/Markdown reports.
    "audit": {"data": "configs/data_v3.yaml", "classes": "configs/classes_v3.yaml",
              "manifest": "data/splits/exclude_v3.csv", "train_extra": "",
              "code_slug": "smartbite-cv-code-v3-audit", "kernel_slug": "smartbite-dataset-audit-v3",
              "kind": "audit", "enable_gpu": False},
    # portion (grams) model: reuses the v2 code dataset; images come from a Kaggle mirror of Nutrition5k
    "portion": {"data": "configs/data_v2.yaml", "classes": "configs/classes_v2.yaml", "manifest": "data/splits/manifest_v2.csv",
                "train_extra": "", "code_slug": "smartbite-cv-code-v2", "kernel_slug": "smartbite-portion-train",
                "kind": "portion", "sources": ["gillesokhin/nutrition5k-dataset"]},
    # plate as the ruler: benchmark the first portion model with plate scaling, then train a plate-relative one
    "plate": {"data": "configs/data_v2.yaml", "classes": "configs/classes_v2.yaml", "manifest": "data/splits/manifest_v2.csv",
              "train_extra": "", "code_slug": "smartbite-cv-code-v3", "kernel_slug": "smartbite-plate-portion",
              "kind": "plate", "sources": ["gillesokhin/nutrition5k-dataset"],
              "extra_files": ["data/raw/nutrition5k/metadata", "data/raw/nutrition5k/dish_ids",
                              "models/food_weight_finder_no_plate.onnx", "models/food_weight_finder_no_plate_settings.json",
                              "data/plate_cache/detections_overhead.jsonl"]},
}
MODELS = ["efficientnet_b0", "mobilenetv2"]
# Kaggle notebooks may run without internet (accounts without phone verification), so the ImageNet
# weights and a timm wheel travel inside the private code dataset: kaggle_out/offline/ -> offline/.
OFFLINE = ROOT / "kaggle_out" / "offline"
WEIGHTS = {"efficientnet_b0": "offline/efficientnet_b0.ra_in1k.safetensors",
           "mobilenetv2": "offline/mobilenetv2_100.ra_in1k.safetensors"}

SETUP = r'''# SmartBite food recognition - training run (Kaggle GPU or Colab)
import os, sys, glob, json, shutil, subprocess, zipfile, time
CODE_SLUG, DATA_CFG, CLASSES_CFG, TRAIN_EXTRA = "@CODE_SLUG@", "@DATA_CFG@", "@CLASSES_CFG@", "@TRAIN_EXTRA@"
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
    # The private code dataset is either auto-extracted or still a zip; handle both. Look in its own
    # mount first: a recursive search over every attached image dataset takes minutes.
    # Mounts are /kaggle/input/<slug> or /kaggle/input/datasets/<owner>/<slug>.
    mounts = glob.glob(f"/kaggle/input/{CODE_SLUG}") + glob.glob(f"/kaggle/input/*/*/{CODE_SLUG}")
    inits = [i for m in mounts for i in glob.glob(f"{m}/**/smartbite_cv/__init__.py", recursive=True)]
    if inits:
        shutil.copytree(os.path.dirname(os.path.dirname(inits[0])), WORK, dirs_exist_ok=True)
    else:
        zips = [z for m in mounts for z in glob.glob(f"{m}/**/smartbite_code.zip", recursive=True)]
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
# Private upload datasets (e.g. the Khana subset) may arrive as zip archives: unpack them where
# smartbite_cv.prepare also looks (/kaggle/working/extracted/<dataset>/<archive name>/...).
for z in (glob.glob("/kaggle/input/smartbite-*-subset/**/*.zip", recursive=True)
          + glob.glob("/kaggle/input/*/*/smartbite-*-subset/**/*.zip", recursive=True)):
    slug = [part for part in z.split("/") if part.startswith("smartbite-") and part.endswith("-subset")][0]
    dest = f"/kaggle/working/extracted/{slug}/{os.path.splitext(os.path.basename(z))[0]}"
    if not os.path.exists(dest):
        zipfile.ZipFile(z).extractall(dest)
        print("unpacked", z, "->", dest)
# Kaggle unpacks an uploaded folder archive but drops its top folder (khana/biryani/x -> biryani/x), while the
# manifest keeps khana/...: link the mount back under that name where smartbite_cv.prepare looks.
for d in sorted(set(glob.glob("/kaggle/input/smartbite-*-subset") + glob.glob("/kaggle/input/*/smartbite-*-subset")
                    + glob.glob("/kaggle/input/*/*/smartbite-*-subset"))):
    ds = os.path.basename(d)[len("smartbite-"):-len("-subset")]
    link_parent = f"/kaggle/working/extracted/{os.path.basename(d)}"
    if os.path.isdir(d) and not os.path.exists(f"{d}/{ds}") and not os.path.exists(f"{link_parent}/{ds}"):
        os.makedirs(link_parent, exist_ok=True)
        os.symlink(d, f"{link_parent}/{ds}")
        print("linked", d, "->", f"{link_parent}/{ds}")
try:
    run(f"python -m smartbite_cv.prepare --config {DATA_CFG} --classes {CLASSES_CFG} --profile {PROFILE} --verify-only "
        f"--write-verified data/splits/manifest_verified.csv --cache --cache-dir {CACHE} --workers 4")
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
    run(f"python -m smartbite_cv.train --config configs/train/base.yaml --config configs/train/{{m}}.yaml {{TRAIN_EXTRA}} "
        f"{{COMMON}} {{weights}} --out runs/{{m}}")
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


PORTION = r'''# Portion (grams) model on Nutrition5k overhead photos.
# 1. The images come from a Kaggle mirror: check a random sample against Google's official bucket (MD5).
import base64, hashlib, random, urllib.parse, urllib.request
# depth-limited: a recursive search would walk every file of the 23 GB mirror
marker = [m for depth in range(4) for m in glob.glob("/kaggle/input/" + "*/" * depth + "imagery/realsense_overhead")]
assert marker, "attach a Nutrition5k mirror with imagery/realsense_overhead"
N5K = os.path.dirname(os.path.dirname(marker[0]))
test_ids = [l.strip() for l in open(f"{WORK}/data/raw/nutrition5k/dish_ids/splits/depth_test_ids.txt") if l.strip()]
have = [d for d in test_ids if os.path.exists(f"{N5K}/imagery/realsense_overhead/{d}/rgb.png")]
sample = random.Random(20261031).sample(have, min(100, len(have)))
bad, checked = [], 0
for d in sample:
    name = urllib.parse.quote(f"nutrition5k_dataset/imagery/realsense_overhead/{d}/rgb.png", safe="")
    try:
        with urllib.request.urlopen(f"https://storage.googleapis.com/storage/v1/b/nutrition5k_dataset/o/{name}?fields=md5Hash", timeout=30) as r:
            official = json.load(r)["md5Hash"]
    except Exception as exc:
        print("could not reach the official bucket:", exc)
        break
    local = base64.b64encode(hashlib.md5(open(f"{N5K}/imagery/realsense_overhead/{d}/rgb.png", "rb").read()).digest()).decode()
    checked += 1
    if local != official:
        bad.append(d)
print(f"mirror check: {checked} sampled test images compared with the official bucket, {len(bad)} differ; "
      f"{len(have)}/{len(test_ids)} test dishes have an overhead photo in the mirror")
assert not bad, f"mirror images differ from the official dataset: {bad[:5]}"
os.makedirs(f"{OUT}/portion", exist_ok=True)
json.dump({"checked": checked, "mismatches": bad, "test_dishes_with_photo": len(have), "test_dishes": len(test_ids)},
          open(f"{OUT}/portion/mirror_check.json", "w"), indent=2)
# 2. Train and evaluate (test split used once).
weights = ("--set model.pretrained_file=offline/efficientnet_b0.ra_in1k.safetensors"
           if os.path.exists(f"{WORK}/offline/efficientnet_b0.ra_in1k.safetensors") else "")
t0 = time.time()
run(f"python -m smartbite_cv.portion train --config configs/train/portion.yaml --profile kaggle {weights} --out runs/portion")
run("python -m smartbite_cv.portion evaluate --run runs/portion --profile kaggle")
print(f"portion model: {(time.time() - t0) / 60:.1f} min")
for f in ["best.pt", "history.csv", "config_resolved.json"]:
    shutil.copy(f"{WORK}/runs/portion/{f}", f"{OUT}/portion/")
shutil.copytree(f"{WORK}/runs/portion/eval", f"{OUT}/portion/eval", dirs_exist_ok=True)
if ON_KAGGLE:
    shutil.rmtree(WORK, ignore_errors=True)
print(open(f"{OUT}/portion/eval/portion_metrics.json").read())
'''


PLATE = r'''# Plate as the ruler (needs internet: Hugging Face weights, Google's Nutrition5k bucket).
import csv, random
run('pip install -q -U "transformers>=4.56" onnx onnxruntime')
run("python -c \"import transformers, onnxruntime; from transformers import Sam2Model, Owlv2ForObjectDetection; print('transformers', transformers.__version__)\"")
HAVE_ORT = True
marker = [m for depth in range(4) for m in glob.glob("/kaggle/input/" + "*/" * depth + "imagery/realsense_overhead")]
assert marker, "attach a Nutrition5k mirror with imagery/realsense_overhead"
N5K = os.path.dirname(os.path.dirname(marker[0]))
SPLITS = f"{WORK}/data/raw/nutrition5k/dish_ids/splits"
P = f"{OUT}/plate"
os.makedirs(P, exist_ok=True)
t0 = time.time()
# 0. Overhead + zoom detections from the previous run (same detector, same images; side stills are redone).
seed = f"{WORK}/data/plate_cache/detections_overhead.jsonl"
if os.path.exists(seed):
    shutil.copy(seed, f"{P}/detections.jsonl")
# 1. One complete still per side-angle camera: HTTP range request for the head of each video (official bucket).
for split, d in (("depth_test", "/tmp/side_test"), ("depth_train", "/tmp/side_train")):
    run(f"python tools/fetch_side_frames.py --ids {SPLITS}/{split}_ids.txt --out {d} --workers 32")
    shutil.copy(f"{d}/frames.csv", f"{P}/frames_{split}.csv")
print(f"stills: {(time.time() - t0) / 60:.1f} min", flush=True)
# 1b. Plate detection for every uncached photo, one process per GPU, then one merged cache.
gpus = max(1, torch.cuda.device_count())
src = f"--n5k {N5K} --side-train /tmp/side_train --side-test /tmp/side_test"
for i in range(gpus):   # every shard starts from the same cache, so the shards split the same to-do list
    if os.path.exists(f"{P}/detections.jsonl"):
        shutil.copy(f"{P}/detections.jsonl", f"/tmp/det_{i}.jsonl")
run("(" + " ".join(f"CUDA_VISIBLE_DEVICES={i} python -m smartbite_cv.plate_portion prefill {src} --cache /tmp/det_{i}.jsonl "
                   f"--shard {i}/{gpus} & p{i}=$!;" for i in range(gpus))
    + " " + " && ".join(f"wait $p{i}" for i in range(gpus)) + ")")
recs = {}
for i in range(gpus):
    for line in (open(f"/tmp/det_{i}.jsonl") if os.path.exists(f"/tmp/det_{i}.jsonl") else []):
        if line.strip():
            r = json.loads(line)
            recs[r["key"]] = r
with open(f"{P}/detections.jsonl", "w") as f:
    f.writelines(json.dumps(r) + chr(10) for r in recs.values())
print(f"detections: {len(recs)} cached ({(time.time() - t0) / 60:.1f} min)", flush=True)
# 2. First portion model on test photos, as is and with plate scaling (reference plate from training photos).
run(f"python -m smartbite_cv.plate_bench --n5k {N5K} --side /tmp/side_test --portion models/food_weight_finder_no_plate.onnx "
    f"--cache {P}/detections.jsonl --out {P}/bench")
print(f"benchmark: {(time.time() - t0) / 60:.1f} min", flush=True)
# 3. Plate-only views (detections cached), plate-relative model, test split used once, ONNX export.
run(f"python -m smartbite_cv.plate_portion views --n5k {N5K} --side-train /tmp/side_train --side-test /tmp/side_test "
    f"--cache {P}/detections.jsonl --out /tmp/views")
shutil.copy("/tmp/views/views.csv", P)
print(f"views: {(time.time() - t0) / 60:.1f} min", flush=True)
weights = ("--set model.pretrained_file=offline/efficientnet_b0.ra_in1k.safetensors"
           if os.path.exists(f"{WORK}/offline/efficientnet_b0.ra_in1k.safetensors") else "")
run(f"python -m smartbite_cv.plate_portion train --views /tmp/views --config configs/train/plate_portion.yaml {weights} --out runs/plate_portion")
run(f"python -m smartbite_cv.plate_portion evaluate --run runs/plate_portion --views /tmp/views --old-preds {P}/bench/predictions.csv")
run("python -m smartbite_cv.plate_portion export --run runs/plate_portion --out artifacts/plate_portion")
shutil.copytree(f"{WORK}/artifacts/plate_portion", f"{P}/artifact", dirs_exist_ok=True)
for f in ["best.pt", "history.csv", "config_resolved.json"]:
    shutil.copy(f"{WORK}/runs/plate_portion/{f}", P)
shutil.copytree(f"{WORK}/runs/plate_portion/eval", f"{P}/eval", dirs_exist_ok=True)
rows = [r for r in csv.DictReader(open("/tmp/views/views.csv")) if r["file"] and r["split"] == "test"]
os.makedirs(f"{P}/views_sample", exist_ok=True)
for r in random.Random(1).sample(rows, min(40, len(rows))):
    shutil.copy(f"/tmp/views/{r['file']}", f"{P}/views_sample/")
print(f"total: {(time.time() - t0) / 60:.1f} min")
if ON_KAGGLE:
    shutil.rmtree(WORK, ignore_errors=True)
print(open(f"{P}/bench/summary.md").read())
print(open(f"{P}/eval/summary.md").read())
'''


AUDIT_SETUP = r'''# SmartBite v3 dataset source audit (CPU only; no model training)
import os, sys, glob, shutil, subprocess, zipfile
CODE_SLUG = "@CODE_SLUG@"
WORK = "/kaggle/working/SmartBite"

def run(cmd):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, cwd=WORK)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")

if not os.path.exists(f"{WORK}/smartbite_cv"):
    mounts = glob.glob(f"/kaggle/input/{CODE_SLUG}") + glob.glob(f"/kaggle/input/*/*/{CODE_SLUG}")
    inits = [i for m in mounts for i in glob.glob(f"{m}/**/smartbite_cv/__init__.py", recursive=True)]
    if inits:
        shutil.copytree(os.path.dirname(os.path.dirname(inits[0])), WORK, dirs_exist_ok=True)
    else:
        zips = [z for m in mounts for z in glob.glob(f"{m}/**/smartbite_code.zip", recursive=True)]
        assert zips, "Attach the generated private code dataset to this Kaggle notebook."
        zipfile.ZipFile(zips[0]).extractall(WORK)

run("python -m pip install -q pyyaml numpy scipy pillow pillow-heif")
run("python -c \"import yaml, numpy, scipy, PIL; print('dataset audit dependencies ready')\"")
'''

AUDIT_RUN = r'''# Inventory configured source folders. This does not decode, deduplicate, or train on images.
run("python tools/audit_v3_sources.py --profile kaggle --config configs/data_v3.yaml "
    "--classes configs/classes_v3.yaml --taxonomy configs/taxonomy_v3.yaml")
print(open(f"{WORK}/reports/data_v3/source_audit.md", encoding="utf-8").read())
shutil.make_archive("/kaggle/working/smartbite_v3_source_audit", "zip",
                    root_dir=f"{WORK}/reports/data_v3")
print("Download smartbite_v3_source_audit.zip from the notebook output.")
'''

def build_notebook(v: dict) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    if v.get("kind") == "audit":
        setup = nbformat.v4.new_code_cell(AUDIT_SETUP.replace("@CODE_SLUG@", v["code_slug"]))
        nb.cells = [
            nbformat.v4.new_markdown_cell(
                "# SmartBite v3: dataset source audit (CPU only)\n"
                "Inventories attached dataset folders against the v3 class/source mappings. "
                "No model training or GPU is used. Review the generated per-class raw counts before building a manifest."),
            setup,
            nbformat.v4.new_code_cell(AUDIT_RUN),
        ]
        return nb
    setup = nbformat.v4.new_code_cell(SETUP.replace("@CODE_SLUG@", v["code_slug"]).replace("@DATA_CFG@", v["data"])
                                      .replace("@CLASSES_CFG@", v["classes"]).replace("@TRAIN_EXTRA@", v["train_extra"]))
    if v.get("kind") == "plate":
        nb.cells = [nbformat.v4.new_markdown_cell(
            "# SmartBite: the plate as the ruler for grams\n"
            "Finds the plate (OWLv2 + SAM 2.1), undoes the camera tilt and scales by the plate. Benchmarks the first "
            "portion model with and without plate scaling on weighed Nutrition5k test dishes (overhead, zoomed, "
            "side-angle stills), then trains a plate-relative portion model on plate-only views."),
            setup, nbformat.v4.new_code_cell(PLATE)]
        return nb
    if v.get("kind") == "portion":
        nb.cells = [nbformat.v4.new_markdown_cell(
            "# SmartBite portion model: grams from one overhead photo\n"
            "Trains `smartbite_cv.portion` (EfficientNet-B0, log-mass + uncertainty) on Nutrition5k overhead RGB "
            "(official depth split), after checking the Kaggle mirror against Google's bucket."),
            setup, nbformat.v4.new_code_cell(PORTION)]
        return nb
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# SmartBite food recognition: training run\n"
            "Trains EfficientNet-B0 (primary) and MobileNetV2 (baseline) on the leakage-safe split in "
            f"`{v['manifest']}`, then evaluates on the untouched test split, calibrates (temperature "
            "scaling), chooses confidence bands on the validation split, exports ONNX and measures CPU latency.\n\n"
            f"**Kaggle:** attach the source datasets listed in `{v['data']}` and the private code dataset; "
            "GPU on; internet is not required (weights and timm are bundled). **Colab:** see the comment in the first cell."),
        setup,
        nbformat.v4.new_code_cell(VERIFY),
        nbformat.v4.new_code_cell(TRAIN.format(models=json.dumps(MODELS), weights=json.dumps(WEIGHTS))),
        nbformat.v4.new_code_cell(PACKAGE),
    ]
    return nb


def write_code_zip(code_dir: Path, v: dict, user: str, code_slug: str) -> None:
    with zipfile.ZipFile(code_dir / "smartbite_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE + [v["manifest"]] + v.get("extra_files", []):
            p = ROOT / item
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
            for f in files:
                z.write(f, f.relative_to(ROOT).as_posix())
        for f in sorted(OFFLINE.glob("*")):
            z.write(f, f"offline/{f.name}")
    (code_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": code_slug, "id": f"{user}/{code_slug}", "licenses": [{"name": "other"}]}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", required=True, help="Kaggle username that will own the private dataset/notebook")
    ap.add_argument("--version", choices=sorted(VERSIONS), default="v1")
    args = ap.parse_args()
    v = VERSIONS[args.version]
    CODE_SLUG, KERNEL_SLUG = v["code_slug"], v["kernel_slug"]

    bundle = ROOT / "kaggle_out" / ("bundle" if args.version == "v1" else f"bundle_{args.version}")
    code_dir, kernel_dir = bundle / "code", bundle / "kernel"
    code_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    if v.get("kind") == "portion":   # reuses the v2 code dataset: build and upload that with --version v2
        code_dir = None
    else:
        write_code_zip(code_dir, v, args.user, CODE_SLUG)

    nb = build_notebook(v)
    suffix = "" if args.version == "v1" else f"_{args.version}"
    nb_name = {"portion": "smartbite_portion_train.ipynb", "plate": "smartbite_plate_portion.ipynb",
                "audit": "smartbite_dataset_audit_v3.ipynb"}.get(
        v.get("kind"), f"smartbite_food_train{suffix}.ipynb")
    nbformat.write(nb, ROOT / "notebooks" / nb_name)
    nbformat.write(nb, kernel_dir / f"{KERNEL_SLUG}.ipynb")
    data_cfg = yaml.safe_load((ROOT / v["data"]).read_text(encoding="utf-8"))
    # sources not hosted on Kaggle travel as private upload datasets (tools/make_upload_subset.py)
    sources = v.get("sources") or [d["kaggle_ref"] or f"{args.user}/smartbite-{ds}-subset"
                                   for ds, d in data_cfg["datasets"].items()]
    sources = sources + [f"{args.user}/{CODE_SLUG}"]
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": f"{KERNEL_SLUG}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": v.get("enable_gpu", True),
        "enable_internet": True, "dataset_sources": sources, "competition_sources": [], "kernel_sources": []},
        indent=2))
    size = f"code zip {(code_dir / 'smartbite_code.zip').stat().st_size / 1e6:.1f} MB; " if code_dir else ""
    print(f"{size}notebook + metadata in {kernel_dir}; dataset sources: {sources}")


if __name__ == "__main__":
    main()
