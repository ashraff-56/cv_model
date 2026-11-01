"""Find a model's files, whether they sit in the shared models/ folder or in a training export folder.

models/ (what the app uses; names say what each model does):
    food_name_finder.onnx  +  food_name_finder_settings.json  +  food_name_finder_how_sure.yaml
    food_weight_finder_with_plate.onnx  +  food_weight_finder_with_plate_settings.json

Training export folder (what smartbite_cv.export / portion export write):
    <folder>/model.onnx  +  <folder>/model_meta.json or portion_meta.json  +  <folder>/bands.yaml
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFiles:
    onnx: Path
    settings: Path            # preprocessing, classes, calibration (JSON)
    how_sure: Path | None     # confidence-band thresholds (dish classifier only)


def model_files(path: str | Path, export_settings_name: str) -> ModelFiles:
    """`path` is a named .onnx file (settings next to it) or an export folder (fixed file names inside)."""
    p = Path(path)
    if p.suffix == ".onnx":
        stem = p.with_suffix("")
        files = ModelFiles(p, Path(f"{stem}_settings.json"), Path(f"{stem}_how_sure.yaml"))
    else:
        files = ModelFiles(p / "model.onnx", p / export_settings_name, p / "bands.yaml")
    for f in (files.onnx, files.settings):
        if not f.exists():
            raise FileNotFoundError(f"model file missing: {f}")
    return files if files.how_sure.exists() else ModelFiles(files.onnx, files.settings, None)
