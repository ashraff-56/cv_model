"""Per-100 g nutrition lookup behind a small interface.

Phase 1 reads nutrition/nutrition.csv (USDA FoodData Central / INDB, sources recorded per row).
Phase 2 swaps in a provider backed by the Firestore `foods` collection; callers only use
`NutritionProvider.get()` and `calories()` / `nutrients_for()`.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from .config import REPO_ROOT

DEFAULT_CSV = REPO_ROOT / "nutrition" / "nutrition.csv"


@dataclass(frozen=True)
class NutritionInfo:
    label: str
    display_name: str
    kcal_per_100g: float
    protein_g_per_100g: float
    carbs_g_per_100g: float
    fat_g_per_100g: float
    default_serving_g: float
    source: str
    source_id: str
    source_url: str
    source_description: str
    approximate: bool
    notes: str


class NutritionProvider(Protocol):
    def get(self, label: str) -> NutritionInfo | None: ...


class CsvNutritionProvider:
    def __init__(self, path: str | Path = DEFAULT_CSV):
        self.path = Path(path)
        with open(self.path, newline="", encoding="utf-8") as f:
            self._rows = {r["label"]: _parse(r) for r in csv.DictReader(f)}

    def get(self, label: str) -> NutritionInfo | None:
        return self._rows.get(label)

    def labels(self) -> list[str]:
        return list(self._rows)


def _parse(r: dict) -> NutritionInfo:
    return NutritionInfo(
        label=r["label"], display_name=r["display_name"],
        kcal_per_100g=float(r["kcal_per_100g"]), protein_g_per_100g=float(r["protein_g_per_100g"]),
        carbs_g_per_100g=float(r["carbs_g_per_100g"]), fat_g_per_100g=float(r["fat_g_per_100g"]),
        default_serving_g=float(r["default_serving_g"]), source=r["source"], source_id=r["source_id"],
        source_url=r["source_url"], source_description=r["source_description"],
        approximate=r["approximate"].strip().lower() in ("1", "true", "yes"), notes=r.get("notes", ""))


@lru_cache(maxsize=1)
def default_provider() -> CsvNutritionProvider:
    return CsvNutritionProvider()


def _lookup(label: str, grams: float, provider: NutritionProvider | None) -> NutritionInfo:
    if not isinstance(grams, (int, float)) or not math.isfinite(grams) or grams < 0:
        raise ValueError(f"grams must be a finite number >= 0, got {grams!r}")
    info = (provider or default_provider()).get(label)
    if info is None:
        raise KeyError(f"no nutrition data for label {label!r}")
    return info


def calories(label: str, grams: float, provider: NutritionProvider | None = None) -> float:
    """kcal for `grams` of `label` = kcal per 100 g x grams / 100."""
    info = _lookup(label, grams, provider)
    return info.kcal_per_100g * grams / 100.0


def nutrients_for(label: str, grams: float, provider: NutritionProvider | None = None) -> dict:
    info = _lookup(label, grams, provider)
    f = grams / 100.0
    return {"label": label, "displayName": info.display_name, "grams": grams,
            "kcal": round(info.kcal_per_100g * f, 1), "proteinG": round(info.protein_g_per_100g * f, 1),
            "carbsG": round(info.carbs_g_per_100g * f, 1), "fatG": round(info.fat_g_per_100g * f, 1),
            "approximate": info.approximate, "source": info.source, "sourceId": info.source_id}
