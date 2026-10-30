import csv
import math

import pytest

from smartbite_cv.config import load_classes
from smartbite_cv.nutrition import CsvNutritionProvider, NutritionInfo, calories, default_provider, nutrients_for


class FakeProvider:
    """Stands in for the Phase-2 Firestore provider: anything with .get(label)."""

    def get(self, label):
        if label != "x":
            return None
        return NutritionInfo("x", "X", 200.0, 10.0, 20.0, 5.0, 150.0, "test", "1", "", "", False, "")


def test_calories_formula_with_swappable_provider():
    p = FakeProvider()
    assert calories("x", 150, p) == pytest.approx(300.0)
    assert calories("x", 0, p) == 0.0
    n = nutrients_for("x", 50, p)
    assert (n["kcal"], n["proteinG"], n["carbsG"], n["fatG"]) == (100.0, 5.0, 10.0, 2.5)


@pytest.mark.parametrize("grams", [-1, float("nan"), float("inf"), "100"])
def test_rejects_bad_grams(grams):
    with pytest.raises(ValueError):
        calories("x", grams, FakeProvider())


def test_unknown_label():
    with pytest.raises(KeyError):
        calories("nope", 100, FakeProvider())


def test_csv_covers_every_food_class_with_traceable_sources():
    provider = default_provider()
    food_classes = [c["slug"] for c in load_classes() if c["slug"] != "not_food"]
    assert sorted(provider.labels()) == sorted(food_classes)
    for label in food_classes:
        info = provider.get(label)
        assert info.kcal_per_100g > 0 and info.default_serving_g > 0
        assert info.source and info.source_id and info.source_description and info.source_url
        # Atwater sanity check: 4P + 4C + 9F should be within 25% of the stated energy
        atwater = 4 * info.protein_g_per_100g + 4 * info.carbs_g_per_100g + 9 * info.fat_g_per_100g
        assert math.isclose(atwater, info.kcal_per_100g, rel_tol=0.25), label
        if not info.source.startswith("USDA"):
            assert info.approximate, f"{label}: non-USDA rows must be flagged approximate"


def test_disagreeing_sources_are_flagged():
    with open(CsvNutritionProvider().path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["kcal_disagreement_pct"] and float(row["kcal_disagreement_pct"]) > 30:
                assert row["approximate"] == "true", row["label"]
