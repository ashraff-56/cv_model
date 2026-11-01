import numpy as np
from PIL import Image

from smartbite_cv.audit import (CORRUPTIONS, band_counts, mcnemar, paired_bootstrap_diff, similarity_bins)
from smartbite_cv.bands import BandThresholds


def _img(seed=0, size=(96, 64)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def test_corruptions_are_deterministic_rgb_and_change_the_image():
    img = _img()
    for name, fn in CORRUPTIONS.items():
        a, b = fn(img.copy()), fn(img.copy())
        assert a.mode == "RGB", name
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"{name} is not deterministic"
        if name != "clean":
            same_shape = np.asarray(a).shape == np.asarray(img).shape
            assert not same_shape or not np.array_equal(np.asarray(a), np.asarray(img)), f"{name} did nothing"


def test_rotation_and_crop_geometry():
    img = _img(size=(100, 50))
    assert CORRUPTIONS["rotated_90"](img).size == (50, 100)
    assert CORRUPTIONS["tight_crop"](img).size == (60, 30)
    assert CORRUPTIONS["low_resolution"](img).size == (100, 50)


def test_similarity_bins_partition_all_items():
    sims = np.array([0.1, 0.55, 0.65, 0.75, 0.85, 0.99])
    correct = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
    nn = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    bins = similarity_bins(sims, correct, nn)
    assert sum(b["n"] for b in bins) == len(sims)
    assert bins[-1]["n"] == 2 and bins[-1]["model_top1"] == 0.5 and bins[-1]["nn1_top1"] == 1.0


def test_mcnemar_counts_and_symmetry():
    a = np.array([1, 1, 1, 0, 0, 1, 1, 1], dtype=bool)
    b = np.array([1, 0, 0, 0, 1, 1, 0, 1], dtype=bool)
    r = mcnemar(a, b)
    assert (r["a_right_b_wrong"], r["a_wrong_b_right"]) == (3, 1)
    assert abs(r["p_value"] - mcnemar(b, a)["p_value"]) < 1e-12
    assert mcnemar(a, a)["p_value"] == 1.0


def test_paired_bootstrap_diff_contains_true_difference():
    rng = np.random.default_rng(1)
    a = rng.random(2000) < 0.9
    b = rng.random(2000) < 0.8
    lo, hi = paired_bootstrap_diff(a, b)
    assert lo < a.mean() - b.mean() < hi and lo > 0


def test_band_counts_route_not_food_to_low():
    classes = ["biryani", "pizza", "not_food"]
    t = BandThresholds(high=0.9, low=0.4, high_top1_target=0.95, low_band_max_top3=0.5)
    probs = np.array([[0.95, 0.03, 0.02], [0.5, 0.3, 0.2], [0.1, 0.1, 0.8], [0.02, 0.01, 0.97]])
    assert band_counts(probs, classes, t) == {"high": 0.25, "medium": 0.25, "low": 0.5}
