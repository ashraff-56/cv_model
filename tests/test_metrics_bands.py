import numpy as np
from scipy.special import softmax

from smartbite_cv.bands import BandThresholds, assign_band, band_report, choose_thresholds
from smartbite_cv.metrics import (confusion_matrix, coverage_curve, fit_temperature, nll, per_class_prf, reliability,
                                  summarize, topk_correct, worst_confused_pairs)


def test_topk_and_prf_on_hand_example():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1], [0.2, 0.7, 0.1]])
    y = np.array([0, 1, 1, 1])
    assert topk_correct(probs, y, 1).tolist() == [True, False, False, True]
    assert topk_correct(probs, y, 2).tolist() == [True, True, True, True]
    prf = per_class_prf(y, probs.argmax(1), 3)
    assert prf["precision"][0] == 0.5 and prf["recall"][0] == 1.0
    assert np.isclose(prf["recall"][1], 1 / 3)
    cm = confusion_matrix(y, probs.argmax(1), 3)
    assert cm.tolist() == [[1, 0, 0], [1, 1, 1], [0, 0, 0]]
    pairs = worst_confused_pairs(cm, ["a", "b", "c"])
    assert {(p["a"], p["b"], p["total"]) for p in pairs} == {("a", "b", 1), ("b", "c", 1)}


def test_summary_macro_f1_ignores_absent_classes():
    probs = np.eye(3)[[0, 0, 1]]
    s = summarize(probs, np.array([0, 0, 1]), ["a", "b", "c"])
    assert s["top1"] == 1.0 and s["macro_f1"] == 1.0


def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2.0, size=(4000, 10))
    y = np.array([rng.choice(10, p=p) for p in softmax(true_logits, axis=1)])
    overconfident = true_logits * 2.5   # a model whose logits are 2.5x too sharp
    t = fit_temperature(overconfident, y)
    assert 2.2 < t < 2.8
    assert nll(overconfident, y, t) < nll(overconfident, y)


def test_ece_zero_for_perfectly_calibrated_bins():
    probs = np.array([[0.8, 0.2]] * 10)
    y = np.array([0] * 8 + [1] * 2)
    assert reliability(probs, y)["ece"] < 1e-9


def test_coverage_curve_is_monotone_in_coverage():
    rng = np.random.default_rng(1)
    probs = softmax(rng.normal(size=(500, 5)) * 3, axis=1)
    y = rng.integers(0, 5, 500)
    c = coverage_curve(probs, y)
    assert np.all(np.diff(c["coverage"]) > 0) and c["coverage"][-1] == 1.0
    assert np.isclose(c["accuracy"][-1], (probs.argmax(1) == y).mean())


def _synthetic_val(n=3000, seed=0):
    """Confidence-dependent correctness: P(correct) ~ confidence (a calibrated model)."""
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.2, 1.0, n)
    y = rng.integers(0, 5, n)
    probs = np.zeros((n, 5))
    for i in range(n):
        top = y[i] if rng.random() < conf[i] else (y[i] + 1) % 5
        probs[i] = (1 - conf[i]) / 4
        probs[i, top] = conf[i]
        if top != y[i] and rng.random() < 0.3:  # sometimes truth is outside the top 3
            probs[i, y[i]] = 0.0
    return probs / probs.sum(1, keepdims=True), y


def test_thresholds_meet_targets_on_the_set_they_were_chosen_on():
    probs, y = _synthetic_val()
    t = choose_thresholds(probs, y, high_top1_target=0.9, low_band_max_top3=0.5, min_count=30)
    conf = probs.max(1)
    assert 0 <= t.low <= t.high <= 1
    assert (probs.argmax(1) == y)[conf >= t.high].mean() >= 0.9
    if t.low > 0:
        assert topk_correct(probs, y, 3)[conf < t.low].mean() <= 0.5


def test_unreachable_target_disables_high_band():
    rng = np.random.default_rng(2)
    probs = softmax(rng.normal(size=(200, 4)), axis=1)
    y = rng.integers(0, 4, 200)   # random labels: accuracy ~25% everywhere
    t = choose_thresholds(probs, y, high_top1_target=0.99)
    assert t.high == 1.0


def test_band_assignment_and_not_food_override():
    t = BandThresholds(high=0.8, low=0.3, high_top1_target=0.95, low_band_max_top3=0.5)
    assert assign_band(0.9, "biryani", t) == ("high", "confident")
    assert assign_band(0.5, "biryani", t) == ("medium", "uncertain")
    assert assign_band(0.1, "biryani", t) == ("low", "low_confidence")
    assert assign_band(0.99, "not_food", t) == ("low", "not_food")
    probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    rep = band_report(probs, np.array([0, 1, 1]), ["biryani", "not_food"], t)
    assert rep["high"]["count"] == 1 and rep["low"]["count"] == 1 and rep["medium"]["count"] == 1


def test_thresholds_roundtrip_yaml(tmp_path):
    t = BandThresholds(high=0.71, low=0.22, high_top1_target=0.95, low_band_max_top3=0.5)
    t.save(tmp_path / "bands.yaml")
    assert BandThresholds.load(tmp_path / "bands.yaml") == t
