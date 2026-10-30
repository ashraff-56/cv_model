"""Leakage-safety of the data preparation on a small synthetic dataset (real code path)."""
import csv
from collections import Counter

import pytest

from smartbite_cv.prepare import build_manifest, collect


@pytest.fixture
def toy_dataset(tmp_path, photo):
    root = tmp_path / "ds"
    for c, cls in enumerate(["a", "b", "c"]):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(30):
            photo(1000 * c + i).save(d / f"{i}.jpg", quality=92)
        # near-duplicates: a smaller re-encoded copy and a mirror of image 0
        photo(1000 * c).resize((200, 150)).save(d / "dup_small.jpg", quality=70)
        photo(1000 * c).transpose(0).save(d / "dup_mirror.jpg", quality=92)
    # one picture filed under two labels -> must be dropped from both
    photo(5000).save(root / "a" / "conflict.jpg")
    photo(5000).save(root / "b" / "conflict.jpg")
    (root / "c" / "broken.jpg").write_bytes(b"not an image")
    return root


def _cfg(seed=7):
    return {"seed": seed, "image_exts": [".jpg"], "min_side": 64, "dedupe": {"max_hamming": 6},
            "split": {"train": 0.7, "val": 0.15, "test": 0.15}, "cap_per_class": 1000, "min_per_class": 5}


def _classes():
    return [{"slug": s, "sources": [{"toy": s}]} for s in ("a", "b", "c")]


def test_dedupe_conflicts_and_split(toy_dataset):
    rows, stats = build_manifest(_cfg(), _classes(), {"toy": toy_dataset}, workers=2)
    assert stats["unreadable_or_small"]["c"] == 1
    assert stats["label_conflict_clusters"] == 1
    assert not any("conflict" in r["relpath"] for r in rows)
    # each class: 30 originals, 2 duplicates of image 0 merged into one cluster
    assert Counter(r["label"] for r in rows) == {"a": 30, "b": 30, "c": 30}
    assert sum(1 for r in rows if r["relpath"].startswith("a/") and int(r["cluster_size"]) == 3) == 1
    # every image appears in exactly one split, and duplicates never straddle splits
    md5s = [r["md5"] for r in rows]
    assert len(md5s) == len(set(md5s))
    per_split = Counter((r["label"], r["split"]) for r in rows)
    assert per_split[("a", "train")] == 21 and per_split[("a", "val")] == 4 and per_split[("a", "test")] == 5


def test_split_is_deterministic_and_order_independent(toy_dataset, monkeypatch):
    rows1, _ = build_manifest(_cfg(), _classes(), {"toy": toy_dataset}, workers=2)
    import smartbite_cv.prepare as prep
    original = prep.collect
    monkeypatch.setattr(prep, "collect", lambda *a, **k: list(reversed(original(*a, **k))))
    rows2, _ = build_manifest(_cfg(), _classes(), {"toy": toy_dataset}, workers=2)
    assert {(r["md5"], r["split"]) for r in rows1} == {(r["md5"], r["split"]) for r in rows2}
    rows3, _ = build_manifest(_cfg(seed=8), _classes(), {"toy": toy_dataset}, workers=2)
    assert {(r["md5"], r["split"]) for r in rows1} != {(r["md5"], r["split"]) for r in rows3}


def test_collect_supports_globs(toy_dataset):
    recs = collect([{"slug": "ab", "sources": [{"toy": "[ab]"}]}], {"toy": toy_dataset}, {".jpg"})
    assert {r["relpath"].split("/")[0] for r in recs} == {"a", "b"}


def test_real_manifest_has_no_cross_split_duplicates():
    """The committed manifest: every MD5 and every pHash appears in only one split."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data" / "splits" / "manifest.csv"
    if not path.exists():
        pytest.skip("manifest not built")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for key in ("md5", "phash"):
        splits_by_key = {}
        for r in rows:
            splits_by_key.setdefault(r[key], set()).add(r["split"])
        assert all(len(s) == 1 for s in splits_by_key.values()), key
