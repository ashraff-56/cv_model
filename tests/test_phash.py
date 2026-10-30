import io

import numpy as np
from PIL import Image

from smartbite_cv.phash import cluster, hamming, near_duplicate_pairs, phash64, phash_pair


def _reencode(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def test_identical_and_recompressed_images_are_near_duplicates(photo):
    img = photo(1)
    assert hamming(phash64(img), phash64(img.copy())) == 0
    small = _reencode(img.resize((160, 120)), quality=60)
    assert hamming(phash64(img), phash64(small)) <= 6


def test_mirror_copy_is_caught_via_flipped_hash(photo):
    img = photo(2)
    mirrored = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    h, h_flip = phash_pair(img)
    m, _ = phash_pair(mirrored)
    assert hamming(h_flip, m) <= 2
    pairs = list(near_duplicate_pairs([h, m], [h_flip, phash_pair(mirrored)[1]], max_dist=6))
    assert pairs == [(0, 1)]


def test_different_images_are_far_apart(photo):
    hashes = [phash64(photo(s)) for s in range(10, 20)]
    dists = [hamming(a, b) for i, a in enumerate(hashes) for b in hashes[i + 1:]]
    assert min(dists) > 6


def test_cluster_is_transitive():
    ids = cluster(5, [(0, 1), (1, 2), (3, 4)])
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] == ids[4] != ids[0]


def test_pairs_blocking_matches_bruteforce():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2**63, size=40, dtype=np.uint64)
    hashes = np.concatenate([base, base ^ np.uint64(0b111)])  # 40 near-copies at distance 3
    flips = rng.integers(0, 2**63, size=80, dtype=np.uint64)
    got = set(near_duplicate_pairs(hashes, flips, max_dist=3, block=7))
    want = {(i, j) for i in range(80) for j in range(i + 1, 80)
            if min(hamming(int(hashes[i]), int(hashes[j])), hamming(int(flips[i]), int(hashes[j]))) <= 3}
    assert got == want
    assert all((i, i + 40) in got for i in range(40))
