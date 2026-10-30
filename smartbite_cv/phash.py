"""Perceptual hashing and near-duplicate clustering for leakage-safe splits.

pHash: 32x32 greyscale -> 2-D DCT -> top-left 8x8 low frequencies -> bit = coefficient > median.
Implemented here (numpy only) so the exact algorithm is documented and has no extra dependency.
We hash each image and its horizontal mirror, so mirrored copies are caught too.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_N = 32


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m


_DCT = _dct_matrix(_N)


def phash64(img: Image.Image) -> int:
    grey = np.asarray(img.convert("L").resize((_N, _N), Image.Resampling.LANCZOS), dtype=np.float64)
    coeffs = (_DCT @ grey @ _DCT.T)[:8, :8].ravel()
    bits = coeffs > np.median(coeffs[1:])  # DC term excluded from the median
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def phash_pair(img: Image.Image) -> tuple[int, int]:
    """(hash of image, hash of its horizontal mirror)."""
    return phash64(img), phash64(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)  # numpy < 2.0
    return table[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)


def near_duplicate_pairs(hashes: np.ndarray, flipped: np.ndarray, max_dist: int, block: int = 512):
    """Yield (i, j), i < j, where image j is within `max_dist` bits of image i or of its mirror."""
    hashes = np.asarray(hashes, dtype=np.uint64)
    flipped = np.asarray(flipped, dtype=np.uint64)
    n = len(hashes)
    for start in range(0, n, block):
        stop = min(start + block, n)
        rows = hashes[start:stop, None]
        cols = hashes[None, start:]
        dist = np.minimum(_popcount(rows ^ cols), _popcount(flipped[start:stop, None] ^ cols))
        ii, jj = np.nonzero(dist <= max_dist)
        for i, j in zip(ii + start, jj + start):
            if i < j:
                yield int(i), int(j)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster(n: int, pairs) -> list[int]:
    """Cluster id (smallest member index) for every item, given duplicate pairs."""
    uf = UnionFind(n)
    for i, j in pairs:
        uf.union(i, j)
    return [uf.find(i) for i in range(n)]
