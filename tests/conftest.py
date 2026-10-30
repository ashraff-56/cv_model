import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _photo(seed: int, size=(320, 240)) -> Image.Image:
    """Deterministic 'photo-like' image: smooth colour blobs plus texture (not just noise)."""
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.float64)
    for _ in range(6):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(20, 120)
        colour = rng.uniform(0, 255, 3)
        mask = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
        img += mask[..., None] * colour
    img += rng.normal(0, 6, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


@pytest.fixture
def photo():
    return _photo
