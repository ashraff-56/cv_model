"""Image loading shared by data prep, training and inference, so all three see the same pixels."""
from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

from PIL import Image, ImageOps

try:  # iPhones save HEIC by default; optional dependency
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - depends on environment
    pass

# Refuse absurdly large images instead of decoding them (decompression-bomb guard).
Image.MAX_IMAGE_PIXELS = 80_000_000

ImageSource = Union[str, Path, bytes, bytearray, BinaryIO]


def load_rgb(src: ImageSource) -> Image.Image:
    """Open an image, apply its EXIF orientation and return it as RGB.

    Phone cameras store portrait photos rotated with an EXIF flag; ignoring it feeds the
    model sideways food. Transparent images are flattened onto white.
    """
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            return background
        return im.convert("RGB")
