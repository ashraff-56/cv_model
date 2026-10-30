"""PyTorch dataset over the manifest + the train/eval transforms."""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T

from .imageio import load_rgb

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def eval_resize(img_size: int, crop_pct: float) -> int:
    """Short-side resize before the centre crop (224 / 0.875 = 256)."""
    return int(math.floor(img_size / crop_pct))


def build_transforms(img_size: int, train: bool, aug: dict | None = None, crop_pct: float = 0.875,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD) -> T.Compose:
    bicubic = T.InterpolationMode.BICUBIC
    if not train:
        return T.Compose([
            T.Resize(eval_resize(img_size, crop_pct), interpolation=bicubic),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    aug = aug or {}
    jitter = aug.get("color_jitter", [0.2, 0.2, 0.15, 0.02])  # mild: colour is a real cue for food
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(aug.get("scale_min", 0.35), 1.0), interpolation=bicubic),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomRotation(aug.get("rotate_deg", 15), interpolation=T.InterpolationMode.BILINEAR)],
                      p=aug.get("rotate_p", 0.3)),
        T.ColorJitter(*jitter),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=aug.get("erasing_p", 0.15)),
    ])


class ManifestDataset(Dataset):
    """Rows from manifest.csv; reads the resized cache copy when present, else the original."""

    def __init__(self, rows: list[dict], class_to_idx: dict[str, int], roots: dict[str, Path],
                 transform, cache_dir: Path | None = None):
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.roots = roots
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def __len__(self) -> int:
        return len(self.rows)

    def path(self, i: int) -> Path:
        row = self.rows[i]
        if self.cache_dir is not None:
            cached = self.cache_dir / f"{row['md5']}.jpg"
            if cached.exists():
                return cached
        return self.roots[row["dataset"]] / row["relpath"]

    def __getitem__(self, i: int):
        img = load_rgb(self.path(i))
        return self.transform(img), self.class_to_idx[self.rows[i]["label"]]

    @property
    def targets(self) -> list[int]:
        return [self.class_to_idx[r["label"]] for r in self.rows]


def sqrt_balanced_sampler(targets: list[int], seed: int) -> WeightedRandomSampler:
    """Sample class c with weight ~ 1/sqrt(n_c): softens imbalance without heavy oversampling."""
    counts = Counter(targets)
    weights = torch.tensor([1.0 / math.sqrt(counts[t]) for t in targets], dtype=torch.double)
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(targets), replacement=True, generator=gen)
