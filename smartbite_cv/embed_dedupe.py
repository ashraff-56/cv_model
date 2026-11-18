"""Second de-duplication pass: ImageNet CNN embeddings + cosine similarity.

pHash misses re-cropped / re-framed copies of the same photo (we found such pairs straddling
train/test at pHash distance 8-10, while unrelated photos also appear at that distance).
Global-pooled features of a pretrained CNN are robust to crops and re-encoding, so pairs above a
high cosine threshold are treated as the same photo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .imageio import load_rgb


class _Files(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths, self.transform = paths, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            return self.transform(load_rgb(self.paths[i])), i
        except Exception:
            return torch.zeros(3, 224, 224), i


def _save(cache_file: Path | None, cached: dict[str, np.ndarray]) -> None:
    if cache_file is None or not cached:
        return
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    all_keys = list(cached)
    tmp = Path(cache_file).with_suffix(".tmp.npz")
    np.savez(tmp, keys=np.array(all_keys), emb=np.stack([cached[k] for k in all_keys]))
    tmp.replace(cache_file)


def embed(paths: list[Path], model_name: str = "efficientnet_b0.ra_in1k", batch_size: int = 64,
          workers: int = 8, cache_file: Path | None = None, keys: list[str] | None = None) -> np.ndarray:
    """L2-normalised embeddings, optionally cached on disk by `keys` (e.g. MD5)."""
    import timm
    from torchvision import transforms as T

    cached: dict[str, np.ndarray] = {}
    if cache_file is not None and Path(cache_file).exists():
        with np.load(cache_file, allow_pickle=False) as data:   # closed again: Windows can't replace an open file
            cached = dict(zip(data["keys"].tolist(), data["emb"]))
    keys = keys or [str(p) for p in paths]
    todo = [i for i, k in enumerate(keys) if k not in cached]
    if todo:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"embedding device={device}; images_to_embed={len(todo)}", flush=True)
        model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
        tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        loader = DataLoader(_Files([paths[i] for i in todo], tf), batch_size=batch_size, num_workers=workers,
                            pin_memory=(device.type == "cuda"))
        torch.set_num_threads(max(1, torch.get_num_threads()))
        done = 0
        with torch.inference_mode():
            for x, idx in loader:
                x = x.to(device, non_blocking=(device.type == "cuda"))
                feats = torch.nn.functional.normalize(model(x), dim=1).cpu().numpy().astype(np.float32)
                for j, f in zip(idx.tolist(), feats):
                    cached[keys[todo[j]]] = f
                done += len(idx)
                if done % (batch_size * 40) < batch_size:
                    print(f"  embedded {done}/{len(todo)}", flush=True)
                    _save(cache_file, cached)   # checkpoint: a crash (e.g. out of memory) keeps finished work
        _save(cache_file, cached)
    return np.stack([cached[k] for k in keys])


def similar_pairs(emb: np.ndarray, min_cosine: float, block: int = 1024):
    """Yield (i, j, cosine), i < j, with cosine >= min_cosine."""
    n = len(emb)
    for start in range(0, n, block):
        sims = emb[start:start + block] @ emb[start:].T
        ii, jj = np.nonzero(sims >= min_cosine)
        for i, j in zip(ii, jj):
            a, b = start + int(i), start + int(j)
            if a < b:
                yield a, b, float(sims[i, j])
