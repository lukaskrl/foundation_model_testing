"""Compact 1.5 mm volume store and patch sampling for the upsampler experiments.

The training cache (``/home/lukas/data/cache/TotalSeg/<fingerprint>``) stores
float32 images AND float32 labels, ~120 MB and ~2.5 s per case, which is too slow
for random crops. ``scripts/upsampler_prepare.py`` converts the 1.5 mm cache
(``data.use_source_affine: true``) once into memory-mappable arrays:

    <root>/index.json             {sid: {"shape": [D, H, W], "split": ..., "affine": ...}}
    <root>/<sid>.img.npy          int16 HU, RAS
    <root>/<sid>.lab.npy          uint8 labels (0 = background, 1..117)
    <root>/norm/<stem>.json       per-volume encoder normalization (filled lazily)

Everything here is in RAS voxel indices at 1.5 mm.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .encoders import NormParams, norm_params

AIR_HU = -1024


class VolumeStore:
    def __init__(self, root):
        self.root = Path(root)
        self.index: Dict[str, dict] = json.loads((self.root / "index.json").read_text())
        self._norm_cache: Dict[str, Dict[str, list]] = {}

    def sids(self, split: str) -> List[str]:
        return sorted(s for s, v in self.index.items() if v["split"] == split)

    def image(self, sid) -> np.ndarray:
        return np.load(self.root / f"{sid}.img.npy", mmap_mode="r")

    def label(self, sid) -> np.ndarray:
        return np.load(self.root / f"{sid}.lab.npy", mmap_mode="r")

    def shape(self, sid) -> Tuple[int, int, int]:
        return tuple(self.index[sid]["shape"])

    # --------------------------------------------------------- normalization
    def norm(self, stem: str, intensity: dict, sid: str) -> NormParams:
        """Per-volume encoder normalization, computed once and kept on disk."""
        path = self.root / "norm" / f"{stem}.json"
        table = self._norm_cache.get(stem)
        if table is None:
            table = json.loads(path.read_text()) if path.exists() else {}
            self._norm_cache[stem] = table
        if sid not in table:
            table[sid] = norm_params(intensity, self.image(sid)).as_list()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".tmp.{os.getpid()}")
            tmp.write_text(json.dumps(table))
            os.replace(tmp, path)
        return NormParams(*table[sid])


def crop(arr: np.ndarray, start: Sequence[int], size: Sequence[int], fill) -> np.ndarray:
    """``arr[start:start+size]`` with out-of-volume voxels set to ``fill``."""
    out = np.full(tuple(size), fill, dtype=arr.dtype)
    src, dst = [], []
    for s, n, N in zip(start, size, arr.shape):
        a, b = max(s, 0), min(s + n, N)
        if b <= a:
            return out
        src.append(slice(a, b)); dst.append(slice(a - s, b - s))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def random_start(shape, size, rng: np.random.Generator, align: int = 1) -> List[int]:
    """Uniform crop start (centred when the volume is smaller than the crop)."""
    st = []
    for N, n in zip(shape, size):
        if N <= n:
            st.append(-((n - N) // 2))
        else:
            s = int(rng.integers(0, N - n + 1))
            st.append(s - s % align)
    return st


def class_balanced_starts(label: np.ndarray, n: int, size: Sequence[int],
                          rng: np.random.Generator) -> List[List[int]]:
    """Patch starts centred on a voxel of a uniformly drawn class present in the
    volume, so rare and thin classes are sampled as often as large organs."""
    flat = np.asarray(label).ravel()
    fg = np.flatnonzero(flat)
    if fg.size == 0:
        return [random_start(label.shape, size, rng) for _ in range(n)]
    cls = flat[fg]
    present = np.unique(cls)
    starts = []
    for _ in range(n):
        c = rng.choice(present)
        pick = fg[cls == c]
        v = np.unravel_index(int(pick[rng.integers(len(pick))]), label.shape)
        starts.append([int(np.clip(x - s // 2, 0, N - s)) if N >= s else -((s - N) // 2)
                       for x, s, N in zip(v, size, label.shape)])
    return starts


def load_patch(store: VolumeStore, sid: str, start, size, with_label=True):
    img = torch.from_numpy(crop(store.image(sid), start, size, AIR_HU).astype(np.float32))
    lab = torch.from_numpy(crop(store.label(sid), start, size, 0).astype(np.int64)) \
        if with_label else None
    return img, lab
