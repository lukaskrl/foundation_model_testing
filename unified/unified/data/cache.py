"""Disk cache for the MODEL-INDEPENDENT preprocessing prefix.

The cached prefix is ``EnsureTyped → Spacingd → CropForegroundd →
ClassesToIndicesd`` — deterministic, and identical for every encoder. The
resample is the bulk of per-sample CPU, so we run it once, pickle the result to
``<cache_dir>/<fingerprint>/<sid>.pt``, and load from there on every subsequent
access.

Orientation and intensity windowing are deliberately NOT cached even though they
are deterministic: they are per-encoder (``model.preprocessing``), so caching
them would fork the cache per encoder. They run on the fly after the load, from
``build_train_post_transforms`` / ``build_val_post_transforms``, which is what
lets ONE cache directory serve every model.

The fingerprint follows from that: it hashes only spacing, crop-foreground margin
and the class-index bake — **not** axcodes or the HU window. Two consequences
worth knowing:

* every Arm W window variant shares the existing cache, so forcing a window costs
  no repopulation;
* changing spacing or the crop margin forks the cache, and the old directory has
  to be deleted by hand once the new one is warm.

Random transforms (patch sampling, flips, intensity jitter, …) must stay
per-epoch and are passed in as ``post_transforms``.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import Dataset


def preprocessing_fingerprint(cfg) -> str:
    """Short stable hash of the cache-affecting preprocessing args.

    Only the model-INDEPENDENT transforms enter the cache (spacing, foreground
    margin, num_classes for class-index baking). Orientation and HU window run
    on the fly after the cache, so they do NOT appear here — one fingerprint
    therefore serves every model.
    """
    d = cfg["data"]
    spec = {
        "spacing": [float(x) for x in d["spacing"]],
        "crop_foreground_margin": int(d.get("crop_foreground_margin", 0)),
        # Including num_classes + a version bump invalidates pre-existing caches
        # that lack the ClassesToIndicesd output baked in. First epoch after
        # this change repopulates at a new fingerprint dir; the old cache can be
        # deleted manually once the new one is warm.
        "cls_indices": {"num_classes": int(d["num_classes"]),
                        "max_samples_per_class": 10000, "version": 1},
    }
    blob = json.dumps(spec, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


class CachedDataset(Dataset):
    """Wraps a base dataset with a disk-cached deterministic prefix.

    First access for a subject runs ``base[idx]`` → ``det_transforms`` and
    pickles the resulting dict to ``cache_dir/<sid>.pt``. Subsequent accesses
    load directly from disk and skip the raw NIfTI + resample entirely.

    ``post_transforms`` (if given) are applied AFTER the cache load, so
    augmentation remains stochastic per epoch.

    Writes are atomic (``tmp → rename``) so multiple dataloader workers
    populating the cache concurrently never produce a half-written file.
    """

    def __init__(
        self,
        base: Dataset,
        det_transforms: Callable[[dict], dict],
        cache_dir: os.PathLike,
        post_transforms: Optional[Callable[[Any], Any]] = None,
    ):
        self.base = base
        self.det = det_transforms
        self.post = post_transforms
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.base)

    def _cache_path(self, idx: int) -> Path:
        sid = self.base.items[idx].subject_id
        return self.cache_dir / f"{sid}.pt"

    def __getitem__(self, idx: int):
        path = self._cache_path(idx)
        if path.exists():
            data = torch.load(path, weights_only=False, map_location="cpu")
        else:
            data = self.base[idx]
            data = self.det(data)
            # PID-suffixed temp avoids collisions if two workers happen to
            # populate the same index concurrently; os.replace is atomic.
            tmp = path.with_suffix(f".tmp.{os.getpid()}")
            torch.save(data, tmp)
            os.replace(tmp, path)
        if self.post is not None:
            data = self.post(data)
        return data
