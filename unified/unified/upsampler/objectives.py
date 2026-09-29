"""Label-free training targets for the upsampler (docs/UPSAMPLER_PLAN.md §1C).

Both objectives return the same thing: the CT the module is guided by, the coarse
features it upsamples, and a target feature map on explicit output centers, all in
voxel units of the guide.

``down``  (plan's primary). Crop 2P^3 of CT. The encoder on the 2x average-pooled
          crop gives the coarse input; the encoder on the full-resolution crop gives
          the target, one octave finer. The guide is the full-resolution crop.
          Mismatch: at test time the coarse features come from a 1.5 mm input, here
          from a 3 mm one.
``label`` (fallback, not label-free). Class-balanced labelled patches; the module
          is trained through a throwaway linear head on segmentation loss. Volumes
          of the linear-probe bank are excluded, and validation uses held-out
          train volumes, so the probe's val split stays untouched.
``crop``  (AnyUp-style). Take a P^3 patch; its features are the coarse input,
          exactly as at test time. Cut a (P/r)^3 sub-crop, resample it r x back to
          P^3 and run the encoder: its cells are r x finer over that region and are
          the target. Mismatch: the target comes from a resampled (blurred) input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .data import AIR_HU, VolumeStore, class_balanced_starts, crop, random_start
from .encoders import FrozenEncoder, NormParams
from .geometry import default_centers, interp_to
from .module import aggregate

Centers = tuple


@dataclass
class Sample:
    guide_hu: torch.Tensor          # (B, 1, D, H, W)
    feats: torch.Tensor             # (B, C, d, h, w)
    feat_centers: Centers
    target: torch.Tensor            # (B, C, *out)
    out_centers: Centers


def _stack_crops(store: VolumeStore, sids, size, rng, enc: FrozenEncoder):
    hu, params = [], []
    for sid in sids:
        st = random_start(store.shape(sid), size, rng)
        hu.append(torch.from_numpy(crop(store.image(sid), st, size, AIR_HU).astype(np.float32)))
        params.append(store.norm(enc.stem, enc.intensity, sid))
    return torch.stack(hu)[:, None], params


def _normalize(hu: torch.Tensor, params: List[NormParams]) -> torch.Tensor:
    return torch.cat([p.apply(h[None]) for h, p in zip(hu, params)])


def _map_centers(cs, start: float, factor: float):
    """Centers ``u`` on a grid resampled by ``factor`` from a crop starting at
    ``start`` -> guide voxels (align_corners=False: x = start + (u + 0.5)/factor - 0.5)."""
    return tuple(start_k + (c + 0.5) / factor - 0.5 for c, start_k in zip(cs, start))


def sample_crop(store, sids, enc: FrozenEncoder, rng, patch=96, ratios=(2, 4)) -> Sample:
    size = (patch,) * 3
    hu, params = _stack_crops(store, sids, size, rng, enc)
    dev = enc.device
    x = _normalize(hu, params).to(dev)
    feats, fc = enc(x)
    r = int(rng.choice(ratios))
    m = patch // r
    s = [int(rng.integers(0, patch - m + 1)) for _ in range(3)]
    sub = x[:, :, s[0]:s[0] + m, s[1]:s[1] + m, s[2]:s[2] + m]
    sub = F.interpolate(sub, size=size, mode="trilinear", align_corners=False)
    target, tc = enc(sub)
    return Sample(hu.to(dev), feats.float(), fc, target.float(), _map_centers(tc, s, r))


def sample_down(store, sids, enc: FrozenEncoder, rng, patch=96) -> Sample:
    size = (2 * patch,) * 3
    hu, params = _stack_crops(store, sids, size, rng, enc)
    dev = enc.device
    hu = hu.to(dev)
    target, tc = enc(_normalize(hu, params))
    lo = F.avg_pool3d(hu, 2, 2)                       # a coarser scan of the same crop
    feats, fc = enc(_normalize(lo, params))
    # low-res voxel u covers hi-res voxels 2u, 2u+1 -> center 2u + 0.5
    fc = tuple(2 * c + 0.5 for c in fc)
    return Sample(hu, feats.float(), fc, target.float(), tc)


def sample_label(store, sids, enc: FrozenEncoder, rng, patch=96) -> Sample:
    """Plan fallback: class-balanced labelled patches; ``target`` holds the labels."""
    size = (patch,) * 3
    hu, lab, params = [], [], []
    for sid in sids:
        st = class_balanced_starts(store.label(sid), 1, size, rng)[0]
        hu.append(torch.from_numpy(crop(store.image(sid), st, size, AIR_HU).astype(np.float32)))
        lab.append(torch.from_numpy(crop(store.label(sid), st, size, 0).astype(np.int64)))
        params.append(store.norm(enc.stem, enc.intensity, sid))
    hu = torch.stack(hu)[:, None]
    dev = enc.device
    feats, fc = enc(_normalize(hu, params).to(dev))
    fine = tuple(hu.shape[2:])
    return Sample(hu.to(dev), feats.float(), fc, torch.stack(lab).to(dev),
                  default_centers(fine, fine))


def label_logits(model, head, s: Sample, out_stride: int = 2, trilinear: bool = False):
    """Full-resolution logits through the upsampler (or trilinear) with a linear head
    applied at coarse resolution. The head is thrown away after training."""
    fine = tuple(s.guide_hu.shape[2:])
    logits = head(s.feats)
    if trilinear:
        return interp_to(logits, s.feat_centers, default_centers(fine, fine))
    out_shape = tuple(n // out_stride for n in fine)
    w, tab = model.weights(s.guide_hu, s.feats, s.feat_centers, out_shape=out_shape)
    y = aggregate(w, logits, tab)
    if out_stride > 1:
        y = interp_to(y, default_centers(out_shape, fine), default_centers(fine, fine))
    return y


SAMPLERS = {"crop": sample_crop, "down": sample_down, "label": sample_label}


# ------------------------------------------------------------------ loss
class FeatureStats(torch.nn.Module):
    """Per-channel mean/std of an encoder's features, estimated once. The loss is
    computed on standardized features so no channel dominates by scale."""

    def __init__(self, C: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(C))
        self.register_buffer("std", torch.ones(C))

    @torch.no_grad()
    def fit(self, feats: Sequence[torch.Tensor]):
        x = torch.cat([f.float().movedim(1, -1).reshape(-1, f.shape[1]) for f in feats])
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(1e-4))
        return self

    def __call__(self, f):
        return (f - self.mean[None, :, None, None, None]) / self.std[None, :, None, None, None]


def feature_loss(pred, target, stats: FeatureStats, w_mse: float = 1.0):
    p, t = stats(pred), stats(target)
    cos = F.cosine_similarity(p, t, dim=1)
    mse = (p - t).pow(2).mean(1)
    return (1 - cos).mean() + w_mse * mse.mean(), cos.mean().detach(), mse.mean().detach()
