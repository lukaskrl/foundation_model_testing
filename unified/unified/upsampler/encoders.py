"""Frozen encoders as the upsampler sees them: RAS in, last-layer features + exact
cell centers out (docs/UPSAMPLER_PLAN.md §1A).

``FrozenEncoder`` wraps an existing backbone from ``configs/models/<stem>.yaml``
and hides three encoder-specific facts from everything downstream:

  * the encoder's input frame (``model.preprocessing.axcodes``). Patches are cut
    in RAS, reoriented for the encoder, and the features are reoriented back, so
    every encoder's cells live in one frame;
  * its intensity normalization, expressed per VOLUME as
    ``(clip(hu, lo, hi) - center) * scale + shift``. All four modes in
    ``unified/data/transforms.py`` have this form; the percentile and z-score
    modes need whole-volume statistics, which ``norm_params`` computes once;
  * where its cells sit on the patch (``unified/upsampler/geometry.py``).

Only the encoder's own output is read: no adapter, no neck, no raw-input branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import (Centers, apply_ornt, centers_resize, centers_stride, ornt_between,
                       reorient_centers)

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------- intensity
@dataclass
class NormParams:
    lo: float
    hi: float
    center: float
    scale: float
    shift: float

    def apply(self, hu: torch.Tensor) -> torch.Tensor:
        return (hu.float().clamp(self.lo, self.hi) - self.center) * self.scale + self.shift

    def as_list(self):
        return [self.lo, self.hi, self.center, self.scale, self.shift]


def norm_params(intensity: dict, vol_hu) -> NormParams:
    """Per-volume parameters of the encoder's pretraining normalization.

    Mirrors ``_orient_intensity_list``: ``range`` (ScaleIntensityRanged),
    ``percentile`` (ScaleIntensityRangePercentilesd, channel_wise), ``znorm``
    (clamp, then z-score over voxels above ``mask_threshold``)."""
    mode = str(intensity.get("mode", "range")).lower()
    b_min, b_max = float(intensity.get("b_min", 0.0)), float(intensity.get("b_max", 1.0))
    clip = bool(intensity.get("clip", True))
    inf = float("inf")
    if mode == "range":
        a_min, a_max = float(intensity["a_min"]), float(intensity["a_max"])
        return NormParams(a_min if clip else -inf, a_max if clip else inf, a_min,
                          (b_max - b_min) / (a_max - a_min), b_min)
    v = np.asarray(vol_hu, dtype=np.float32).ravel()
    if mode == "percentile":
        lo, hi = np.percentile(v, [float(intensity.get("lower", 0.05)),
                                   float(intensity.get("upper", 99.95))])
        lo, hi = float(lo), float(hi)
        return NormParams(lo if clip else -inf, hi if clip else inf, lo,
                          (b_max - b_min) / (hi - lo), b_min)
    if mode == "znorm":
        a_min, a_max = float(intensity["a_min"]), float(intensity["a_max"])
        x = np.clip(v, a_min, a_max)
        m = x > float(intensity.get("mask_threshold", 0.0))
        vals = x[m] if m.any() else x
        std = float(vals.std(ddof=1))                     # torch.std is unbiased
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        return NormParams(a_min, a_max, float(vals.mean()), 1.0 / std, 0.0)
    raise ValueError(f"intensity mode {mode!r} not supported by the upsampler pipeline")


# ---------------------------------------------------------------- encoders
def load_model_cfg(stem: str) -> dict:
    from unified.utils import load_config
    return load_config(str(REPO / "configs" / "models" / f"{stem}.yaml"))


class FrozenEncoder:
    """``enc(x_ras)`` with ``x_ras`` (B, 1, D, H, W) already normalized by
    ``enc.norm_params(volume)`` returns ``(feats, centers)``: features (B, C, d, h, w)
    in RAS and their per-axis cell centers in voxel units of ``x_ras``."""

    def __init__(self, stem: str, device="cuda", pretrained: bool = True,
                 tap: Optional[int] = None, amp: bool = True):
        import unified.models.backbones  # noqa: F401  (registers backbones)
        from unified.models import build_backbone
        from unified.data.transforms import _resolved_preprocessing

        self.stem = stem
        self.cfg = load_model_cfg(stem)
        m = self.cfg["model"]
        self.name = m["name"]
        self.axcodes, self.intensity = _resolved_preprocessing(self.cfg)
        self.to_enc = ornt_between("RAS", self.axcodes)
        self.to_ras = ornt_between(self.axcodes, "RAS")
        weights = m.get("weights") if pretrained else None
        bb = build_backbone(self.name, weights=weights, **m.get("kwargs", {}))
        self.bb = bb.to(device).eval()
        for p in self.bb.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)
        self.tap = tap
        self.amp = amp and self.device.type == "cuda"

    # ------------------------------------------------------------ intensity
    def norm_params(self, vol_hu) -> NormParams:
        return norm_params(self.intensity, vol_hu)

    # ------------------------------------------------------------ features
    def _native(self, x) -> Tuple[torch.Tensor, Centers]:
        """Encoder-frame features and centers for an encoder-frame input ``x``."""
        D, H, W = x.shape[2:]
        bb = self.bb
        if self.name == "dino3d":
            # int n = the last n blocks; a tuple = global block indices
            n = 1 if self.tap is None else (int(self.tap),)
            f = bb.vit.get_intermediate_layers(x, n=n, reshape=True, return_class_token=False,
                                               norm=True)[-1]
            return f, tuple(centers_stride(n_, 16) for n_ in f.shape[2:])
        if self.name == "sam_med3d":
            f = bb._run_encoder(x)                    # resize to the 128^3 canvas
            return f, tuple(centers_resize(n_, N) for n_, N in zip(f.shape[2:], (D, H, W)))
        if self.name == "ctclip":
            f = bb._run_ctvit(x)                      # depth padded, in-plane resized
            tps = bb.encoder.temporal_patch_size
            cs = (centers_stride(f.shape[2], tps),
                  centers_resize(f.shape[3], H), centers_resize(f.shape[4], W))
            return f, cs
        native = bb.encoder_forward(x)
        feats = [t for t in native if torch.is_tensor(t) and t.dim() == 5 and t.shape[1] > 1]
        f = feats[self.tap if self.tap is not None else -1]
        if self.name == "ctfm":
            # SegResNet encoder: four stride-2 3x3x3 convs with padding 1 and no
            # pooling, so cell i is centred on input voxel 16 * i, not mid-cell
            # (derived from the layers in scripts/test_upsampler.py --encoders).
            s = D // f.shape[2]
            return f, tuple(centers_stride(n_, s, 0.0) for n_ in f.shape[2:])
        return f, tuple(centers_resize(n_, N) for n_, N in zip(f.shape[2:], (D, H, W)))

    @torch.no_grad()
    def __call__(self, x_ras: torch.Tensor):
        x = apply_ornt(x_ras.to(self.device).float(), self.to_enc)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            f, cs = self._native(x)
        f = apply_ornt(f, self.to_ras)
        cs, _ = reorient_centers(cs, tuple(x.shape[2:]), self.to_ras)
        return f, tuple(c.float() for c in cs)

    def feature_dim(self) -> int:
        with torch.no_grad():
            f, _ = self(torch.zeros(1, 1, 96, 96, 96))
        return int(f.shape[1])


def encoder_input(vol_hu_crop: torch.Tensor, p: NormParams) -> torch.Tensor:
    """(B, 1, D, H, W) HU -> encoder input with that volume's normalization."""
    return p.apply(vol_hu_crop)


def downsample_hu(hu: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Simulate a coarser scan: average-pool HU (anti-aliased), before normalizing."""
    return F.avg_pool3d(hu.float(), factor, factor)
