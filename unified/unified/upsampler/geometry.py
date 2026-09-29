"""Grid geometry for the 3D feature upsampler.

A grid is described PER AXIS by its cell centers, in voxel units of the fine patch
the encoder saw (the "guide"). That is enough to describe every encoder in the
suite exactly, without assuming a regular lattice laid over the patch:

  * resize, then patchify (SAM-Med3D; CT-CLIP in-plane): ``(i + 0.5) * D / d - 0.5``.
    This is also the layout ``F.interpolate(align_corners=False)`` assumes, so the
    trilinear baseline and the upsampler agree on where a cell is.
  * patchify without resizing (3DINO): ``16 * i + 7.5``.
  * pad, then patchify (CT-CLIP depth, padded at the end to a multiple of 10):
    ``10 * i + 4.5``. The last cell is partly padding.
  * a chain of stride-2 3x3x3 convs with padding 1 (CT-FM's SegResNet encoder):
    ``16 * i``. The receptive field of output ``i`` is centred on input ``2 * i``
    at every level, so it does NOT sit mid-cell.

Orientation helpers follow nibabel's ``apply_orientation`` semantics and are checked
against MONAI's ``Orientation`` in ``scripts/test_upsampler.py``. (The helpers in
``scripts/_probe_features.py`` are not: they get SPL's flips and SRA's permutation
wrong, though their round trip is consistent.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch

Centers = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# ----------------------------------------------------------------- centers
def centers_resize(n_cells: int, n_fine: int) -> torch.Tensor:
    """Cells laid evenly over the patch (resize-then-patchify, align_corners=False)."""
    i = torch.arange(n_cells, dtype=torch.float64)
    return (i + 0.5) * (n_fine / n_cells) - 0.5


def centers_stride(n_cells: int, stride: float, offset: float = None) -> torch.Tensor:
    """Cells at a fixed stride from the patch origin. ``offset`` defaults to mid-cell."""
    if offset is None:
        offset = (stride - 1.0) / 2.0
    return torch.arange(n_cells, dtype=torch.float64) * float(stride) + float(offset)


def default_centers(coarse_shape: Sequence[int], fine_shape: Sequence[int]) -> Centers:
    return tuple(centers_resize(int(c), int(f)) for c, f in zip(coarse_shape, fine_shape))


def _spacing(c: torch.Tensor) -> float:
    return float((c[-1] - c[0]) / (len(c) - 1)) if len(c) > 1 else 1.0


# ----------------------------------------------------------------- pooling
def cell_bounds(centers: torch.Tensor, n_fine: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extent ``[lo, hi)`` of each cell: midpoints between neighbouring centers,
    the two end cells mirrored, everything clipped to the patch ``[-0.5, n - 0.5]``."""
    c = centers.double()
    if len(c) == 1:
        lo = torch.tensor([-0.5], dtype=torch.float64)
        hi = torch.tensor([n_fine - 0.5], dtype=torch.float64)
        return lo, hi
    mid = (c[1:] + c[:-1]) / 2
    lo = torch.cat([c[:1] - (c[1] - c[0]) / 2, mid])
    hi = torch.cat([mid, c[-1:] + (c[-1] - c[-2]) / 2])
    return lo.clamp(-0.5, n_fine - 0.5), hi.clamp(-0.5, n_fine - 0.5)


def box_pool_matrix(centers: torch.Tensor, n_fine: int) -> torch.Tensor:
    """(n_cells, n_fine) row-normalized overlap of each cell with each fine voxel.

    Multiplying by it averages a fine signal over each cell's footprint, for any
    center layout (non-integer strides, padded cells, a single cell)."""
    lo, hi = cell_bounds(centers.detach().cpu(), n_fine)
    x = torch.arange(n_fine, dtype=torch.float64)
    ov = (torch.minimum(hi[:, None], x[None] + 0.5)
          - torch.maximum(lo[:, None], x[None] - 0.5)).clamp_min(0)
    s = ov.sum(1, keepdim=True)
    if (s <= 0).any():
        raise ValueError("a cell lies entirely outside the patch")
    return (ov / s).float()


def pool_to_grid(t: torch.Tensor, mats: Sequence[torch.Tensor]) -> torch.Tensor:
    """Separable box pooling of ``t`` (B, C, D, H, W) with three pool matrices."""
    m0, m1, m2 = (m.to(device=t.device, dtype=t.dtype) for m in mats)
    t = torch.einsum("bcdhw,wk->bcdhk", t, m2.t())
    t = torch.einsum("bcdhw,hk->bcdkw", t, m1.t())
    return torch.einsum("bcdhw,dk->bckhw", t, m0.t())


def pool_matrices(centers: Centers, fine_shape: Sequence[int]) -> List[torch.Tensor]:
    return [box_pool_matrix(c, int(n)) for c, n in zip(centers, fine_shape)]


# --------------------------------------------------------------- neighbours
@dataclass
class NeighborTable:
    """Separable neighbourhood of every output voxel on the coarse grid.

    Per axis ``k``: ``idx[k]`` (K, n_out_k) clamped coarse indices, ``valid[k]``
    whether the unclamped index was inside the grid, ``rel[k]`` the offset of that
    coarse center from the output voxel, in coarse-cell units. ``K = 2r + 1``.
    The 3D neighbourhood is the product of the three axes.
    """
    idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    valid: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    rel: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    coarse_shape: Tuple[int, int, int]
    out_shape: Tuple[int, int, int]

    @property
    def offsets(self) -> List[Tuple[int, int, int]]:
        K = [t.shape[0] for t in self.idx]
        return [(a, b, c) for a in range(K[0]) for b in range(K[1]) for c in range(K[2])]

    def to(self, device) -> "NeighborTable":
        mv = lambda ts: tuple(t.to(device) for t in ts)  # noqa: E731
        return NeighborTable(mv(self.idx), mv(self.valid), mv(self.rel),
                             self.coarse_shape, self.out_shape)


def neighbor_table(coarse_centers: Centers, out_centers: Centers,
                   radius: int = 1) -> NeighborTable:
    idx, valid, rel = [], [], []
    for c, x in zip(coarse_centers, out_centers):
        c = c.detach().cpu().double(); x = x.detach().cpu().double()
        n_c = len(c)
        s = _spacing(c)
        # nearest coarse center to every output voxel
        if n_c > 1:
            j = torch.searchsorted(c, x).clamp(1, n_c - 1)
            left_closer = (x - c[j - 1]).abs() <= (c[j] - x).abs()
            j = torch.where(left_closer, j - 1, j)
        else:
            j = torch.zeros(len(x), dtype=torch.long)
        offs = torch.arange(-radius, radius + 1)
        jj = j[None] + offs[:, None]                          # (K, n_out)
        ok = (jj >= 0) & (jj < n_c)
        jc = jj.clamp(0, n_c - 1)
        idx.append(jc.long())
        valid.append(ok)
        rel.append(((c[jc] - x[None]) / s).float())
    return NeighborTable(tuple(idx), tuple(valid), tuple(rel),
                         tuple(len(c) for c in coarse_centers),
                         tuple(len(x) for x in out_centers))


# -------------------------------------------------------------- orientation
def ornt_between(src_codes: str, dst_codes: str) -> np.ndarray:
    import nibabel as nib
    return nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(src_codes),
                                           nib.orientations.axcodes2ornt(dst_codes))


def apply_ornt(t: torch.Tensor, ornt: np.ndarray) -> torch.Tensor:
    """Reorient the last three axes of ``t`` like ``nibabel.apply_orientation``:
    flip input axis ``i`` if ``ornt[i, 1] == -1``, then input axis ``i`` becomes
    output axis ``ornt[i, 0]``."""
    lead = t.dim() - 3
    flips = [lead + i for i in range(3) if ornt[i, 1] == -1]
    if flips:
        t = torch.flip(t, flips)
    perm = np.argsort(ornt[:, 0])
    return t.permute(*range(lead), *(lead + int(p) for p in perm)).contiguous()


def reorient_centers(centers: Centers, fine_shape: Sequence[int],
                     ornt: np.ndarray) -> Tuple[Centers, Tuple[int, int, int]]:
    """Carry a grid's centers (and its fine patch shape) through ``apply_ornt``.

    Flipping axis ``i`` maps a center ``x`` to ``n_i - 1 - x`` and reverses the cell
    order, exactly as the flipped feature tensor reverses it."""
    cs = list(centers); ns = list(fine_shape)
    for i in range(3):
        if ornt[i, 1] == -1:
            cs[i] = torch.flip(ns[i] - 1 - cs[i], [0])
    perm = np.argsort(ornt[:, 0])
    return tuple(cs[int(p)] for p in perm), tuple(int(ns[int(p)]) for p in perm)


def resolve_grids(fine_shape, coarse_shape, out_stride: int, feat_centers=None,
                  out_shape=None, out_centers=None) -> Tuple[Centers, Centers]:
    """Fill in the default layouts: features laid evenly over the patch, output
    evenly at ``out_stride`` unless ``out_shape`` or explicit centers are given."""
    if feat_centers is None:
        feat_centers = default_centers(coarse_shape, fine_shape)
    if out_centers is None:
        if out_shape is None:
            out_shape = [max(1, int(n) // out_stride) for n in fine_shape]
        out_centers = default_centers(out_shape, fine_shape)
    return tuple(feat_centers), tuple(out_centers)


# ------------------------------------------------------------ interpolation
def linear_matrix(centers_c: torch.Tensor, centers_out: torch.Tensor) -> torch.Tensor:
    """(n_out, n_c) 1D linear interpolation between coarse centers, clamped to the
    end cells outside them. On the default layout this is exactly
    ``F.interpolate(mode="linear", align_corners=False)``."""
    c, x = centers_c.detach().cpu().double(), centers_out.detach().cpu().double()
    n = len(c)
    M = torch.zeros(len(x), n, dtype=torch.float64)
    if n == 1:
        M[:, 0] = 1.0
        return M.float()
    xc = x.clamp(c[0], c[-1])
    j = torch.searchsorted(c, xc, right=True).clamp(1, n - 1)
    t = (xc - c[j - 1]) / (c[j] - c[j - 1])
    rows = torch.arange(len(x))
    M[rows, j - 1] = 1 - t
    M[rows, j] += t
    return M.float()


def interp_to(f: torch.Tensor, feat_centers: Centers, out_centers: Centers) -> torch.Tensor:
    """Separable trilinear interpolation of ``f`` (B, C, d, h, w) at any output centers."""
    mats = [linear_matrix(c, o) for c, o in zip(feat_centers, out_centers)]
    return pool_to_grid(f.float(), mats)
