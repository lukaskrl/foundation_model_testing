"""Encoder-agnostic, image-guided 3D feature upsampler (docs/UPSAMPLER_PLAN.md §1B).

Every output voxel attends to the (2r+1)^3 coarse cells nearest to it and returns a
convex combination of the encoder's RAW coarse features:

    out(x) = sum_o  w_o(x) * F(n_o(x)),
    w(x)   = softmax_o( tau * <q(x), k(n_o(x))> / sqrt(dim) + b(rel_o(x)) )

* ``q`` comes from a small CNN over the CT, box-pooled onto the output grid.
* ``k`` is the same CNN output pooled onto the coarse grid, optionally plus a
  channel-count-invariant read of the features (AnyUp's feature-agnostic layer).
* ``b`` is a relative-position bias in coarse-cell units, separable over axes.

Because the output only mixes the encoder's own vectors, one set of weights serves
any channel count, and the module can place semantics but cannot invent them. For a
fixed set of weights the output is linear in ``F``, so a linear probe can be applied
BEFORE upsampling (``aggregate(weights, W @ F)``) at a fraction of the cost.

Memory: the 27 neighbours are never materialized together. Two custom autograd
functions loop over the offsets and recompute each gather in the backward pass, so
activation memory is O(one gather), not O(27 gathers).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import (NeighborTable, interp_to, neighbor_table, pool_matrices,
                       pool_to_grid, resolve_grids)


# ---------------------------------------------------------------- gathers
def _plain(t):
    """Drop tensor subclasses (MONAI MetaTensor) but keep autograd: their
    __torch_function__ overrides break new_empty and slow every op here."""
    return t.as_subclass(torch.Tensor) if type(t) is not torch.Tensor else t


def _gather(t, i0, i1, i2):
    """t (B, C, d, h, w) -> (B, C, n0, n1, n2), separable index gather."""
    return t.index_select(2, i0).index_select(3, i1).index_select(4, i2)


def _scatter(g, i0, i1, i2, shape):
    """Adjoint of ``_gather``: accumulate g (B, C, n0, n1, n2) onto (B, C, *shape)."""
    B, C, n0, n1, _ = g.shape
    d, h, w = shape
    t = g.new_zeros(B, C, n0, n1, w).index_add_(4, i2, g)
    t = g.new_zeros(B, C, n0, h, w).index_add_(3, i1, t)
    return g.new_zeros(B, C, d, h, w).index_add_(2, i0, t)


def _offset_indices(tab: NeighborTable):
    I0, I1, I2 = tab.idx
    return [(I0[a], I1[b], I2[c]) for a, b, c in tab.offsets]


def _acc_dtype(t):
    """Accumulate in at least float32 (bf16 features), but keep float64 if given."""
    return torch.float64 if t.dtype == torch.float64 else torch.float32


class _NeighborDot(torch.autograd.Function):
    """S[:, o] = <q, gather_o(k)> over channels.  q (B,Ck,*out), k (B,Ck,*coarse)."""

    @staticmethod
    def forward(ctx, q, k, tab: NeighborTable):
        idx = _offset_indices(tab)
        S = q.new_empty((q.shape[0], len(idx), *q.shape[2:]))
        for o, (i0, i1, i2) in enumerate(idx):
            S[:, o] = (q * _gather(k, i0, i1, i2)).sum(1)
        ctx.save_for_backward(q, k)
        ctx.tab = tab
        return S

    @staticmethod
    def backward(ctx, dS):
        q, k = ctx.saved_tensors
        idx = _offset_indices(ctx.tab)
        dq = torch.zeros_like(q) if ctx.needs_input_grad[0] else None
        dk = torch.zeros_like(k) if ctx.needs_input_grad[1] else None
        for o, (i0, i1, i2) in enumerate(idx):
            g = dS[:, o:o + 1]
            if dq is not None:
                dq += g * _gather(k, i0, i1, i2)
            if dk is not None:
                dk += _scatter(g * q, i0, i1, i2, k.shape[2:])
        return dq, dk, None


class _NeighborAggregate(torch.autograd.Function):
    """out = sum_o w[:, o] * gather_o(f).  w (B,K,*out), f (B,C,*coarse).

    Accumulates in (at least) float32 and walks the channels in chunks so a 1024-channel
    feature map never needs more than one chunk-sized temporary per offset."""

    @staticmethod
    def forward(ctx, w, f, tab: NeighborTable, chunk: int):
        idx = _offset_indices(tab)
        B, C = f.shape[:2]
        dt = _acc_dtype(f)
        out = torch.zeros(B, C, *w.shape[2:], device=f.device, dtype=dt)
        for c0 in range(0, C, chunk):
            fc = f[:, c0:c0 + chunk]
            acc = out[:, c0:c0 + chunk]
            for o, (i0, i1, i2) in enumerate(idx):
                acc += w[:, o:o + 1].to(dt) * _gather(fc, i0, i1, i2).to(dt)
        ctx.save_for_backward(w, f)
        ctx.tab, ctx.chunk = tab, chunk
        return out

    @staticmethod
    def backward(ctx, dout):
        w, f = ctx.saved_tensors
        idx = _offset_indices(ctx.tab)
        dt = _acc_dtype(f)
        dw = torch.zeros(w.shape, device=w.device, dtype=dt) if ctx.needs_input_grad[0] else None
        df = torch.zeros(f.shape, device=f.device, dtype=dt) if ctx.needs_input_grad[1] else None
        C = f.shape[1]
        for c0 in range(0, C, ctx.chunk):
            fc = f[:, c0:c0 + ctx.chunk]
            gc = dout[:, c0:c0 + ctx.chunk].to(dt)
            for o, (i0, i1, i2) in enumerate(idx):
                if dw is not None:
                    dw[:, o] += (gc * _gather(fc, i0, i1, i2).to(dt)).sum(1)
                if df is not None:
                    df[:, c0:c0 + ctx.chunk] += _scatter(
                        w[:, o:o + 1].to(dt) * gc, i0, i1, i2, f.shape[2:])
        return (dw.to(w.dtype) if dw is not None else None,
                df.to(f.dtype) if df is not None else None, None, None)


def neighbor_dot(q, k, tab):
    return _NeighborDot.apply(_plain(q), _plain(k), tab)


DENSE_MAX_CELLS = 1024


def _aggregate_dense(w, f, tab):
    """Same result as the offset loop, as one batched GEMM with an explicit
    (N_out, N_coarse) weight matrix. Much faster when the coarse grid is small
    (216 cells for a 6^3 ViT grid) and the map is thin (118 logit channels)."""
    B, K = w.shape[:2]
    out = w.shape[2:]
    N = out[0] * out[1] * out[2]
    d, h, ww = tab.coarse_shape
    I0, I1, I2 = tab.idx
    flat = torch.stack([(I0[a][:, None, None] * (h * ww) + I1[b][None, :, None] * ww
                         + I2[c][None, None, :]).reshape(-1) for a, b, c in tab.offsets])
    dt = _acc_dtype(f)
    A = torch.zeros(B, N, d * h * ww, device=f.device, dtype=dt).scatter_add(
        2, flat.t()[None].expand(B, N, K), w.reshape(B, K, N).transpose(1, 2).to(dt))
    y = torch.bmm(A, f.reshape(B, f.shape[1], -1).transpose(1, 2).to(dt))
    return y.transpose(1, 2).reshape(B, f.shape[1], *out)


def aggregate(w, f, tab, chunk: int = 256, dense: bool = None):
    """Apply precomputed neighbourhood weights to any coarse map ``f``.

    ``dense=None`` picks the GEMM path when the coarse grid has at most
    ``DENSE_MAX_CELLS`` cells and the map has at most ``chunk`` channels (a probe's
    logits), and the memory-light offset loop otherwise (1024-channel features)."""
    w, f = _plain(w), _plain(f)
    n_c = tab.coarse_shape[0] * tab.coarse_shape[1] * tab.coarse_shape[2]
    if dense is None:
        dense = n_c <= DENSE_MAX_CELLS and f.shape[1] <= chunk
    if dense:
        return _aggregate_dense(w, f, tab)
    return _NeighborAggregate.apply(w, f, tab, chunk)


def valid_mask(tab: NeighborTable, device) -> torch.Tensor:
    """(K, *out) bool: neighbour inside the coarse grid on all three axes."""
    V0, V1, V2 = (v.to(device) for v in tab.valid)
    return torch.stack([V0[a][:, None, None] & V1[b][None, :, None] & V2[c][None, None, :]
                        for a, b, c in tab.offsets])


def rel_positions(tab: NeighborTable, device):
    """Three (K, n_k) per-axis relative positions, one row per offset (a, b, c)."""
    R0, R1, R2 = (r.to(device) for r in tab.rel)
    offs = tab.offsets
    return (R0[[a for a, _, _ in offs]], R1[[b for _, b, _ in offs]],
            R2[[c for _, _, c in offs]])


# ---------------------------------------------------------------- modules
class ChannelNorm(nn.Module):
    """LayerNorm over channels at each voxel. Local, so a voxel's query does not
    depend on the rest of the crop, and a constant input stays constant."""

    def __init__(self, ch):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        return self.ln(x.movedim(1, -1)).movedim(-1, 1)


class GuidanceEncoder(nn.Module):
    """Small CNN over the windowed CT at full resolution. Replicate padding keeps a
    constant input exactly constant (no border artefacts)."""

    def __init__(self, in_ch: int, width: int = 32, depth: int = 3):
        super().__init__()
        layers = []
        c = in_ch
        for i in range(depth):
            layers.append(nn.Conv3d(c, width, 3, padding=1, padding_mode="replicate"))
            if i < depth - 1:
                layers += [ChannelNorm(width), nn.GELU()]
            c = width
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FeatureAgnosticLayer(nn.Module):
    """AnyUp's channel-count-invariant read of a feature map (arXiv 2510.12764).

    Each channel, standardized over space, is convolved with the same ``M`` learned
    3x3x3 filters; a softmax over the ``M`` responses is averaged across channels.
    Output (B, M, *) for any input width, and invariant to channel order."""

    def __init__(self, n_basis: int = 32, chunk: int = 128):
        super().__init__()
        self.basis = nn.Conv3d(1, n_basis, 3, padding=1, padding_mode="replicate")
        self.chunk = chunk

    def forward(self, f):
        B, C = f.shape[:2]
        f = f.float()
        f = (f - f.mean((2, 3, 4), keepdim=True)) / (f.std((2, 3, 4), keepdim=True) + 1e-5)
        acc = 0.0
        for c0 in range(0, C, self.chunk):
            fc = f[:, c0:c0 + self.chunk]
            n = fc.shape[1]
            r = self.basis(fc.reshape(B * n, 1, *f.shape[2:]))
            acc = acc + r.softmax(1).reshape(B, n, -1, *f.shape[2:]).sum(1)
        return acc / C


class AxisBias(nn.Module):
    """b(rel) = sum over axes of phi(rel_axis), phi(r) = -a * r^2 + mlp(r).

    Shared across axes (isotropic in cell units, so anisotropic grids are handled by
    the geometry, not by the weights). Initialised to a Gaussian with sigma = 0.5
    cells and a zero MLP, so before training the upsampler is a smooth Gaussian
    interpolation, not a uniform blur."""

    def __init__(self, hidden: int = 16, sigma0: float = 0.5):
        super().__init__()
        self.log_a = nn.Parameter(torch.tensor(math.log(1.0 / (2 * sigma0 ** 2))))
        self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def phi(self, r):
        return -self.log_a.exp() * r ** 2 + self.mlp(r[..., None])[..., 0]

    def forward(self, rels):
        """rels: three (K, n_k) tensors -> (K, n0, n1, n2)."""
        p0, p1, p2 = (self.phi(r) for r in rels)
        return p0[:, :, None, None] + p1[:, None, :, None] + p2[:, None, None, :]


DEFAULT_WINDOWS = ((-1000.0, 1000.0), (-150.0, 250.0))


class GuidedUpsampler3D(nn.Module):
    """See module docstring.

    forward(ct_hu, feats, feat_centers=None, out_shape=None, out_centers=None)
      ct_hu        (B, 1, D, H, W) raw Hounsfield units on the fine grid, RAS
      feats        (B, C, d, h, w) frozen encoder features, same frame and FOV
      feat_centers per-axis cell centers of ``feats`` in fine-voxel units
                   (default: laid evenly over the patch)
      out_shape    output grid, laid evenly over the patch (default: stride 2)
      out_centers  explicit output centers instead (e.g. a sub-crop only)
    """

    def __init__(self, dim: int = 32, width: int = 32, depth: int = 3, radius: int = 1,
                 windows: Sequence[Tuple[float, float]] = DEFAULT_WINDOWS,
                 feature_keys: bool = False, n_basis: int = 32, out_stride: int = 2,
                 chunk: int = 256):
        super().__init__()
        self.windows = [tuple(map(float, w)) for w in windows]
        self.dim, self.radius, self.out_stride, self.chunk = dim, radius, out_stride, chunk
        self.guide = GuidanceEncoder(len(self.windows), width, depth)
        self.to_q = nn.Conv3d(width, dim, 1)
        self.to_k = nn.Conv3d(width, dim, 1)
        # Zero queries: scores start as the positional bias alone, i.e. the module
        # starts as a Gaussian interpolation and only moves from there.
        nn.init.zeros_(self.to_q.weight)
        nn.init.zeros_(self.to_q.bias)
        self.bias = AxisBias()
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.feature_keys = feature_keys
        if feature_keys:
            self.fa = FeatureAgnosticLayer(n_basis)
            self.fa_to_k = nn.Conv3d(n_basis, dim, 1)

    # ------------------------------------------------------------ helpers
    def window(self, ct_hu):
        chans = [((ct_hu.clamp(lo, hi) - lo) / (hi - lo)) * 2 - 1 for lo, hi in self.windows]
        return torch.cat(chans, 1)

    # ------------------------------------------------------------ core
    def weights(self, ct_hu, feats, feat_centers=None, out_shape=None, out_centers=None):
        """Attention weights (B, K, *out) and the neighbour table. Linear-probe
        users call this once, then ``aggregate`` on whatever map they like."""
        ct_hu, feats = _plain(ct_hu), _plain(feats)
        fine = tuple(ct_hu.shape[2:])
        coarse = tuple(feats.shape[2:])
        fc, oc = resolve_grids(fine, coarse, self.out_stride, feat_centers, out_shape,
                               out_centers)
        tab = neighbor_table(fc, oc, self.radius).to(ct_hu.device)
        g = self.guide(self.window(ct_hu.float()))
        q = self.to_q(pool_to_grid(g, pool_matrices(oc, fine)))
        k = self.to_k(pool_to_grid(g, pool_matrices(fc, fine)))
        if self.feature_keys:
            k = k + self.fa_to_k(self.fa(feats))
        S = neighbor_dot(q, k, tab) * (self.log_tau.exp() / math.sqrt(self.dim))
        S = S + self.bias(rel_positions(tab, ct_hu.device))[None]
        S = S.masked_fill(~valid_mask(tab, ct_hu.device)[None], float("-inf"))
        return S.softmax(1), tab

    def forward(self, ct_hu, feats, feat_centers=None, out_shape=None, out_centers=None):
        w, tab = self.weights(ct_hu, feats, feat_centers, out_shape, out_centers)
        return aggregate(w, feats, tab, self.chunk)


# ---------------------------------------------------------------- baselines
class TrilinearUpsampler(nn.Module):
    """Trilinear interpolation at any grid layout. On the default layout it equals
    ``F.interpolate(mode="trilinear", align_corners=False)``."""

    def __init__(self, out_stride: int = 2):
        super().__init__()
        self.out_stride = out_stride

    def weights(self, ct_hu, feats, feat_centers=None, out_shape=None, out_centers=None):
        raise NotImplementedError("use forward(); trilinear has no neighbourhood weights")

    def forward(self, ct_hu, feats, feat_centers=None, out_shape=None, out_centers=None):
        fc, oc = resolve_grids(tuple(ct_hu.shape[2:]), tuple(feats.shape[2:]), self.out_stride,
                               feat_centers, out_shape, out_centers)
        return interp_to(feats, fc, oc)


class BilateralUpsampler(nn.Module):
    """Classic joint bilateral upsampling (Kopf et al. 2007) in 3D, not learned.

    w_o ∝ exp(-|rel|^2 / 2 s^2) * exp(-(I(x) - I_cell)^2 / 2 r^2), with I the CT
    in the first window, averaged over the output voxel and over the coarse cell.
    Uses the same neighbourhood and aggregation as the learned module."""

    def __init__(self, spatial_sigma: float = 0.5, range_sigma: float = 0.1,
                 radius: int = 1, window=(-1000.0, 1000.0), out_stride: int = 2,
                 chunk: int = 256):
        super().__init__()
        self.s, self.r, self.radius = spatial_sigma, range_sigma, radius
        self.win, self.out_stride, self.chunk = window, out_stride, chunk

    def weights(self, ct_hu, feats, feat_centers=None, out_shape=None, out_centers=None):
        fine, coarse = tuple(ct_hu.shape[2:]), tuple(feats.shape[2:])
        fc, oc = resolve_grids(fine, coarse, self.out_stride, feat_centers, out_shape,
                               out_centers)
        tab = neighbor_table(fc, oc, self.radius).to(ct_hu.device)
        lo, hi = self.win
        I = (ct_hu.float().clamp(lo, hi) - lo) / (hi - lo)
        i_out = pool_to_grid(I, pool_matrices(oc, fine))
        i_cell = pool_to_grid(I, pool_matrices(fc, fine))
        R0, R1, R2 = rel_positions(tab, ct_hu.device)
        spatial = -(R0[:, :, None, None] ** 2 + R1[:, None, :, None] ** 2
                    + R2[:, None, None, :] ** 2) / (2 * self.s ** 2)
        S = spatial[None].expand(ct_hu.shape[0], *spatial.shape).clone()
        if math.isfinite(self.r):
            diff = torch.stack([(i_out - _gather(i_cell, *ix))[:, 0]
                                for ix in _offset_indices(tab)], 1)
            S = S - diff ** 2 / (2 * self.r ** 2)
        S = S.masked_fill(~valid_mask(tab, ct_hu.device)[None], float("-inf"))
        return S.softmax(1), tab

    def forward(self, ct_hu, feats, feat_centers=None, out_shape=None, out_centers=None):
        w, tab = self.weights(ct_hu, feats, feat_centers, out_shape, out_centers)
        return aggregate(w, feats, tab, self.chunk)
