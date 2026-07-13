"""Mask2Former-style dense segmentation head (Regime A: fixed query→class).

A denser alternative to :class:`UnifiedSegHead`, registered as
``"mask2former_head"`` and selected purely from ``head.name`` in config. It
keeps the exact same public contract:

    head(x_in, feats) -> (B, num_classes, D, H, W)  |  List[Tensor]

so the existing DiceCE loss, deep-supervision wrapper, sliding-window
inference and evaluator all consume it unchanged.

Architecture (three parts, mirroring Mask2Former's meta-architecture):

1. **Dense pixel decoder (two-tier)** — a U-Net trunk (MONAI ``UnetrUpBlock``,
   the same block :class:`UnifiedSegHead` uses) that fuses the finest-first
   pyramid down and taps *two* per-voxel *mask-feature* maps: a **fine** map at
   ``mask_feat_stride`` used only for the final full-resolution readout, and a
   **coarse** map at ``attn_feat_stride`` (``>= mask_feat_stride``) that drives
   the per-layer masked-attention masks and the deep-supervision aux. Only ONE
   full-resolution mask readout happens per forward (the final one), so
   ``mask_feat_stride=1`` (finest masks — the accuracy win) no longer costs a
   full-res, ``num_classes``-channel logit volume *per decoder layer*: that
   ``(dec_layers+1)×`` blow-up is what forced the naive full-res run down to
   ``bs=2``/``dec_layers=3``. Set ``attn_feat_stride=None`` to collapse back to
   a single tier (the original behavior, for reproducing older runs).

2. **Transformer decoder** — ``dec_layers`` layers of *masked* cross-attention
   (each query attends only to its own predicted foreground) + self-attention
   + FFN, over the coarse encoder levels (``memory_strides``) fed round-robin.
   Cheap in 3D: N queries × a few thousand tokens.

3. **Mask prediction** — each query emits a ``mask_dim`` embedding; the dot
   product with the mask-feature map gives one mask logit volume per query.

**Regime A (implemented here): fixed assignment.** ``num_queries =
num_classes * queries_per_class`` and query ``k`` owns class ``k`` (no
Hungarian matching, no no-object class). The per-query mask logits ARE the
per-class logits (aggregated over ``queries_per_class``), so the output drops
straight into softmax DiceCE — one variable changed vs. the U-Net head (the
decoder), loss/eval identical. Regime B (learnable queries + matching + class
head) is a future flag; see ``_aggregate`` / the mask-embedding path for the
extension point.

Everything is size-agnostic (no fixed-size positional tables): memory carries
only a learned per-level embedding and masked attention supplies spatial
grounding — required because sliding-window / full-volume eval feed shapes
that differ from the training patch.
"""
from __future__ import annotations
from typing import List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import register_head


def _import_unetr_up():
    from monai.networks.blocks.unetr_block import UnetrUpBlock
    return UnetrUpBlock


class _MLP(nn.Module):
    """Small multi-layer perceptron (Mask2Former's mask-embedding head)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class _DecoderLayer(nn.Module):
    """Masked cross-attention → self-attention → FFN (post-norm), à la M2F."""

    def __init__(self, hidden_dim: int, nheads: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, q_pos, memory, mem_pos, attn_mask):
        # Masked cross-attention (queries → encoder memory of one scale).
        q2, _ = self.cross_attn(
            query=q + q_pos, key=memory + mem_pos, value=memory,
            attn_mask=attn_mask, need_weights=False,
        )
        q = self.norm1(q + q2)
        # Self-attention among queries.
        q2, _ = self.self_attn(query=q + q_pos, key=q + q_pos, value=q, need_weights=False)
        q = self.norm2(q + q2)
        # FFN.
        q2 = self.linear2(self.dropout(F.relu(self.linear1(q))))
        q = self.norm3(q + q2)
        return q


@register_head("mask2former_head")
class Mask2FormerHead(nn.Module):
    def __init__(
        self,
        num_classes: int = 118,
        feature_channels: Sequence[int] = (32, 64, 128, 256, 512),
        feature_strides: Sequence[int] = (1, 2, 4, 8, 16),
        decoder_channels: int = 32,     # accepted for build_head compat; unused
        norm: str = "instance",
        spatial_dims: int = 3,
        deep_supervision: bool = False,
        # --- Mask2Former-specific (defaulted; wire through config later) ---
        hidden_dim: int = 256,
        mask_dim: int = 128,
        dec_layers: int = 9,
        nheads: int = 8,
        dim_feedforward: int = 1024,
        mask_feat_stride: int = 2,
        attn_feat_stride: int = None,   # coarse tier for per-layer attn masks; None => == mask_feat_stride
        queries_per_class: int = 1,
        memory_strides: Sequence[int] = (16, 8, 4),
        **kwargs,
    ):
        super().__init__()
        if spatial_dims != 3:
            raise ValueError("Mask2FormerHead is 3D-only for now")
        UnetrUpBlock = _import_unetr_up()

        self.feature_channels = tuple(int(c) for c in feature_channels)
        self.feature_strides = tuple(int(s) for s in feature_strides)
        self.num_classes = int(num_classes)
        self.queries_per_class = int(queries_per_class)
        self.num_queries = self.num_classes * self.queries_per_class
        self.nheads = int(nheads)
        self.deep_supervision = bool(deep_supervision)
        n = len(self.feature_channels)

        if len(self.feature_strides) != n:
            raise ValueError("feature_channels and feature_strides length mismatch")
        if int(mask_feat_stride) not in self.feature_strides:
            raise ValueError(f"mask_feat_stride {mask_feat_stride} not in {self.feature_strides}")
        for s in memory_strides:
            if int(s) not in self.feature_strides:
                raise ValueError(f"memory stride {s} not in {self.feature_strides}")

        self.mask_level = self.feature_strides.index(int(mask_feat_stride))
        self.mem_levels = [self.feature_strides.index(int(s)) for s in memory_strides]

        # Two-tier mask features. attn_feat_stride=None collapses to a single
        # tier (== mask_feat_stride), exactly reproducing the pre-two-tier head.
        attn_stride = int(mask_feat_stride if attn_feat_stride is None else attn_feat_stride)
        if attn_stride not in self.feature_strides:
            raise ValueError(f"attn_feat_stride {attn_stride} not in {self.feature_strides}")
        self.attn_level = self.feature_strides.index(attn_stride)
        if self.attn_level < self.mask_level:
            raise ValueError("attn_feat_stride must be >= mask_feat_stride (coarser or equal)")
        # The coarse tap must be a *decoded* level (produced by an up block),
        # i.e. no coarser than the second-coarsest pyramid level.
        if self.attn_level > n - 2:
            raise ValueError(
                f"attn_feat_stride {attn_stride} is too coarse; must be a decoded "
                f"level (stride <= {self.feature_strides[n - 2]})"
            )
        self.two_tier = self.attn_level > self.mask_level

        # 1. Dense pixel decoder: UnetrUpBlock trunk from coarsest down to
        #    mask_level. up_ks[i] produces the feature at level up_ks[i] from
        #    level up_ks[i]+1 fused with skip feats[up_ks[i]].
        self.up_ks = list(range(n - 2, self.mask_level - 1, -1))  # e.g. [3, 2, 1]
        self.ups = nn.ModuleList([
            UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feature_channels[k + 1],
                out_channels=self.feature_channels[k],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm,
                res_block=True,
            )
            for k in self.up_ks
        ])
        self.mask_proj = nn.Conv3d(self.feature_channels[self.mask_level], mask_dim, kernel_size=1)
        # Coarse-tier mask projection (only when the two tiers differ). Reads the
        # trunk feature at attn_level; its logits drive per-layer attention + aux.
        if self.two_tier:
            self.coarse_mask_proj = nn.Conv3d(
                self.feature_channels[self.attn_level], mask_dim, kernel_size=1
            )

        # 2. Transformer decoder: project each memory scale to hidden_dim.
        self.mem_proj = nn.ModuleList([
            nn.Conv3d(self.feature_channels[lvl], hidden_dim, kernel_size=1)
            for lvl in self.mem_levels
        ])
        self.level_embed = nn.Parameter(torch.zeros(len(self.mem_levels), hidden_dim))
        self.query_feat = nn.Embedding(self.num_queries, hidden_dim)
        self.query_pos = nn.Embedding(self.num_queries, hidden_dim)
        self.layers = nn.ModuleList([
            _DecoderLayer(hidden_dim, nheads, dim_feedforward) for _ in range(dec_layers)
        ])
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # 3. Mask-embedding head.
        self.mask_embed = _MLP(hidden_dim, hidden_dim, mask_dim, num_layers=3)

        # Deep supervision: primary + up to 3 earlier decoder layers.
        self.aux_count = min(3, dec_layers) if self.deep_supervision else 0

    # -- helpers ----------------------------------------------------------
    def _pixel_decode(self, feats: List[torch.Tensor]):
        """Return (coarse, fine) mask-feature maps.

        ``fine`` is at ``mask_feat_stride`` (final full-res readout); ``coarse``
        is tapped mid-trunk at ``attn_feat_stride`` (per-layer attn + aux). When
        the head is single-tier they are the same tensor.
        """
        d = feats[len(self.feature_channels) - 1]
        coarse = None
        for up, k in zip(self.ups, self.up_ks):
            d = up(d, feats[k])
            if self.two_tier and k == self.attn_level:
                coarse = self.coarse_mask_proj(d)   # (B, mask_dim, d_c, h_c, w_c)
        fine = self.mask_proj(d)                     # (B, mask_dim, d, h, w)
        return (coarse if self.two_tier else fine), fine

    def _mask_logits(self, q: torch.Tensor, mask_features: torch.Tensor) -> torch.Tensor:
        """Per-query mask logits at mask-feature resolution → (B, N, d, h, w)."""
        emb = self.mask_embed(self.decoder_norm(q))            # (B, N, mask_dim)
        return torch.einsum("bqc,bcdhw->bqdhw", emb, mask_features)

    def _aggregate(self, mask_logits: torch.Tensor) -> torch.Tensor:
        """(B, N, ...) query logits → (B, num_classes, ...) class logits.

        Regime A fixed assignment: query k→class k. With queries_per_class>1,
        average the logits of the queries owning each class.
        """
        if self.queries_per_class == 1:
            return mask_logits
        b = mask_logits.shape[0]
        spatial = mask_logits.shape[2:]
        m = mask_logits.view(b, self.num_classes, self.queries_per_class, *spatial)
        return m.mean(dim=2)

    def _attn_mask(self, mask_logits: torch.Tensor, mem_shape) -> torch.Tensor:
        """Bool cross-attention mask (True = blocked) → (B*nheads, N, L_s)."""
        b, nq = mask_logits.shape[:2]
        m = F.interpolate(mask_logits.detach(), size=mem_shape, mode="trilinear", align_corners=False)
        m = m.flatten(2)                                       # (B, N, L_s)
        blocked = m.sigmoid() < 0.5                            # background → block
        # A query that blocks EVERY location would zero out its attention row
        # (→ NaN). Let such queries attend everywhere (Mask2Former's guard).
        blocked = blocked & ~blocked.all(dim=-1, keepdim=True)
        blocked = blocked.unsqueeze(1).expand(b, self.nheads, nq, -1)
        return blocked.reshape(b * self.nheads, nq, -1)

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        x_in: torch.Tensor,
        feats: List[torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if len(feats) != len(self.feature_channels):
            raise ValueError(
                f"head expected {len(self.feature_channels)} features, got {len(feats)}"
            )
        b = feats[0].shape[0]
        out_size = x_in.shape[2:]

        coarse_mf, fine_mf = self._pixel_decode(feats)         # coarse: attn/aux; fine: final
        memory, mem_shapes = [], []
        for proj, lvl in zip(self.mem_proj, self.mem_levels):
            m = proj(feats[lvl])                               # (B, hidden, d_s, h_s, w_s)
            mem_shapes.append(tuple(m.shape[2:]))
            memory.append(m.flatten(2).transpose(1, 2))        # (B, L_s, hidden)

        q = self.query_feat.weight.unsqueeze(0).expand(b, -1, -1)
        q_pos = self.query_pos.weight.unsqueeze(0).expand(b, -1, -1)

        # Per-layer predictions at the COARSE tier — they only feed the next
        # layer's attention mask (and, later, deep-sup aux), never the output.
        preds: List[torch.Tensor] = [self._mask_logits(q, coarse_mf)]
        for i, layer in enumerate(self.layers):
            s = i % len(memory)
            attn_mask = self._attn_mask(preds[-1], mem_shapes[s])
            mem_pos = self.level_embed[s].view(1, 1, -1)
            q = layer(q, q_pos, memory[s], mem_pos, attn_mask)
            preds.append(self._mask_logits(q, coarse_mf))

        # The ONLY full-resolution readout: final queries × the fine mask features.
        main = F.interpolate(
            self._aggregate(self._mask_logits(q, fine_mf)),
            size=out_size, mode="trilinear", align_corners=False,
        )
        if self.deep_supervision and self.training:
            # Finest-first list: primary (full res) + earlier decoder layers at
            # the coarse tier — the DS loss wrapper resamples the label per
            # prediction, so aux stay cheap.
            outs = [main]
            for j in range(1, self.aux_count + 1):
                outs.append(self._aggregate(preds[-1 - j]))
            return outs
        return main
