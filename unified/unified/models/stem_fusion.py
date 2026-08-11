"""Arm S — the matched-stem wrapper.

Arm N compares encoders *as delivered*: a hierarchical CNN hands the decoder five
genuinely different tensors, a columnar ViT hands it five resampled copies of one.
That gap is real, but it conflates two things — how good the pretrained semantics
are, and whether the architecture happens to route fine detail to a decoder.

This wrapper separates them. It bolts the **same** ``SpatialPriorModule3D`` onto
*every* backbone and adds it into contract levels 0–3::

    fused_k = ReLU(GroupNorm(inner_k + spm_k))     k = 0..3   (strides 1, 2, 4, 8)
    fused_4 = inner_4                              (stride 16, untouched)

For a columnar ViT that supplies the stride-1 wire it never had. For a CNN it is
a fresh stem added alongside the pretrained one — because a CNN's own stride-1
level *is* a two-conv stem on raw voxels, so the honest contrast is
pretrained-stem vs fresh-stem, not stem vs no-stem.

The point is not the level Arm S reaches, it is the **delta**::

    compensable(m) = gap_N(m) - gap_S(m)    missing fine detail — a stem fixes it
    irreducible(m) = gap_S(m)               missing pretrained mid-level features

Expect compensable ~ 0 for CNNs and large for columnar ViTs.

Two properties make the arm a controlled comparison, and both come from this
being *one wrapper applied uniformly* rather than a per-backbone edit:

* every backbone gets a bit-identical stem, so the added capacity is a constant
  (348.6 k for the SPM at ``inplanes=16``, plus 960 GroupNorm affine parameters);
* no backbone module is touched, so Arm N and Arm S differ by exactly this one
  wrapper and nothing else.

``freeze_backbone=True`` still freezes only the pretrained encoder: the inner
backbone's own ``freeze_encoder`` runs unchanged (keeping its neck trainable) and
the stem + fusion norms here stay learnable, matching how every other adapter
behaves under a frozen probe.
"""
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn

from .seg_model import BackboneInterface
from .backbones._neck import SpatialPriorModule3D, _group_count


# Inner pyramid modes that already run a raw-input conv branch. Wrapping one of
# these would stack two stems and silently double the added capacity, which
# breaks the "constant across backbones" property the arm depends on.
_ALREADY_HAS_STEM = {"spm", "vit_adapter"}


class _StemFusionAdapter(nn.Module):
    """The trainable half: one shared stem plus per-level fusion norms."""

    def __init__(self, contract_channels, in_channels: int = 1, inplanes: int = 16):
        super().__init__()
        fine = tuple(contract_channels[:4])
        self.spm = SpatialPriorModule3D(
            in_channels=in_channels, inplanes=inplanes, out_channels=fine,
        )
        # Post-sum normalisation. The inner branch arrives activated (ChannelNeck
        # ends in GroupNorm+ReLU) while the SPM projections are bare convs, so the
        # sum needs re-normalising before the head sees it.
        self.norms = nn.ModuleList([nn.GroupNorm(_group_count(c), c) for c in fine])
        self.act = nn.ReLU(inplace=True)

    def forward(self, inner_feats: List[torch.Tensor], x: torch.Tensor) -> List[torch.Tensor]:
        spm_feats = self.spm(x)
        fused = [
            self.act(norm(inner + spm))
            for inner, spm, norm in zip(inner_feats[:4], spm_feats, self.norms)
        ]
        return [*fused, inner_feats[4]]


class StemFusionBackbone(BackboneInterface):
    """Wrap any contract-compliant backbone and fuse a shared conv stem into it.

    Parameters
    ----------
    inner : BackboneInterface
        An already-constructed backbone. Must implement the
        ``encoder_forward`` / ``adapter_forward`` split so the frozen path keeps
        the pretrained encoder under ``no_grad``.
    stem_inplanes : int, default 16
        SPM width. Held constant across backbones for the headline Arm S; exposed
        so a later budget-matched variant can tune it per encoder.
    """

    def __init__(self, inner: BackboneInterface, stem_inplanes: int = 16,
                 in_channels: int = 1):
        super().__init__()
        for hook in ("encoder_forward", "adapter_forward"):
            if getattr(inner, hook, None) is None:
                raise TypeError(
                    f"{type(inner).__name__} has no {hook}(); Arm S needs the "
                    "encoder/adapter split so the frozen probe keeps the "
                    "pretrained encoder under no_grad."
                )
        mode = getattr(inner, "pyramid_mode", None)
        if mode in _ALREADY_HAS_STEM:
            raise ValueError(
                f"pyramid_mode={mode!r} already runs a raw-input conv branch; "
                "wrapping it would stack two stems and break the constant-capacity "
                "property Arm S depends on. Use the probe-honest inner mode "
                "(e.g. 'upsample') and let this wrapper supply the stem."
            )
        self.inner = inner
        self.adapter = _StemFusionAdapter(
            contract_channels=self.EXPECTED_CHANNELS,
            in_channels=in_channels,
            inplanes=stem_inplanes,
        )

    def encoder_forward(self, x: torch.Tensor):
        # Prepend the raw volume so the trainable stem can reach it from
        # adapter_forward. SegModel detaches these before the adapter runs, which
        # is a no-op for an input leaf.
        return [x, *self.inner.encoder_forward(x)]

    def adapter_forward(self, native, input_shape) -> List[torch.Tensor]:
        x_in, *inner_native = native
        inner_feats = self.inner.adapter_forward(inner_native, input_shape)
        return self.adapter(inner_feats, x_in)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])

    def freeze_encoder(self) -> None:
        """Freeze the pretrained encoder only.

        Delegates to the inner backbone (which keeps its own neck learnable, and
        for dino3d's vit_adapter mode applies its bespoke rule), then leaves this
        wrapper's stem and fusion norms trainable. The base-class rule would
        wrongly freeze the inner neck, since that neck is not reachable through
        *this* module's ``self.adapter``.
        """
        self.inner.freeze_encoder()

    def trainable_modules(self):
        """This wrapper's stem *and* whatever the inner backbone reports.

        Without the inner entry a frozen Arm S run would leave the inner neck in
        eval mode while the matching Arm N run trains it — an arm difference with
        nothing to do with the stem.
        """
        return [self.adapter, *self.inner.trainable_modules()]

    def stem_param_count(self) -> int:
        """Parameters this wrapper adds. Constant across backbones by design."""
        return sum(p.numel() for p in self.adapter.parameters())
