"""Uniform U-Net segmentation head — first consumer of the pyramid contract.

The head is backbone-agnostic. It consumes ``BackboneInterface.NUM_LEVELS``
feature maps with the shapes laid out in ``seg_model.py`` and produces a
``(B, num_classes, D, H, W)`` logit tensor.

Head ↔ contract interface (a stable, documented public API):

    head(x_in: Tensor, feats: List[Tensor]) -> Tensor | List[Tensor]

    x_in:  (B, 1, D, H, W) — raw input volume. May be unused by some heads;
           kept so a future head can mix raw-input context if it chooses.
    feats: ``NUM_LEVELS`` feature tensors, **finest first**, matching
           ``BackboneInterface.EXPECTED_CHANNELS`` and
           ``BackboneInterface.EXPECTED_STRIDES``.

    Returns a single tensor at eval time (or with deep supervision off), or
    a list of multi-resolution logits **finest first** at train time with
    deep supervision on.

A small registry lets ``head.name`` in config pick the head. A future
deformable / Mask2Former-style head registers under a new name and is
selected purely from config.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Sequence, Type, Union

import torch
import torch.nn as nn


HEAD_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_head(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def deco(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in HEAD_REGISTRY:
            raise ValueError(f"head {name!r} already registered")
        HEAD_REGISTRY[name] = cls
        return cls
    return deco


def build_head(name: str, **kwargs) -> nn.Module:
    if name not in HEAD_REGISTRY:
        raise KeyError(f"unknown head {name!r}. Registered: {sorted(HEAD_REGISTRY)}")
    return HEAD_REGISTRY[name](**kwargs)


def _import_monai():
    from monai.networks.blocks.dynunet_block import UnetOutBlock
    from monai.networks.blocks.unetr_block import UnetrUpBlock
    return UnetOutBlock, UnetrUpBlock


@register_head("unified_seg_head")
class UnifiedSegHead(nn.Module):
    """Convolutional U-Net decoder over a finest-first power-of-two pyramid.

    Default contract (5 levels, finest first):

        feats[0]: (B,  32, D,    H,    W)        stride 1
        feats[1]: (B,  64, D/2,  H/2,  W/2)      stride 2
        feats[2]: (B, 128, D/4,  H/4,  W/4)      stride 4
        feats[3]: (B, 256, D/8,  H/8,  W/8)      stride 8
        feats[4]: (B, 512, D/16, H/16, W/16)     stride 16

    Decoding (with N = number of levels):

        d_{N-1} = feats[N-1]                                  # coarsest
        d_{k}   = UpBlock(d_{k+1}, feats[k])  out_ch=c_k     # k = N-2 .. 0
        logits  = 1×1 conv (d_0)

    No fresh conv on the raw input is used for fine detail — those features
    come from the backbone adapter.

    Deep supervision (training-time): returns
    ``[logits(d_0), aux(d_1), aux(d_2), …]`` (finest first), one per
    ``ds_weight`` in the loss config.
    """

    def __init__(
        self,
        num_classes: int = 118,
        feature_channels: Sequence[int] = (32, 64, 128, 256, 512),
        feature_strides: Sequence[int] = (1, 2, 4, 8, 16),
        decoder_channels: int = 32,  # kept for backward-compat; not used
        norm: str = "instance",
        spatial_dims: int = 3,
        deep_supervision: bool = False,
    ):
        super().__init__()
        UnetOutBlock, UnetrUpBlock = _import_monai()

        self.feature_channels = tuple(feature_channels)
        self.feature_strides = tuple(feature_strides)
        self.deep_supervision = deep_supervision

        if len(self.feature_channels) != len(self.feature_strides):
            raise ValueError("feature_channels and feature_strides length mismatch")
        if len(self.feature_channels) < 2:
            raise ValueError("UnifiedSegHead expects at least 2 pyramid levels")
        for i in range(1, len(self.feature_strides)):
            if self.feature_strides[i] != 2 * self.feature_strides[i - 1]:
                raise ValueError(
                    "feature_strides must be a finest-first ×2 ladder "
                    f"(got {self.feature_strides})"
                )

        # ups[k] (k = 0..N-2) takes d_{k+1} + skip feats[k] -> d_k.
        # Channel transition: c_{k+1} -> c_k (UnetrUpBlock convention).
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
            for k in range(len(self.feature_channels) - 1)
        ])

        # Main 1×1 conv: d_0 (stride 1) -> num_classes logits.
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feature_channels[0],
            out_channels=num_classes,
        )

        # Deep-supervision aux heads: one per intermediate decoder stage
        # d_1, d_2, ..., d_{ds_aux} (finest aux first). We stop short of
        # the coarsest stage (stride > 8 tends to underweight segmentation).
        # Number of aux heads = min(3, N-2) so a 5-level pyramid emits at
        # strides {1, 2, 4, 8} matching ds_weights = [1, 0.5, 0.25, 0.125].
        if deep_supervision:
            self.aux_count = min(3, len(self.feature_channels) - 2)
            self.aux_heads = nn.ModuleList([
                UnetOutBlock(
                    spatial_dims=spatial_dims,
                    in_channels=self.feature_channels[k],
                    out_channels=num_classes,
                )
                for k in range(1, 1 + self.aux_count)
            ])
        else:
            self.aux_count = 0
            self.aux_heads = nn.ModuleList()

    def forward(
        self,
        x_in: torch.Tensor,
        feats: List[torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if len(feats) != len(self.feature_channels):
            raise ValueError(
                f"head expected {len(self.feature_channels)} features, got {len(feats)}"
            )
        n = len(self.feature_channels)
        # decoder[k] = d_k at stride feature_strides[k]; finest first.
        decoder: List[torch.Tensor] = [None] * n  # type: ignore[list-item]
        decoder[n - 1] = feats[n - 1]
        for k in range(n - 2, -1, -1):
            decoder[k] = self.ups[k](decoder[k + 1], feats[k])

        main = self.out(decoder[0])

        if self.deep_supervision and self.training:
            outs: List[torch.Tensor] = [main]
            for ai in range(self.aux_count):
                outs.append(self.aux_heads[ai](decoder[ai + 1]))
            return outs
        return main
