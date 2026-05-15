"""Uniform UNETR-style segmentation head.

Takes 4 multi-scale 3D feature maps with fixed channels {64,128,256,512} and strides
{4,8,16,32} and produces a (B, num_classes, D, H, W) logit tensor.
"""
from __future__ import annotations
from typing import List, Sequence

import torch
import torch.nn as nn

# MONAI is required at run time. We import lazily so the module can be inspected
# without MONAI installed.
def _import_monai():
    from monai.networks.blocks.dynunet_block import UnetOutBlock
    from monai.networks.blocks.unetr_block import (
        UnetrBasicBlock,
        UnetrUpBlock,
    )
    return UnetOutBlock, UnetrBasicBlock, UnetrUpBlock


class UnifiedSegHead(nn.Module):
    """UNETR-style decoder.

    Pyramid (input → output):

        x_in  (B,1,D,H,W)
        feat0 (B,64, D/4, ...)
        feat1 (B,128,D/8, ...)
        feat2 (B,256,D/16,...)
        feat3 (B,512,D/32,...)

        enc0 = UnetrBasicBlock(x_in) -> (B, dc, D, H, W)
        dec3 = UnetrUpBlock(feat3, feat2) -> (B, 256, D/16, ...)
        dec2 = UnetrUpBlock(dec3,  feat1) -> (B, 128, D/8,  ...)
        dec1 = UnetrUpBlock(dec2,  feat0) -> (B, 64,  D/4,  ...)
        dec0 = UnetrUpBlock(dec1,  enc0_down) -> (B, dc,  D,    ...)
        out  = 1x1x1 conv -> (B, num_classes, D, H, W)

    where ``dc`` = ``decoder_channels`` (default 32).
    """

    def __init__(
        self,
        num_classes: int = 118,
        feature_channels: Sequence[int] = (64, 128, 256, 512),
        feature_strides: Sequence[int] = (4, 8, 16, 32),
        decoder_channels: int = 32,
        norm: str = "instance",
        spatial_dims: int = 3,
    ):
        super().__init__()
        if len(feature_channels) != 4 or len(feature_strides) != 4:
            raise ValueError("UnifiedSegHead expects exactly 4 pyramid levels")
        UnetOutBlock, UnetrBasicBlock, UnetrUpBlock = _import_monai()

        self.feature_channels = tuple(feature_channels)
        self.feature_strides = tuple(feature_strides)
        c0, c1, c2, c3 = self.feature_channels
        dc = decoder_channels

        # Encoder side: 1 block at full resolution + 1 block at stride 4
        # (matching feat0). enc0 supplies the deepest skip connection at full res.
        self.enc0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=dc,
            kernel_size=3,
            stride=1,
            norm_name=norm,
            res_block=True,
        )

        # Decoder: progressive upsampling. UnetrUpBlock(in, skip_in -> out)
        # internally does ConvTranspose to upsample `in`, concat with `skip`, then
        # two conv blocks.
        self.up3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=c3,
            out_channels=c2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm,
            res_block=True,
        )  # input: feat3 (stride 32) + feat2 (stride 16) -> stride 16
        self.up2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=c2,
            out_channels=c1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm,
            res_block=True,
        )  # stride 16 -> stride 8
        self.up1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=c1,
            out_channels=c0,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm,
            res_block=True,
        )  # stride 8 -> stride 4
        # To bridge from stride 4 back to full resolution we need ONE more
        # upsample step of factor 4 (= two 2× steps). We do this in two blocks
        # so the architecture is purely powers-of-two compatible.
        self.up0a = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=c0,
            out_channels=dc * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm,
            res_block=True,
        )  # stride 4 -> stride 2
        # For up0a's skip we synthesize a stride-2 feature from x_in via a strided conv
        # of `enc0`. That keeps the head input-agnostic about which backbones expose
        # stride-2 features (only SegResNet does; most don't). Must produce `dc*2`
        # channels because up0a expects `out_channels` (= dc*2) on its skip input.
        self.enc0_down2 = nn.Sequential(
            nn.Conv3d(dc, dc * 2, kernel_size=2, stride=2, bias=False),
            _make_norm(norm, dc * 2, spatial_dims=spatial_dims),
            nn.ReLU(inplace=True),
        )

        self.up0b = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=dc * 2,
            out_channels=dc,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm,
            res_block=True,
        )  # stride 2 -> stride 1, with enc0 skip
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=dc,
            out_channels=num_classes,
        )

    def forward(
        self,
        x_in: torch.Tensor,
        feats: List[torch.Tensor],
    ) -> torch.Tensor:
        if len(feats) != 4:
            raise ValueError(f"expected 4 features, got {len(feats)}")
        f0, f1, f2, f3 = feats

        enc0 = self.enc0(x_in)                       # (B, dc,   D, H, W)
        enc0_down2 = self.enc0_down2(enc0)           # (B, dc,   D/2,...)

        d3 = self.up3(f3, f2)                        # (B, c2,   D/16,...)
        d2 = self.up2(d3, f1)                        # (B, c1,   D/8, ...)
        d1 = self.up1(d2, f0)                        # (B, c0,   D/4, ...)
        d0a = self.up0a(d1, enc0_down2)              # (B, dc*2, D/2, ...)
        d0b = self.up0b(d0a, enc0)                   # (B, dc,   D,   ...)

        return self.out(d0b)                         # (B, num_classes, D, H, W)


def _make_norm(name: str, channels: int, *, spatial_dims: int = 3) -> nn.Module:
    name = name.lower()
    if name == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if name == "batch":
        return nn.BatchNorm3d(channels)
    if name == "group":
        return nn.GroupNorm(8, channels)
    raise ValueError(f"unknown norm {name!r}")
