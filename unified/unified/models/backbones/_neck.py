"""Small per-level adapters that bring native encoder features onto the
contract pyramid (4 levels × {64,128,256,512} channels × strides {4,8,16,32}).

Two operations: a 1×1×1 conv for channel adaptation and an optional spatial
resize (stride conv to down-sample, trilinear interpolate to up-sample).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelNeck(nn.Module):
    """1×1×1 conv + GroupNorm + ReLU. Cheap and uniform across backbones."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        groups = min(8, out_ch)
        while out_ch % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(x)))


class StrideAdapter(nn.Module):
    """Re-sample a 3D feature map to a target stride relative to a reference shape.

    The reference shape ``(D_target, H_target, W_target)`` is computed at runtime
    from the input image shape passed via ``forward``. If the feature is already
    at the target stride, this is a no-op.
    """

    def __init__(self, mode: str = "trilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, feat: torch.Tensor, target_shape):
        if tuple(feat.shape[2:]) == tuple(target_shape):
            return feat
        return F.interpolate(
            feat,
            size=tuple(target_shape),
            mode=self.mode,
            align_corners=False if self.mode == "trilinear" else None,
        )


class PyramidNeck(nn.Module):
    """Adapter that takes a list of native-encoder features and emits the
    contract pyramid.

    Parameters
    ----------
    in_channels : tuple of 4 ints
        Channels of the 4 native features picked from the encoder.
    out_channels : tuple of 4 ints, default (64, 128, 256, 512)
    strides : tuple of 4 ints, default (4, 8, 16, 32)
        Pyramid strides on the contract side.
    """

    def __init__(
        self,
        in_channels,
        out_channels=(64, 128, 256, 512),
        strides=(4, 8, 16, 32),
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("PyramidNeck expects 4 input channel counts")
        self.strides = tuple(strides)
        self.ch_necks = nn.ModuleList(
            [ChannelNeck(ic, oc) for ic, oc in zip(in_channels, out_channels)]
        )
        self.resize = StrideAdapter()

    def forward(self, feats, input_shape):
        """feats: List[Tensor], input_shape: (D, H, W). Returns 4 contract tensors."""
        D, H, W = input_shape
        out = []
        for f, neck, s in zip(feats, self.ch_necks, self.strides):
            target = (D // s, H // s, W // s)
            f = neck(f)
            f = self.resize(f, target)
            out.append(f)
        return out
