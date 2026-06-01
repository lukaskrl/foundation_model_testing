"""Per-level adapter primitives used by backbone-specific adapters.

Two building blocks:

* ``ChannelNeck``  — 1×1×1 conv + GroupNorm + ReLU. Pure channel adaptation.
* ``DownsampleNeck`` — 3×3×3 stride-2 conv + GroupNorm + ReLU. One factor-2
  spatial downsample (and channel projection) when a backbone needs to
  synthesize a coarser contract level than its native pyramid offers.

We never *upsample* a pretrained feature: if a level is finer than any
native feature provides, the backbone-specific adapter generates it from a
fresh conv branch on the raw input (Swin conv stem, ViT SpatialPriorModule).
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _group_count(ch: int, max_groups: int = 8) -> int:
    g = min(max_groups, ch)
    while ch % g != 0:
        g -= 1
    return g


class ChannelNeck(nn.Module):
    """1×1×1 conv + GroupNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(x)))


class DownsampleNeck(nn.Module):
    """2×2×2 stride-2 conv + GroupNorm + ReLU. One factor-2 downsample step.

    Kernel size 2 keeps param count manageable when the synthesized level
    needs a channel expansion (256 → 512 is 1.05 M params here vs 3.5 M for
    a 3×3×3 conv). The adapter is meant to be a lean lift, not a third
    encoder stage.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class StrideAdapter(nn.Module):
    """Legacy spatial resize helper (kept so stub backbones still import).

    Not used by the new pyramid contract — the active adapters avoid
    interpolation entirely. Stubs (biomedparse, ctclip) that referenced this
    in older code paths import it here so the package still imports cleanly.
    """

    def __init__(self, mode: str = "trilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, feat: torch.Tensor, target_shape):
        import torch.nn.functional as F
        if tuple(feat.shape[2:]) == tuple(target_shape):
            return feat
        return F.interpolate(
            feat,
            size=tuple(target_shape),
            mode=self.mode,
            align_corners=False if self.mode == "trilinear" else None,
        )


class PyramidNeck(nn.Module):
    """Bundle of per-level ``ChannelNeck`` plus optional coarser-level synthesis.

    Parameters
    ----------
    native_channels : sequence of ints
        Channel count of each native encoder feature, finest first.
    contract_channels : sequence of ints
        Target channel count at each contract level, finest first.
    extra_down : int, default 0
        Number of extra coarser levels to synthesize beyond the deepest
        native feature, via successive ``DownsampleNeck`` stages. The final
        output channels at each synthesized level come from
        ``contract_channels[len(native_channels) + i]``.

    The total output length is ``len(native_channels) + extra_down`` and must
    equal ``len(contract_channels)``. A native feature must be passed in for
    each native level; the synthesized levels are produced from the deepest
    *adapted* feature.

    Spatial strides are not checked here — the adapter is responsible for
    arranging native features at the target strides. The neck only does
    1×1 channel projection plus, optionally, a strided downsample chain.
    """

    def __init__(
        self,
        native_channels,
        contract_channels=(32, 64, 128, 256, 512),
        extra_down: int = 0,
    ):
        super().__init__()
        native_channels = tuple(native_channels)
        contract_channels = tuple(contract_channels)
        if len(native_channels) + extra_down != len(contract_channels):
            raise ValueError(
                f"native ({len(native_channels)}) + extra_down ({extra_down})"
                f" != contract ({len(contract_channels)})"
            )
        self.ch_necks = nn.ModuleList(
            [ChannelNeck(ic, oc) for ic, oc in zip(native_channels, contract_channels)]
        )
        self.downs = nn.ModuleList()
        prev_ch = contract_channels[len(native_channels) - 1] if native_channels else None
        for i in range(extra_down):
            target_idx = len(native_channels) + i
            self.downs.append(DownsampleNeck(prev_ch, contract_channels[target_idx]))
            prev_ch = contract_channels[target_idx]

    def forward(self, native_feats):
        if len(native_feats) != len(self.ch_necks):
            raise ValueError(
                f"got {len(native_feats)} native features, expected {len(self.ch_necks)}"
            )
        out = [neck(f) for neck, f in zip(self.ch_necks, native_feats)]
        cur = out[-1]
        for down in self.downs:
            cur = down(cur)
            out.append(cur)
        return out
