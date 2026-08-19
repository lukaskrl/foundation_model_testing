"""Per-level adapter primitives used by backbone-specific adapters.

Building blocks:

* ``ChannelNeck``  — 1×1×1 conv + GroupNorm + ReLU. Pure channel adaptation.
* ``DownsampleNeck`` — 3×3×3 stride-2 conv + GroupNorm + ReLU. One factor-2
  spatial downsample (and channel projection) when a backbone needs to
  synthesize a coarser contract level than its native pyramid offers.
* ``UpsampleNeck`` — channel-project + trilinearly resample a *single* encoder
  feature to every contract level. The probe-honest alternative to the ViT
  ``SpatialPriorModule`` path: the fine levels are upsampled copies of the
  pretrained bottleneck, with NO fresh conv branch on the raw input, so a
  frozen-encoder probe measures the pretrained representation itself rather
  than an independent raw-pixel network. The only trainable parameters are
  per-level 1×1 projections, mirroring ``PyramidNeck``.

For the multi-scale CNN adapters we still never upsample a pretrained feature
(``PyramidNeck`` is channel-projection plus optional strided downsample). The
``UpsampleNeck`` exists specifically for single-scale ViT / bottleneck
encoders, which genuinely have no finer-than-stride-16 features to offer.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ConvStem(nn.Module):
    """Two 3×3×3 convs on the raw input volume → the stride-1 contract level.

    For encoders whose native pyramid starts at stride 2 or coarser (the Swin
    family, an inflated 2D ResNet, a per-slice 2D backbone) the finest contract
    level cannot come from pretrained features, and we never upsample them. A
    small fresh stem fills exactly that one level while strides 2–16 still come
    from the encoder, which is why ``native``-mode backbones using it remain
    probe-comparable — unlike ``spm``, which supplies four of the five levels.

    ``voco.py`` keeps a structurally identical stem inline rather than importing
    this one: its parameter names appear in already-trained ``best.pt`` files, and
    renaming them would break resuming or re-evaluating those runs. Keep the two
    in sync — same layers, same parameter count (28,640 at the defaults).
    """

    def __init__(self, out_ch: int, in_ch: int = 1, hidden_ch: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_ch), hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


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


class UpsampleNeck(nn.Module):
    """Synthesize the full contract pyramid from a single encoder feature.

    Channel-projects the encoder's single (stride-16) feature map with a
    per-level ``ChannelNeck`` and trilinearly resamples it to each contract
    level's target spatial size. There is NO raw-input branch: every output
    level is a deterministic function of the (optionally frozen) encoder
    bottleneck, so the finer levels carry no more information than the coarse
    one. That is the intended behaviour for a representation probe — it does
    not let a fresh conv stem compensate for a ViT's lack of native
    multi-scale detail (cf. ``SpatialPriorModule3D``).

    Projection runs on the encoder's native grid (cheap) and the activated,
    low-channel result is resampled to the level target.

    Parameters
    ----------
    in_channels : int
        Channel count of the encoder bottleneck fed to ``forward``.
    contract_channels, contract_strides : sequences, finest first
        Target channels / spatial strides of each contract level.
    """

    def __init__(
        self,
        in_channels: int,
        contract_channels=(32, 64, 128, 256, 512),
        contract_strides=(1, 2, 4, 8, 16),
    ):
        super().__init__()
        if len(contract_channels) != len(contract_strides):
            raise ValueError("contract_channels and contract_strides length mismatch")
        self.contract_strides = tuple(int(s) for s in contract_strides)
        self.projections = nn.ModuleList(
            [ChannelNeck(in_channels, oc) for oc in contract_channels]
        )

    def forward(self, bottleneck: torch.Tensor, input_shape):
        D, H, W = (int(s) for s in input_shape)
        feats = []
        for stride, proj in zip(self.contract_strides, self.projections):
            target = (D // stride, H // stride, W // stride)
            f = proj(bottleneck)
            if tuple(f.shape[2:]) != target:
                f = F.interpolate(f, size=target, mode="trilinear", align_corners=False)
            feats.append(f)
        return feats


class MultiCanvasNeck(nn.Module):
    """Assemble the contract pyramid from several encoder passes at *different
    input canvases*.

    The probe-honest middle ground between ``UpsampleNeck`` and
    ``SpatialPriorModule3D``. ``UpsampleNeck`` gives every level a resampled
    copy of one feature map, so the fine levels carry no information the coarse
    one lacks. ``SpatialPriorModule3D`` fixes that by adding a fresh conv branch
    on raw voxels — which means a frozen-encoder probe partly measures that
    branch instead of the pretrained representation. Here the extra scales come
    from *re-running the pretrained encoder* on a differently-sized input, so
    the pyramid carries genuine multi-scale information while every level stays
    a pure function of the pretrained weights.

    ``level_sources[i]`` indexes which encoder output feeds contract level
    ``i``. The convention used by the CT-CLIP adapter is: canvases sorted
    ascending, smallest canvas → coarsest level (few tokens, each covering a
    large physical extent = global context), largest canvas → finest levels
    (many tokens, each covering a small extent = local detail). Levels finer
    than the largest canvas can serve are resampled from it.

    Only per-level 1×1 projections are trainable, identical in count to
    ``UpsampleNeck`` (one ``ChannelNeck`` per level), so the two modes are
    parameter-matched and differ *only* in where each level's features come
    from. That makes them a clean A/B pair.
    """

    def __init__(
        self,
        in_channels: int,
        level_sources,
        contract_channels=(32, 64, 128, 256, 512),
        contract_strides=(1, 2, 4, 8, 16),
    ):
        super().__init__()
        if not (len(contract_channels) == len(contract_strides) == len(level_sources)):
            raise ValueError(
                "contract_channels, contract_strides and level_sources must all "
                f"have the same length; got {len(contract_channels)}, "
                f"{len(contract_strides)}, {len(level_sources)}"
            )
        self.contract_strides = tuple(int(s) for s in contract_strides)
        self.level_sources = tuple(int(s) for s in level_sources)
        self.projections = nn.ModuleList(
            [ChannelNeck(in_channels, oc) for oc in contract_channels]
        )

    def forward(self, sources, input_shape):
        n = len(sources)
        if max(self.level_sources) >= n:
            raise ValueError(
                f"level_sources {self.level_sources} references encoder output "
                f"{max(self.level_sources)} but only {n} were provided"
            )
        D, H, W = (int(s) for s in input_shape)
        feats = []
        for stride, src, proj in zip(
            self.contract_strides, self.level_sources, self.projections
        ):
            target = (D // stride, H // stride, W // stride)
            f = proj(sources[src])
            if tuple(f.shape[2:]) != target:
                f = F.interpolate(f, size=target, mode="trilinear", align_corners=False)
            feats.append(f)
        return feats


class LayerTapNeck(MultiCanvasNeck):
    """Assemble the contract pyramid from several *depths of one encoder pass*.

    Identical machinery to ``MultiCanvasNeck`` — per-level 1×1 projection of a
    chosen source, then resample — but the sources are reads of the residual
    stream at different transformer layers rather than passes at different input
    canvases. One forward pass instead of several, which is why it is the cheap
    option for a columnar (non-hierarchical) ViT.

    The semantics of the source set differ, and it matters for what a result
    means. Multi-canvas sources differ in *scale*: each pass has its own token
    grid. Layer taps all share one token grid and one width; they differ in how
    much context has been mixed in. So this neck buys the decoder per-level
    variation in abstraction, NOT in resolution — the finest level is still a
    resampled copy of a stride-16 grid, and nothing here relaxes the encoder's
    token-scale bound.

    A second consequence of reading a residual stream: the taps are nested, not
    independent. A late tap still carries what an early one contributed, so the
    fine levels get less-abstracted access to information that largely survives
    downstream rather than information that would otherwise be gone. Expect a
    smaller effect than a CNN pyramid's skips, where downsampling genuinely
    destroys the fine detail.

    ``level_sources`` follows the same convention as the parent (index into the
    source list, finest level first). For the earliest-tap-to-finest-level
    ordering used by the CT-CLIP adapter — the convention plain-ViT dense
    prediction settled on — pass ascending taps and an identity mapping.

    Trainable parameters are one ``ChannelNeck`` per level, exactly as in
    ``UpsampleNeck``, so ``upsample`` and ``layerwise`` are parameter-matched
    and differ only in which tensor each level reads.
    """


class SpatialPriorModule3D(nn.Module):
    """Conv stem on raw input that emits features at strides {1, 2, 4, 8}.

    Adapted from 3DINO's ``SpatialPriorModule`` (with the first conv changed
    from stride 2 to stride 1 so we keep a stride-1 level, and ``GroupNorm``
    in place of ``SyncBatchNorm``). Parameter count is dominated by the two
    deepest stages — keep ``inplanes`` small.

    Lives here rather than in a backbone module because two different consumers
    need it: the per-backbone ``pyramid_mode='spm'`` adapters (where it
    *replaces* the fine levels), and the generic ``StemFusionBackbone`` wrapper
    (where it is *added* to whatever the backbone already produces). Keeping one
    definition means every Arm-S run gets a bit-identical stem, which is the
    property that makes the arm a controlled comparison.
    """

    def __init__(
        self,
        in_channels: int = 1,
        inplanes: int = 16,
        out_channels=(32, 64, 128, 256),
    ):
        super().__init__()
        c0, c1, c2, c3 = out_channels

        def gn(ch):
            return nn.GroupNorm(_group_count(ch), ch)

        # stride 1 stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            gn(inplanes), nn.ReLU(inplace=True),
            nn.Conv3d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            gn(inplanes), nn.ReLU(inplace=True),
            nn.Conv3d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            gn(inplanes), nn.ReLU(inplace=True),
        )
        # stride 1 -> 2
        self.down1 = nn.Sequential(
            nn.Conv3d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            gn(2 * inplanes), nn.ReLU(inplace=True),
        )
        # stride 2 -> 4
        self.down2 = nn.Sequential(
            nn.Conv3d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            gn(4 * inplanes), nn.ReLU(inplace=True),
        )
        # stride 4 -> 8
        self.down3 = nn.Sequential(
            nn.Conv3d(4 * inplanes, 8 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            gn(8 * inplanes), nn.ReLU(inplace=True),
        )

        self.proj0 = nn.Conv3d(inplanes,     c0, kernel_size=1, bias=False)
        self.proj1 = nn.Conv3d(2 * inplanes, c1, kernel_size=1, bias=False)
        self.proj2 = nn.Conv3d(4 * inplanes, c2, kernel_size=1, bias=False)
        self.proj3 = nn.Conv3d(8 * inplanes, c3, kernel_size=1, bias=False)

    def forward(self, x):
        c0 = self.stem(x)
        c1 = self.down1(c0)
        c2 = self.down2(c1)
        c3 = self.down3(c2)
        return [self.proj0(c0), self.proj1(c1), self.proj2(c2), self.proj3(c3)]
