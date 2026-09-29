"""I1 / I2 — the pyramid-free interface (``docs/HEAD_DESIGN.md`` §7).

Arms N/S/B/W force every encoder onto the fixed 5-level contract, so any level an
encoder does not natively produce is MANUFACTURED — by a fresh stem, a
``DownsampleNeck``, or resampling. Encoder identity is therefore perfectly
correlated with which manufacturing strategy was used, and no between-encoder
number in those arms is a clean comparison.

I1 and I2 drop the fixed level count instead of filling it.

**I1 — encoder-only.** The adapter may not invent spatial information:
  * no raw-voxel path, no synthesized deepest level, no upsampling of features;
  * exactly one feature per DISTINCT native grid (a 5-grid CNN gives 5, a columnar
    ViT gives 1);
  * taps sharing a grid go through ONE shared 1x1 projection and are summed, so
    extra taps buy abstraction, never resolution and never parameters;
  * every projection lands on one common width ``W``;
  * the decoder climbs to stride 1 with a WEIGHT-SHARED upsample block applied
    ``k`` times, ``k`` = the stride doublings between the coarsest native grid
    and stride 1.

**I2 — I1 plus one identical stem.** Every encoder also gets the same fresh
``ConvStem`` at stride 1. Nothing else changes, and the cost is a constant
+37,344 parameters (28,640 stem + 8,704 projection) for every encoder.

**Upsampled skip (optional, on I1 or I2).** ``up_skip`` delivers the coarsest
projected feature a second time, upsampled to ``up_stride`` (default 2), as a skip:
``"trilinear"`` by interpolation (the control), ``"guided"`` by the frozen,
pretrained guided upsampler of ``docs/UPSAMPLER_PLAN.md``, which mixes the same
vectors with CT-guided attention weights. Both add ZERO trainable parameters, so
``guided - trilinear`` isolates the upsampler. The guided upsampler needs the CT in
Hounsfield units, which rides along as input channel 1 (``data.guide_hu: true``);
the trilinear control takes the same two-channel input and ignores channel 1, so the
data pipeline is identical for the pair. Cell centers assume the evenly laid
layout, which is exact for dino3d (patch 16 on a multiple-of-16 input).

``I2 - I1`` is then the **compensable resolution deficit**, measured in the same
units with the same parameters for a CNN and for a ViT. Under the 5-level contract
this is unmeasurable, because each encoder gets a different fine path by
construction.

Because the upsample block and the output conv are each instantiated ONCE and
reused at every resolution, the decoder's parameter count is independent of how
many grids an encoder exposes — a 5-grid CNN and a 1-grid ViT get byte-identical
decoders. ``scripts/test_interface.py`` asserts exactly that; it is the property
the whole comparison rests on.

Scope: strides must be isotropic and powers of two. Per ``docs/NATIVE_GRIDS.md``
that holds for ctfm, vista3d, the SuPreM pair, voco, suprem_swinunetr, merlin and
dino3d, and fails for biomedparse (anisotropic), ctclip (stride 9.6) and
sam_med3d (stride 12). Those raise rather than being silently resampled.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones._neck import ChannelNeck, ConvStem, _group_count
from .head import register_head
from .seg_model import BackboneInterface

STEM_HIDDEN = 32          # ConvStem width before projection; 28,640 params at this value


def _exact_pow2_stride(inp: int, out: int, axis: str) -> int:
    if out <= 0 or inp % out != 0:
        raise ValueError(
            f"native grid does not divide the input on axis {axis} ({inp}/{out}); "
            "this encoder is out of scope for I1/I2 (see docs/NATIVE_GRIDS.md)"
        )
    s = inp // out
    if s & (s - 1):
        raise ValueError(
            f"native stride {s} on axis {axis} is not a power of two; "
            "this encoder is out of scope for I1/I2 (see docs/NATIVE_GRIDS.md)"
        )
    return s


class InterfaceBackbone(BackboneInterface):
    """Wraps any backbone that implements ``encoder_forward`` and exposes its
    NATIVE grids at one common width, coarsest-first.

    Unlike every other backbone here this does NOT satisfy the 5-level contract —
    that is the point — so ``NUM_LEVELS``/``EXPECTED_*`` are cleared and
    ``assert_contract`` is never applicable. Pair it with ``interface_seg_head``.

    The raw input volume rides along in ``encoder_forward``'s output when (and only
    when) the I2 stem is enabled, because ``SegModel._forward_frozen`` hands
    ``adapter_forward`` the native list and a shape, never the input tensor.
    ``voco.py`` does the same thing for the same reason.
    """

    EXPECTED_STRIDES = ()
    EXPECTED_CHANNELS = ()
    NUM_LEVELS = 0

    def __init__(self, inner: nn.Module, patch_size: Sequence[int],
                 width: int = 256, stem: bool = False,
                 stem_hidden: int = STEM_HIDDEN, up_skip: str = None,
                 up_ckpt: str = None, up_stride: int = 2):
        super().__init__()
        if not hasattr(inner, "encoder_forward"):
            raise TypeError(
                f"{type(inner).__name__} has no encoder_forward/adapter_forward split, "
                "so its native features cannot be read (see docs/NATIVE_GRIDS.md)"
            )
        self.inner = inner
        self.width = int(width)
        self.use_stem = bool(stem)

        raw_idx, taps = self._probe(tuple(patch_size))
        self._raw_indices = raw_idx
        # stride -> positions within the FILTERED tap list
        groups: "OrderedDict[int, List[int]]" = OrderedDict()
        widths: Dict[int, int] = {}
        for pos, (stride, ch) in enumerate(taps):
            groups.setdefault(stride, []).append(pos)
            if stride in widths and widths[stride] != ch:
                raise ValueError(
                    f"taps sharing stride {stride} have different widths "
                    f"({widths[stride]} vs {ch}); a single shared 1x1 projection "
                    "is not defined for them"
                )
            widths[stride] = ch
        self._groups = groups

        adapter = nn.Module()
        adapter.proj = nn.ModuleDict(
            {f"s{s}": ChannelNeck(widths[s], self.width) for s in groups}
        )
        if self.use_stem:
            adapter.stem = ConvStem(out_ch=stem_hidden)
            adapter.stem_proj = ChannelNeck(stem_hidden, self.width)
        self.adapter = adapter

        # The inner backbone's own contract adapter is never executed here. Drop it
        # so parameter counts describe what actually runs.
        if hasattr(inner, "adapter") and inner.adapter is not None:
            inner.adapter = nn.Identity()

        self.up_skip = None if up_skip in (None, "none") else str(up_skip)
        self.up_stride = int(up_stride)
        self._tabs = {}
        if self.up_skip is not None:
            if self.up_skip not in ("trilinear", "guided"):
                raise ValueError(f"up_skip must be 'trilinear' or 'guided', got {up_skip!r}")
            if self.up_stride >= max(groups) or self.up_stride & (self.up_stride - 1):
                raise ValueError(f"up_stride {up_stride} must be a power of two below "
                                 f"the coarsest native stride {max(groups)}")
        if self.up_skip == "guided":
            from ..upsampler.module import GuidedUpsampler3D
            if not up_ckpt:
                raise ValueError("up_skip='guided' needs up_ckpt (a trained upsampler)")
            ck = torch.load(up_ckpt, map_location="cpu", weights_only=False)
            up = GuidedUpsampler3D(**ck["kwargs"])
            up.load_state_dict(ck["model"])
            if up.feature_keys:
                raise ValueError("an upsampler with feature keys reads the raw encoder "
                                 "features; only CT-only upsamplers are supported here")
            for p in up.parameters():
                p.requires_grad_(False)
            # Outside self.adapter, so freeze_encoder freezes it and SegModel keeps
            # it in eval: a frozen, pretrained part of the pipeline like the encoder.
            self.upsampler = up.eval()

    # ------------------------------------------------------------------ probing
    @torch.no_grad()
    def _probe(self, patch: Tuple[int, ...]):
        """Discover native taps by running the encoder once on a dummy volume.

        Returns ``(raw_input_positions, [(stride, channels), ...])``. Reading the
        shapes rather than trusting a hand-written table means the interface can
        never silently drift from the encoder (``docs/NATIVE_GRIDS.md`` is
        generated by the same measurement).
        """
        was_training = self.inner.training
        self.inner.eval()
        x = torch.zeros(1, 1, *patch)
        out = self.inner.encoder_forward(x)
        self.inner.train(was_training)

        raw, taps = [], []
        for i, t in enumerate(out):
            if not torch.is_tensor(t) or t.ndim != 5:
                raise TypeError(f"tap {i} is not a 5D tensor: {type(t).__name__}")
            _, c, d, h, w = t.shape
            if c == 1 and (d, h, w) == tuple(patch) and torch.equal(t, x):
                raw.append(i)          # the raw volume, passed along for a stem
                continue
            sd = _exact_pow2_stride(patch[0], d, "D")
            sh = _exact_pow2_stride(patch[1], h, "H")
            sw = _exact_pow2_stride(patch[2], w, "W")
            if not (sd == sh == sw):
                raise ValueError(
                    f"tap {i} has anisotropic stride ({sd},{sh},{sw}); out of scope "
                    "for I1/I2 (see docs/NATIVE_GRIDS.md)"
                )
            taps.append((sd, c))
        if not taps:
            raise ValueError("encoder produced no usable native taps")
        return raw, taps

    # ------------------------------------------------------------------- public
    def output_strides(self) -> List[int]:
        """Strides of the features this backbone emits, COARSEST FIRST."""
        s = set(self._groups)
        if self.up_skip is not None:
            s.add(self.up_stride)
        if self.use_stem:
            s.add(1)
        return sorted(s, reverse=True)

    def head_kwargs(self) -> Dict[str, object]:
        """What ``interface_seg_head`` must be built with for THIS encoder.

        The stride ladder is a property of the encoder's native grids, so it
        cannot come from a static config block the way the 5-level contract's
        can. ``scripts/train.py`` merges this over ``cfg["head"]``.
        """
        return {"width": self.width, "feature_strides": tuple(self.output_strides())}

    def k_upblocks(self) -> int:
        """Shared-upsample applications needed to reach stride 1."""
        return max(self._groups).bit_length() - 1

    # ------------------------------------------------------------------ forward
    def encoder_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        hu = None
        if self.up_skip is not None:
            if x.shape[1] != 2:
                raise ValueError("the upsampled-skip interface takes [image, HU] input "
                                 f"(data.guide_hu: true); got {x.shape[1]} channel(s)")
            x, hu = x[:, :1], x[:, 1:2]
        out = list(self.inner.encoder_forward(x))
        taps = [t for i, t in enumerate(out) if i not in self._raw_indices]
        res = ([x] if self.use_stem else []) + taps
        if self.up_skip == "guided":
            # Frozen upsampler: its attention weights depend only on the CT and the
            # grid, so they are computed here, under the encoder's no_grad, and
            # applied to the TRAINABLE projection in adapter_forward.
            coarse = min(tuple(t.shape[2:]) for t in taps)
            res.append(self._guided_weights(hu, coarse, tuple(x.shape[2:])))
        return res

    def _up_geometry(self, coarse, fine):
        from ..upsampler.geometry import default_centers, neighbor_table
        out = tuple(n // self.up_stride for n in fine)
        key = (coarse, fine)
        if key not in self._tabs:
            fc, oc = default_centers(coarse, fine), default_centers(out, fine)
            self._tabs[key] = (fc, oc, neighbor_table(fc, oc, self.upsampler.radius))
        return (out, *self._tabs[key])

    def _guided_weights(self, hu, coarse, fine):
        out, fc, oc, _ = self._up_geometry(coarse, fine)
        with torch.autocast(device_type=hu.device.type, enabled=False):
            dummy = hu.new_zeros(hu.shape[0], 1, *coarse, dtype=torch.float32)
            w, _ = self.upsampler.weights(hu.float(), dummy, fc, out_centers=oc)
        return w

    def _upsampled_skip(self, src, w, fine):
        coarse = tuple(src.shape[2:])
        out = tuple(n // self.up_stride for n in fine)
        if self.up_skip == "trilinear":
            return F.interpolate(src, size=out, mode="trilinear", align_corners=False)
        from ..upsampler.module import aggregate
        _, _, _, tab = self._up_geometry(coarse, tuple(fine))
        return aggregate(w, src, tab.to(src.device)).to(src.dtype)

    def adapter_forward(self, native, input_shape) -> List[torch.Tensor]:
        native = list(native)
        w = native.pop() if self.up_skip == "guided" else None
        if self.use_stem:
            x_raw, taps = native[0], list(native[1:])
        else:
            x_raw, taps = None, list(native)

        feats: Dict[int, torch.Tensor] = {}
        for stride, positions in self._groups.items():
            proj = self.adapter.proj[f"s{stride}"]
            acc = None
            for p in positions:                  # ONE conv, applied then summed
                y = proj(taps[p])
                acc = y if acc is None else acc + y
            feats[stride] = acc

        if self.up_skip is not None:
            skip = self._upsampled_skip(feats[max(self._groups)], w, tuple(input_shape))
            s = self.up_stride
            feats[s] = feats[s] + skip if s in feats else skip

        if self.use_stem:
            s1 = self.adapter.stem_proj(self.adapter.stem(x_raw))
            feats[1] = feats[1] + s1 if 1 in feats else s1

        return [feats[s] for s in sorted(feats, reverse=True)]

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])


class _SharedUpBlock(nn.Module):
    """One ×2 transposed conv + residual conv block, reused at every resolution.

    Skips are fused by ADDITION rather than concatenation: with concatenation the
    block's input width would depend on whether a native tap exists at that stride,
    and the same weights could not serve both cases — which is exactly what weight
    sharing requires.
    """

    def __init__(self, width: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(width, width, kernel_size=2, stride=2, bias=False)
        self.conv1 = nn.Conv3d(width, width, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(width), width)
        self.conv2 = nn.Conv3d(width, width, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(width), width)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = x + skip
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


@register_head("interface_seg_head")
class InterfaceSegHead(nn.Module):
    """Decoder for ``InterfaceBackbone``: ONE up block and ONE output conv, reused.

    Consumes features COARSEST-FIRST, all at ``width``. Climbs by ×2 until stride 1,
    adding a native tap wherever one exists at the new stride. Because both modules
    are shared across resolutions, the decoder is byte-identical for every encoder
    regardless of how many native grids it has or how many doublings it needs — the
    property that makes I1/I2 a controlled comparison rather than another capacity
    comparison.

    Returns a finest-first list of logits at strides 1, 2, 4, ... when training with
    deep supervision (matching ``ds_weights = [1, 0.5, 0.25, 0.125]``), otherwise
    the single stride-1 tensor.
    """

    def __init__(self, num_classes: int = 118, width: int = 256,
                 feature_strides: Sequence[int] = (16, 8, 4, 2, 1),
                 deep_supervision: bool = False, **_ignored):
        super().__init__()
        strides = [int(s) for s in feature_strides]
        if sorted(strides, reverse=True) != strides:
            raise ValueError(f"feature_strides must be coarsest-first, got {strides}")
        if strides[-1] < 1 or any(s & (s - 1) for s in strides):
            raise ValueError(f"feature_strides must be powers of two, got {strides}")
        self.feature_strides = tuple(strides)
        self.width = int(width)
        self.deep_supervision = bool(deep_supervision)
        self.up = _SharedUpBlock(self.width)
        self.out = nn.Conv3d(self.width, num_classes, kernel_size=1)

    def forward(self, x_in, feats: List[torch.Tensor]):
        if len(feats) != len(self.feature_strides):
            raise ValueError(
                f"expected {len(self.feature_strides)} features "
                f"(strides {self.feature_strides}), got {len(feats)}"
            )
        by_stride = dict(zip(self.feature_strides, feats))
        s = self.feature_strides[0]
        cur = feats[0]
        stages: List[torch.Tensor] = []
        while s > 1:
            s //= 2
            cur = self.up(cur, by_stride.get(s))
            stages.append(cur)
        if not stages:
            raise ValueError("coarsest stride is already 1; nothing to decode")
        if self.training and self.deep_supervision:
            return [self.out(t) for t in reversed(stages)]   # finest first
        return self.out(stages[-1])
