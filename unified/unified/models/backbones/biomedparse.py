"""BiomedParse adapter (2D FocalNet running per-axial-slice).

BiomedParse's backbone is **2D** (FocalNet on each axial slice independently),
so producing 3D features requires both depth-axis mixing *and* depth
downsampling. The adapter:

  * Runs FocalNet per slice → 4 native stages with **HW** strides
    `{4, 8, 16, 32}` and channels ``embed_dim · (1, 2, 4, 8)``. Each stage
    keeps full depth ``D``.
  * For contract levels at strides ``{4, 8, 16}`` (cubic), reshapes a stage
    to ``(B, c, D, h, w)``, mixes slices with a depth-axis 3D conv, depth-pools
    to the cubic target ``D/s``, then 1×1 channel-projects to the contract
    width.
  * For contract levels at strides ``{1, 2}`` — finer than any FocalNet
    stage — runs a small 3D conv stem on the raw input volume. The stem is
    the only source of fine detail; mirroring the Swin-family pattern.

This is the **least apples-to-apples** backbone in the suite — the per-slice
forward has no native depth context, and only the depth-mixers / SPM stem
inject any. The stem and depth mixers count as part of the trainable
adapter; the FocalNet weights are frozen.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_backbone
from ..seg_model import BackboneInterface

BIOMED_REPO = Path("/store/home/skrljl/projects/foundation_models/BiomedParse")


def _import_focal():
    if str(BIOMED_REPO) not in sys.path:
        sys.path.insert(0, str(BIOMED_REPO))
    # focal.py does `from detectron2.modeling import Backbone` at module level
    # but only uses it for the optional D2FocalNet wrapper. Inject a stub so
    # we can import the plain FocalNet without a real detectron2 install.
    import types
    if "detectron2" not in sys.modules:
        stub = types.ModuleType("detectron2")
        stub_modeling = types.ModuleType("detectron2.modeling")

        class _StubBackbone:  # the only symbol focal.py imports
            pass

        stub_modeling.Backbone = _StubBackbone
        stub.modeling = stub_modeling
        sys.modules["detectron2"] = stub
        sys.modules["detectron2.modeling"] = stub_modeling
    try:
        from src.model.backbone.focal import FocalNet  # type: ignore
        return FocalNet
    except ImportError as e:
        raise NotImplementedError(
            "BiomedParse Focal backbone import failed. Inspect "
            f"{BIOMED_REPO}/src/model/backbone/ and adjust the import."
        ) from e


def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class _BiomedAdapter(nn.Module):
    """Trainable post-encoder modules for BiomedParse.

    Two paths:
      * ``stem_s1`` / ``stem_s2`` produce features at strides 1 and 2 from the
        raw input volume.
      * ``levels[i]`` lifts FocalNet stage ``i`` (HW stride ``hw_strides[i]``,
        full depth) to a cubic-stride contract feature.
    """

    def __init__(self, native_channels, contract_channels, stem_channels: int = 32):
        super().__init__()
        c1, c2 = contract_channels[0], contract_channels[1]
        # raw -> stride 1
        self.stem_s1 = nn.Sequential(
            nn.Conv3d(1, stem_channels, kernel_size=3, padding=1, bias=False),
            _gn(stem_channels), nn.ReLU(inplace=True),
            nn.Conv3d(stem_channels, c1, kernel_size=3, padding=1, bias=False),
            _gn(c1), nn.ReLU(inplace=True),
        )
        # stride 1 -> stride 2 (raw input branch, independent of FocalNet)
        self.stem_s2 = nn.Sequential(
            nn.Conv3d(1, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _gn(stem_channels), nn.ReLU(inplace=True),
            nn.Conv3d(stem_channels, c2, kernel_size=3, padding=1, bias=False),
            _gn(c2), nn.ReLU(inplace=True),
        )
        # Per-FocalNet-level depth mixer + 1x1 channel projection. The
        # depth-axis 3x1x1 conv gives the per-slice tokens *some* receptive
        # field along D; without it the model sees independent slices.
        contract_for_focal = contract_channels[2:]  # s4, s8, s16
        if len(native_channels) != len(contract_for_focal):
            raise ValueError(
                f"native_channels ({len(native_channels)}) does not match "
                f"focal-level contract ({len(contract_for_focal)})"
            )
        self.depth_mix = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ic, ic, kernel_size=(3, 1, 1), padding=(1, 0, 0),
                          groups=ic, bias=False),
                _gn(ic),
                nn.ReLU(inplace=True),
            )
            for ic in native_channels
        ])
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ic, oc, kernel_size=1, bias=False),
                _gn(oc),
                nn.ReLU(inplace=True),
            )
            for ic, oc in zip(native_channels, contract_for_focal)
        ])


@register_backbone("biomedparse")
class BiomedParseBackbone(BackboneInterface):
    HW_STRIDES = (4, 8, 16)  # FocalNet stages used: res2, res3, res4

    def __init__(
        self,
        weights: str,
        backbone: str = "focal",
        embed_dim: int = 192,
        depths=(2, 2, 18, 2),
        focal_levels=(4, 4, 4, 4),
        focal_windows=(3, 3, 3, 3),
        patch_size: int = 4,
        use_conv_embed: bool = True,
        scaling_modulator: bool = True,
        use_layerscale: bool = True,
        use_postln: bool = True,
    ):
        super().__init__()
        if backbone != "focal":
            raise ValueError("only focal backbone is supported")

        FocalNet = _import_focal()
        self.focal = FocalNet(
            in_chans=3,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=list(depths),
            focal_levels=list(focal_levels),
            focal_windows=list(focal_windows),
            use_conv_embed=use_conv_embed,
            scaling_modulator=scaling_modulator,
            use_layerscale=use_layerscale,
            use_postln=use_postln,
        )
        # FocalNet's per-stage output channels follow embed_dim · {1, 2, 4, 8}.
        # We use stages 0..2 (HW strides 4, 8, 16). Stage 3 (HW stride 32) is
        # discarded — the contract stops at stride 16.
        native_channels = tuple(embed_dim * (2 ** i) for i in range(3))

        if weights:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            focal_state = {}
            for k, v in state.items():
                for pref in ("model.backbone.", "backbone."):
                    if k.startswith(pref):
                        focal_state[k[len(pref):]] = v
                        break
            if not focal_state:
                focal_state = state
            missing, unexpected = self.focal.load_state_dict(focal_state, strict=False)
            n_loaded = len(focal_state) - len(unexpected)
            if n_loaded < 50:
                raise RuntimeError(
                    f"BiomedParse: only {n_loaded} keys actually loaded into FocalNet; "
                    "prefix likely still mismatched"
                )

        self.adapter = _BiomedAdapter(
            native_channels=native_channels,
            contract_channels=self.EXPECTED_CHANNELS,
        )

    def encoder_forward(self, x):
        """Run the per-slice FocalNet and return raw_x + reshaped 3D stages."""
        B, C, D, H, W = x.shape
        if C != 1:
            raise ValueError("BiomedParse adapter expects 1-channel CT input")
        # (B, 1, D, H, W) -> (B*D, 3, H, W) (broadcast intensity into 3 channels)
        flat = x.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W).expand(-1, 3, -1, -1)
        stages = self.focal(flat)
        if not isinstance(stages, dict):
            raise RuntimeError("FocalNet was expected to return a dict of res* features")
        # We use res2..res4 (HW strides 4, 8, 16). res5 (HW stride 32) is dropped.
        keys = ("res2", "res3", "res4")
        # (B*D, c, h, w) -> (B, c, D, h, w)
        out = []
        for k in keys:
            f2d = stages[k]
            c, h, w = f2d.shape[-3], f2d.shape[-2], f2d.shape[-1]
            f3d = f2d.reshape(B, D, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
            out.append(f3d)
        # Pass raw input alongside FocalNet stages so the adapter can run its
        # stride-1/stride-2 stem on it.
        return [x, *out]

    def adapter_forward(self, native, input_shape):
        x_in, *focal_stages = native
        D, H, W = input_shape
        s1 = self.adapter.stem_s1(x_in)                 # stride 1
        s2 = self.adapter.stem_s2(x_in)                 # stride 2
        outs = [s1, s2]
        for i, (f3d, hw_stride) in enumerate(zip(focal_stages, self.HW_STRIDES)):
            target_d = D // hw_stride
            # Depth mix first (preserves DHW), then adaptive-pool depth to
            # target_d. HW are already at hw_stride from FocalNet.
            f = self.adapter.depth_mix[i](f3d)
            if f.shape[2] != target_d:
                f = F.adaptive_avg_pool3d(f, (target_d, f.shape[3], f.shape[4]))
            f = self.adapter.proj[i](f)
            outs.append(f)
        return outs

    def forward_features(self, x):
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])
