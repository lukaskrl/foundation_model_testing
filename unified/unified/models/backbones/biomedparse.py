"""BiomedParse adapter (stub).

BiomedParse's backbone is 2D (Focal/FocalNet). To produce 3D features we run the
backbone independently on each axial slice, then add a small depth-axis 3D conv
after each stage to give the temporal dimension some receptive field.

This is the LEAST apples-to-apples backbone in the suite — the per-slice forward
has no native depth context. See docs/HEAD_DESIGN.md.

The actual Focal backbone construction lives in BiomedParse's repo; we lazy-import
it from the sibling checkout. AzureML / Olympus dependencies are not needed for
inference of just the backbone.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from ._neck import ChannelNeck, StrideAdapter

BIOMED_REPO = Path("/store/home/skrljl/projects/foundation_models/BiomedParse")


def _import_focal():
    if str(BIOMED_REPO) not in sys.path:
        sys.path.insert(0, str(BIOMED_REPO))
    # focal.py does `from detectron2.modeling import Backbone` at module level
    # but only uses it for the optional D2FocalNet wrapper. Inject a stub so we
    # can import the plain FocalNet without a real detectron2 install.
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
            f"{BIOMED_REPO}/src/model/backbone/ and adjust the import in "
            "unified/models/backbones/biomedparse.py."
        ) from e


@register_backbone("biomedparse")
class BiomedParseBackbone(BackboneInterface):
    NATIVE_STRIDES = (4, 8, 16, 32)

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
        slice_size: int = 256,
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
        # FocalNet's per-stage output channels follow embed_dim × {1, 2, 4, 8}.
        native_channels = tuple(embed_dim * (2 ** i) for i in range(4))

        if weights:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            # BiomedParse v2 stores Focal weights under "model.backbone." prefix
            # (Lightning checkpoint layout).
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

        # Per-stage 1×1×1 channel projection to {64,128,256,512}, and a 1D depth
        # conv (implemented as 3D conv with kernel (3,1,1)) to mix slices.
        out_ch = (64, 128, 256, 512)
        self.ch_necks = nn.ModuleList(
            [ChannelNeck(ic, oc) for ic, oc in zip(native_channels, out_ch)]
        )
        self.depth_mix = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(oc, oc, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
                nn.GroupNorm(min(8, oc), oc),
                nn.ReLU(inplace=True),
            )
            for oc in out_ch
        ])
        self.resize = StrideAdapter()
        self.slice_size = slice_size

    def forward_features(self, x):
        B, C, D, H, W = x.shape
        if C != 1:
            raise ValueError("BiomedParse adapter expects 1-channel CT input")
        # (B, 1, D, H, W) -> (B*D, 3, H, W)
        slices = x.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W)
        slices = slices.expand(-1, 3, -1, -1)
        # Run 2D Focal backbone — returns dict of feature maps per stage.
        stages = self.focal(slices)
        if isinstance(stages, dict):
            ordered = [stages[k] for k in ("res2", "res3", "res4", "res5")]
        else:
            ordered = list(stages)
        out = []
        for f2d, neck, mixer, s in zip(
            ordered, self.ch_necks, self.depth_mix, self.NATIVE_STRIDES
        ):
            # (B*D, c, h, w) -> (B, c, D, h, w)
            c, h, w = f2d.shape[-3], f2d.shape[-2], f2d.shape[-1]
            f3d = f2d.reshape(B, D, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
            f3d = neck(f3d)
            f3d = mixer(f3d)
            target = (D // s, H // s, W // s)
            f3d = self.resize(f3d, target)
            out.append(f3d)
        return out
