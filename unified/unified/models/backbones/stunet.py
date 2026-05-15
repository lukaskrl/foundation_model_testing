"""STU-Net adapter (stub).

STU-Net is nnU-Net V1-based; the encoder code is NOT in the local checkout. To use
this adapter:

  1. git clone https://github.com/uni-medical/STU-Net.git vendor/STU-Net
  2. Install nnUNet V1 (1.7.0) in a separate env (the older PyTorch is incompatible
     with the default env).
  3. Wire `_build_stunet_encoder(size)` below to construct the encoder from the
     upstream code.

For now this raises NotImplementedError so the framework's tests / CI catch any
config that points here without the vendor code present.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from ._neck import PyramidNeck

VENDOR = Path(__file__).resolve().parents[3] / "vendor" / "STU-Net"


def _build_stunet_encoder(size: str):
    """Return the bare encoder module of STU-Net (no decoder/head)."""
    if not VENDOR.exists():
        raise NotImplementedError(
            "STU-Net upstream code missing. Clone it:\n"
            "    git clone https://github.com/uni-medical/STU-Net.git vendor/STU-Net\n"
            "Then implement _build_stunet_encoder() to import the appropriate "
            "STUNet network and return its encoder."
        )
    sys.path.insert(0, str(VENDOR))
    # TODO: STU-Net's network is `nnunet.network_architecture.STUNet.STUNet` or
    # similar; the precise import depends on upstream layout. The encoder is the
    # downsampling pathway accessed via model.conv_blocks_context (nnU-Net V1).
    raise NotImplementedError(
        "STU-Net encoder construction is not implemented. After cloning the "
        "upstream repo, build a STUNet model with the matching `size` "
        "({small,base,large,huge}) and return its encoder pathway."
    )


@register_backbone("stunet_small")
@register_backbone("stunet_base")
@register_backbone("stunet_large")
@register_backbone("stunet_huge")
class STUNetBackbone(BackboneInterface):
    SIZE_CHANNELS = {
        # native nnU-Net V1 with 5 stages: in_channels per stage at strides
        # {1, 2, 4, 8, 16}. We pick stages [1..4]. These are typical STU-Net
        # numbers and need confirmation against the upstream code once cloned.
        "small": (32, 64, 128, 256, 320),
        "base":  (32, 64, 128, 256, 320),
        "large": (64, 128, 256, 512, 512),
        "huge":  (96, 192, 384, 768, 768),
    }

    def __init__(self, weights: str, size: str = "small", plans_pkl: str | None = None):
        super().__init__()
        if size not in self.SIZE_CHANNELS:
            raise ValueError(f"unknown STU-Net size {size!r}")
        self.encoder = _build_stunet_encoder(size)  # raises until vendor cloned

        if weights:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            self.encoder.load_state_dict(state, strict=False)

        ch5 = self.SIZE_CHANNELS[size]
        # Pick stages [1..4] at strides {2,4,8,16}; PyramidNeck resamples to {4,8,16,32}.
        self.neck = PyramidNeck(in_channels=ch5[1:5])

    def forward_features(self, x):
        # nnU-Net V1 encoder forward: each conv_block_context returns its output;
        # we collect them.
        feats = []
        skip = x
        for stage in self.encoder.conv_blocks_context:
            skip = stage(skip)
            feats.append(skip)
        native = feats[1:5]
        return self.neck(native, input_shape=x.shape[2:])
