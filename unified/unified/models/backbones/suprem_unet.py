"""SuPreM U-Net 3D backbone adapter.

SuPreM uses a custom UNet3D (not MONAI's UNet) — see
``SuPreM/direct_inference/model/Unet.py``. Layer naming:
``module.backbone.down_tr64 / down_tr128 / down_tr256 / down_tr512`` for the
encoder and ``module.backbone.up_tr*`` for the decoder.

Native encoder pyramid (skip tensors before the per-stage downsample):

    skip_out64  : ( 64, D,    H,    W   )   stride 1
    skip_out128 : (128, D/2,  H/2,  W/2 )   stride 2
    skip_out256 : (256, D/4,  H/4,  W/4 )   stride 4
    out512_skip : (512, D/8,  H/8,  W/8 )   stride 8

Native top stride is 8; the contract needs stride 16, which the adapter
synthesizes with a single 3×3×3 stride-2 conv on the (channel-adapted)
stride-8 feature.
"""
from __future__ import annotations
from pathlib import Path
import sys

import torch
import torch.nn as nn

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from ._neck import PyramidNeck

SUPREM_REPO = Path("/store/home/skrljl/projects/foundation_models/SuPreM")


def _import_unet3d():
    import importlib.util
    src = SUPREM_REPO / "direct_inference" / "model" / "Unet.py"
    if not src.exists():
        raise FileNotFoundError(
            f"{src}: SuPreM repo not available. Required for SuPreM U-Net adapter."
        )
    spec = importlib.util.spec_from_file_location("_suprem_unet", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.UNet3D


def _extract_backbone_state(ckpt):
    if isinstance(ckpt, dict):
        for k in ("net", "state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    needle = "backbone."
    out = {}
    for k, v in ckpt.items():
        i = k.find(needle)
        if i >= 0:
            out[k[i + len(needle):]] = v
    return out


@register_backbone("suprem_unet")
class SupremUNet3DBackbone(BackboneInterface):
    def __init__(
        self,
        weights: str,
        act: str = "relu",
    ):
        super().__init__()
        UNet3D = _import_unet3d()
        self.unet = UNet3D(n_class=1, act=act)

        if weights:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            state = _extract_backbone_state(ckpt)
            if not state:
                raise RuntimeError(
                    "SuPreM U-Net: no 'backbone.*' keys found in checkpoint"
                )
            missing, unexpected = self.unet.load_state_dict(state, strict=False)
            critical_missing = [
                k for k in missing
                if k.startswith(("down_tr64.", "down_tr128.", "down_tr256.", "down_tr512."))
            ]
            if len(critical_missing) > 5:
                raise RuntimeError(
                    f"SuPreM U-Net: too many missing encoder keys "
                    f"({len(critical_missing)}). Sample: {critical_missing[:5]}"
                )

        # Native channels at strides (1, 2, 4, 8).
        native_ch = (64, 128, 256, 512)
        # Contract needs strides (1, 2, 4, 8, 16) — extra_down=1 synthesizes
        # the missing stride-16 level via a single 3×3 stride-2 conv.
        self.adapter = PyramidNeck(
            native_channels=native_ch,
            contract_channels=self.EXPECTED_CHANNELS,
            extra_down=1,
        )

    def encoder_forward(self, x):
        # Replicate UNet3D's encoder forward without the decoder. Each
        # ``down_trN`` returns ``(downsampled_out, skip_at_input_stride)``.
        # We use the skip (taken BEFORE the per-stage stride-2 downsample) so
        # the four features are at strides 1, 2, 4, 8.
        out64,  s1 = self.unet.down_tr64(x)
        out128, s2 = self.unet.down_tr128(out64)
        out256, s4 = self.unet.down_tr256(out128)
        _,      s8 = self.unet.down_tr512(out256)
        return [s1, s2, s4, s8]

    def adapter_forward(self, native, input_shape):
        return self.adapter(native)

    def forward_features(self, x):
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])
