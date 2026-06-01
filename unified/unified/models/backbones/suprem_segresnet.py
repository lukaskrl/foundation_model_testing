"""SuPreM SegResNet backbone adapter.

SuPreM's smallest backbone (4.70 M params total for the full SegResNet). The
encoder portion is MONAI ``SegResNet``'s ``convInit`` + ``down_layers`` —
*not* the SegResEncoder class that VISTA/CT-FM use. Layer naming differs:

  * ``module.convInit.conv.weight`` (init_filters=16)
  * ``module.down_layers.0..3`` (4 stages — fewer than CT-FM/VISTA's 5)
  * ``module.up_layers``, ``module.up_samples``, ``module.conv_final`` (decoder)

Native encoder pyramid (``init_filters=16``, ``blocks_down=[1,2,2,4]``):

    L0: ( 16, D,   H,   W   )    stride 1
    L1: ( 32, D/2, H/2, W/2 )    stride 2
    L2: ( 64, D/4, H/4, W/4 )    stride 4
    L3: (128, D/8, H/8, W/8 )    stride 8

Native top stride is 8; the contract needs stride 16, which the adapter
synthesizes with one 3×3×3 stride-2 conv on the (channel-adapted) stride-8
feature.

We construct a MONAI ``SegResNet`` (which includes both encoder and decoder)
to receive the weights, then expose intermediate encoder features for the
unified pyramid contract.
"""
from __future__ import annotations

import torch

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from ._neck import PyramidNeck


def _extract_state(ckpt):
    if isinstance(ckpt, dict):
        for k in ("net", "state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    out = {}
    for k, v in ckpt.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        out[nk] = v
    return out


@register_backbone("suprem_segresnet")
class SupremSegResNetBackbone(BackboneInterface):
    def __init__(
        self,
        weights: str,
        init_filters: int = 16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        in_channels: int = 1,
        out_channels: int = 32,  # matches SuPreM's training (organ_embedding=32)
        norm: str = "instance",
    ):
        super().__init__()
        from monai.networks.nets import SegResNet
        self.net = SegResNet(
            spatial_dims=3,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            blocks_down=tuple(blocks_down),
            blocks_up=tuple(blocks_up),
            norm=norm,
        )

        if weights:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            state = _extract_state(ckpt)
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            if len([k for k in missing if k.startswith(("convInit.", "down_layers."))]) > 5:
                raise RuntimeError(
                    f"SuPreM SegResNet: too many missing encoder keys "
                    f"({len(missing)} missing total) — checkpoint mismatch"
                )

        native_ch = tuple(init_filters * (2 ** i) for i in range(len(blocks_down)))
        self.adapter = PyramidNeck(
            native_channels=native_ch,
            contract_channels=self.EXPECTED_CHANNELS,
            extra_down=self.NUM_LEVELS - len(native_ch),
        )

    def encoder_forward(self, x):
        # Reproduce MONAI SegResNet.encode internals (without decoder).
        # Returns 4 skip tensors at strides {1, 2, 4, 8}.
        net = self.net
        x = net.convInit(x)
        if hasattr(net, "dropout") and net.dropout is not None:
            x = net.dropout(x)
        skips = []
        for block in net.down_layers:
            x = block(x)
            skips.append(x)
        return skips

    def adapter_forward(self, native, input_shape):
        return self.adapter(native)

    def forward_features(self, x):
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])
