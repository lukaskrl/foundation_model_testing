"""VISTA-3D SegResNet-DS encoder backbone adapter.

Drops the VISTA point/class heads — we only use the SegResEncoder. Native
pyramid (``init_filters=48``, ``blocks_down=[1,2,2,4,4]``):

    L0: ( 48, D,    H,    W   )    stride 1
    L1: ( 96, D/2,  H/2,  W/2 )    stride 2
    L2: (192, D/4,  H/4,  W/4 )    stride 4
    L3: (384, D/8,  H/8,  W/8 )    stride 8
    L4: (768, D/16, H/16, W/16)    stride 16

Maps one-to-one onto the contract pyramid; the adapter only does 1×1 channel
projection at each level.
"""
from __future__ import annotations
from pathlib import Path

import torch

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from ._neck import PyramidNeck

VISTA_REPO = Path("/store/home/skrljl/projects/foundation_models/VISTA/vista3d")


def _import_seg_res_encoder():
    import importlib.util
    src = VISTA_REPO / "vista3d" / "modeling" / "segresnetds.py"
    if not src.exists():
        raise FileNotFoundError(src)
    spec = importlib.util.spec_from_file_location("_vista_segresnetds", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SegResEncoder


@register_backbone("vista3d")
class VistaBackbone(BackboneInterface):
    def __init__(
        self,
        weights: str,
        init_filters: int = 48,
        blocks_down=(1, 2, 2, 4, 4),
        in_channels: int = 1,
        norm="instance",
    ):
        super().__init__()
        SegResEncoder = _import_seg_res_encoder()
        if len(blocks_down) != 5:
            raise ValueError(
                "VISTA-3D pretrained encoder uses 5 down-blocks. Set blocks_down=[1,2,2,4,4]."
            )

        self.encoder = SegResEncoder(
            spatial_dims=3,
            init_filters=init_filters,
            in_channels=in_channels,
            blocks_down=tuple(blocks_down),
            norm=norm,
        )

        if weights:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            encoder_state = {}
            for k, v in ckpt.items():
                for prefix in ("image_encoder.encoder.", "image_encoder.",
                               "module.image_encoder.", "encoder."):
                    if k.startswith(prefix):
                        encoder_state[k[len(prefix):]] = v
                        break
            if not encoder_state:
                encoder_state = ckpt
            self.encoder.load_state_dict(encoder_state, strict=False)

        native_ch = tuple(init_filters * (2 ** i) for i in range(len(blocks_down)))
        self.adapter = PyramidNeck(
            native_channels=native_ch,
            contract_channels=self.EXPECTED_CHANNELS,
            extra_down=0,
        )

    def encoder_forward(self, x):
        return list(self.encoder(x))

    def adapter_forward(self, native, input_shape):
        return self.adapter(native)

    def forward_features(self, x):
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])
