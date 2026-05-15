"""VISTA-3D SegResNet-DS encoder backbone adapter.

Drops the VISTA point/class heads — we only use the SegResEncoder. Native
outputs are 4 features at strides {2, 4, 8, 16} with channels
{init_filters * (1, 2, 4, 8)}; we resample down to strides {4, 8, 16, 32}.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from ._neck import PyramidNeck

# VISTA's SegResEncoder lives in the sibling repo; importing requires it on PYTHONPATH.
VISTA_REPO = Path("/store/home/skrljl/projects/foundation_models/VISTA/vista3d")


def _import_seg_res_encoder():
    # Load segresnetds.py directly. Going through `import vista3d.modeling...`
    # triggers vista3d/__init__.py, which transitively imports `scripts.utils`
    # and other VISTA-only modules we don't need.
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
        if len(blocks_down) < 5:
            raise ValueError(
                "VISTA-3D pretrained encoder uses 5 down-blocks. Set blocks_down=[1,2,2,4,4]."
            )

        # SegResEncoder forward returns a list of features (one per block_down level)
        # with channels [init_filters * 2**i for i in range(len(blocks_down))].
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
            # The pretrained VISTA-3D checkpoint contains the full model;
            # we keep only keys belonging to the encoder.
            # In VISTA's vista3d.py, the encoder is registered under .image_encoder
            # or .feature_extractor depending on version. We match by prefix.
            encoder_state = {}
            for k, v in ckpt.items():
                for prefix in ("image_encoder.encoder.", "image_encoder.",
                               "module.image_encoder.", "encoder."):
                    if k.startswith(prefix):
                        encoder_state[k[len(prefix):]] = v
                        break
            if not encoder_state:
                # Try a direct load; this works if the user supplies an encoder-only ckpt.
                encoder_state = ckpt
            self.encoder.load_state_dict(encoder_state, strict=False)

        # All-stage channels. We pick stages [1..4] of a 5-stage encoder so the
        # pyramid we feed to the neck is at strides {2, 4, 8, 16} with channels
        # (init_filters*2, *4, *8, *16). The neck then re-samples to {4,8,16,32}.
        all_ch = tuple(init_filters * (2 ** i) for i in range(len(blocks_down)))
        self.pick = slice(1, 5)
        self.neck = PyramidNeck(in_channels=all_ch[self.pick])

    def forward_features(self, x):
        feats = list(self.encoder(x))
        return self.neck(feats[self.pick], input_shape=x.shape[2:])
