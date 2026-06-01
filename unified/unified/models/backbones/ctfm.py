"""CT-FM SegResEncoder backbone adapter.

Architectural twin of the VISTA-3D backbone (both wrap MONAI's SegResEncoder),
but the pretrained weights come from CT-FM's intra-sample SimCLR objective on
148k CT scans (project-lighter). Two HuggingFace repos provide the weights:

  * ``ct_fm_feature_extractor`` — encoder-only safetensors (preferred). Top-level
    keys like ``conv_init.weight``, ``layers.0.blocks.0.conv1.weight``, etc.
  * ``ct_fm_segresnet`` — full SegResNet (encoder + decoder) safetensors. Keys
    are prefixed with ``encoder.`` for the encoder portion.

This adapter accepts either; when the checkpoint contains a decoder, only the
encoder portion is loaded and the decoder keys are silently dropped.

Native pyramid (``init_filters=32``, ``blocks_down=[1,2,2,4,4]``):

    L0: ( 32, D,    H,    W   )    stride 1
    L1: ( 64, D/2,  H/2,  W/2 )    stride 2
    L2: (128, D/4,  H/4,  W/4 )    stride 4
    L3: (256, D/8,  H/8,  W/8 )    stride 8
    L4: (512, D/16, H/16, W/16)    stride 16

Maps **one-to-one** onto the contract pyramid — the adapter is purely
channel-only 1×1 convs.

CT-FM was pretrained with orientation ``SPL`` and HU window ``[-1024, 2048] →
[0, 1]``. The orientation must be set per-encoder via ``model.preprocessing``;
the HU window matches the unified base.
"""
from __future__ import annotations
from pathlib import Path

import torch

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from ._neck import PyramidNeck

VISTA_REPO = Path("/store/home/skrljl/projects/foundation_models/VISTA/vista3d")


def _import_seg_res_encoder():
    # Reuse the same SegResEncoder implementation as the VISTA adapter so we
    # don't pull in different forks of the architecture.
    import importlib.util
    src = VISTA_REPO / "vista3d" / "modeling" / "segresnetds.py"
    if not src.exists():
        raise FileNotFoundError(
            f"{src}: VISTA repo not available — CT-FM adapter reuses VISTA's "
            "SegResEncoder. Either install VISTA or swap in MONAI's SegResEncoder."
        )
    spec = importlib.util.spec_from_file_location("_vista_segresnetds", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SegResEncoder


def _load_state_any_format(path: str):
    """Load .safetensors or PyTorch .ckpt/.pt into a flat state-dict."""
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(p))
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("state_dict", "net", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        return ckpt
    return ckpt


@register_backbone("ctfm")
class CTFMBackbone(BackboneInterface):
    def __init__(
        self,
        weights: str,
        init_filters: int = 32,
        blocks_down=(1, 2, 2, 4, 4),
        in_channels: int = 1,
        norm: str = "instance",
    ):
        super().__init__()
        SegResEncoder = _import_seg_res_encoder()
        if len(blocks_down) != 5:
            raise ValueError(
                "CT-FM encoder uses 5 down-blocks. Set blocks_down=[1,2,2,4,4]."
            )

        self.encoder = SegResEncoder(
            spatial_dims=3,
            init_filters=init_filters,
            in_channels=in_channels,
            blocks_down=tuple(blocks_down),
            norm=norm,
        )

        if weights:
            state = _load_state_any_format(weights)
            encoder_state = {}
            for prefix in ("encoder.", "trunk.encoder.", "backbone.encoder.", ""):
                candidate = {
                    k[len(prefix):]: v for k, v in state.items()
                    if k.startswith(prefix)
                }
                candidate = {
                    k: v for k, v in candidate.items()
                    if not k.startswith(("up_layers.", "up_samples.", "out.", "head.", "conv_final."))
                }
                if candidate:
                    encoder_state = candidate
                    break
            # CT-FM persisted InstanceNorm3d running stats; modern torch
            # InstanceNorm3d rejects them outright.
            encoder_state = {
                k: v for k, v in encoder_state.items()
                if not k.endswith((".running_mean", ".running_var", ".num_batches_tracked"))
            }
            missing, unexpected = self.encoder.load_state_dict(
                encoder_state, strict=False,
            )
            if len(missing) > 10:
                raise RuntimeError(
                    f"CT-FM: {len(missing)} missing encoder keys — "
                    f"checkpoint prefix likely mismatched. Sample missing: {missing[:5]}"
                )

        # Native channels per stage: init_filters · (1, 2, 4, 8, 16).
        native_ch = tuple(init_filters * (2 ** i) for i in range(len(blocks_down)))
        self.adapter = PyramidNeck(
            native_channels=native_ch,
            contract_channels=self.EXPECTED_CHANNELS,
            extra_down=0,
        )

    def encoder_forward(self, x):
        # SegResEncoder.forward returns features at strides
        # (1, 2, 4, 8, 16) for blocks_down=(1,2,2,4,4).
        return list(self.encoder(x))

    def adapter_forward(self, native, input_shape):
        return self.adapter(native)

    def forward_features(self, x):
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])
