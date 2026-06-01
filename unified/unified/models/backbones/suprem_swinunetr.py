"""SuPreM SwinUNETR backbone adapter.

Identical architecture to VoCo's SwinUNETR (MONAI's SwinTransformer at
feature_size=48). The pretrained weights come from SuPreM's supervised
pretraining on AbdomenAtlas 1.1 (2100 CT volumes, 25 organ + 7 tumor
annotations).

Native pyramid: 5 levels at strides ``{2, 4, 8, 16, 32}`` with channels
``feature_size · (1, 2, 4, 8, 16)``. Adapter maps strides 2..16 onto contract
levels 1..4 and a conv stem on raw input supplies the missing stride-1 level.
Stride-32 is discarded.

Checkpoint layout: ``ckpt['net']`` is a state-dict whose SwinViT keys are
prefixed with ``module.backbone.swinViT.``. Other keys (``module.organ_embedding``,
``module.controller``, ``module.GAP``, ``module.precls_conv``,
``module.text_to_vision``) belong to the SuPreM classification heads and are
discarded.

SuPreM was pretrained with a narrow soft-tissue HU window (-175..250 → 0..1);
override via ``model.preprocessing.intensity`` in the config.
"""
from __future__ import annotations

import torch

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from .voco import _SwinAdapter


def _extract_swinvit_state(ckpt):
    if isinstance(ckpt, dict):
        for k in ("net", "state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    needle = "backbone.swinViT."
    out = {}
    for k, v in ckpt.items():
        i = k.find(needle)
        if i >= 0:
            out[k[i + len(needle):]] = v
    return out


@register_backbone("suprem_swinunetr")
class SupremSwinUNETRBackbone(BackboneInterface):
    def __init__(
        self,
        weights: str,
        feature_size: int = 48,
        in_chans: int = 1,
        use_v2: bool = False,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(7, 7, 7),
        patch_size=(2, 2, 2),
    ):
        super().__init__()
        from monai.networks.nets.swin_unetr import SwinTransformer
        self.swinViT = SwinTransformer(
            in_chans=in_chans,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=list(depths),
            num_heads=list(num_heads),
            spatial_dims=3,
            use_v2=use_v2,
        )

        if weights:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            state = _extract_swinvit_state(ckpt)
            if not state:
                raise RuntimeError(
                    "SuPreM SwinUNETR: no 'backbone.swinViT.*' keys found in checkpoint"
                )
            missing, unexpected = self.swinViT.load_state_dict(state, strict=False)
            unexpected_swin = [k for k in unexpected if "decoder" not in k and "out." not in k]
            if unexpected_swin and len(unexpected_swin) > 20:
                raise RuntimeError(
                    f"SuPreM SwinUNETR: too many unexpected swinViT keys ({len(unexpected_swin)})"
                )

        native_ch = (
            feature_size * 1,
            feature_size * 2,
            feature_size * 4,
            feature_size * 8,
        )
        self.adapter = _SwinAdapter(
            native_channels=native_ch,
            contract_channels=self.EXPECTED_CHANNELS,
        )

    def encoder_forward(self, x):
        hidden = list(self.swinViT(x.contiguous(), normalize=True))
        if len(hidden) < 5:
            raise RuntimeError(
                f"SwinViT expected to return >=5 levels, got {len(hidden)}"
            )
        return [x] + hidden[0:4]

    def adapter_forward(self, native, input_shape):
        x_in, *swin_levels = native
        s1 = self.adapter.stem(x_in)
        rest = [nk(f) for nk, f in zip(self.adapter.ch_necks, swin_levels)]
        return [s1, *rest]

    def forward_features(self, x):
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])
