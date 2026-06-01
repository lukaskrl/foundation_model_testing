"""VoCo / SwinUNETR backbone adapter.

Uses MONAI's SwinUNETR.swinViT encoder. The encoder natively returns 5 levels
(``x0_out..x4_out``) at strides ``{2, 4, 8, 16, 32}`` with channels
``feature_size · (1, 2, 4, 8, 16)``.

Mapping onto the contract ``{1, 2, 4, 8, 16}``:

  * Levels 0..3 of swinViT (strides 2..16) → contract levels 1..4 (1×1 conv).
  * Contract level 0 (stride 1) is filled by a small **conv stem** that
    runs on the raw input volume. The Swin patch-embed cannot produce
    stride-1 features and we never upsample pretrained features.
  * Level 4 of swinViT (stride 32) is discarded; the head's coarsest level
    is stride 16.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from ._neck import ChannelNeck


def _strip_prefixes(state_dict):
    """VoCo / VoComni checkpoints come wrapped in several common forms."""
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "network_weights" in state_dict:
        state_dict = state_dict["network_weights"]
    elif "net" in state_dict:
        state_dict = state_dict["net"]
    elif "student" in state_dict:
        state_dict = state_dict["student"]
    new = {}
    for k, v in state_dict.items():
        nk = k
        for pref in ("module.", "backbone.", "swin_vit.", "swinViT.", "encoder."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        new[nk] = v
    return new


class _SwinAdapter(nn.Module):
    """Trainable post-encoder modules for Swin-family backbones."""

    def __init__(self, native_channels, contract_channels, stem_channels: int = 32):
        super().__init__()
        # Raw-input → stride-1 conv stem (provides the missing finest level).
        c0 = contract_channels[0]
        self.stem = nn.Sequential(
            nn.Conv3d(1, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, stem_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(stem_channels, c0, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c0),
            nn.ReLU(inplace=True),
        )
        # 1×1 channel adapters for native Swin levels (strides 2..16).
        if len(native_channels) != len(contract_channels) - 1:
            raise ValueError(
                f"Swin adapter expects {len(contract_channels) - 1} native levels, "
                f"got {len(native_channels)}"
            )
        self.ch_necks = nn.ModuleList(
            [ChannelNeck(ic, oc)
             for ic, oc in zip(native_channels, contract_channels[1:])]
        )


@register_backbone("voco_b")
@register_backbone("voco_h")
class VoCoBackbone(BackboneInterface):
    def __init__(
        self,
        weights: str,
        feature_size: int = 48,
        in_chans: int = 1,
        use_v2: bool = True,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(7, 7, 7),
        patch_size=(2, 2, 2),
    ):
        super().__init__()
        from monai.networks.nets.swin_unetr import SwinTransformer  # lazy
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
            state = torch.load(weights, map_location="cpu", weights_only=False)
            state = _strip_prefixes(state)
            missing, unexpected = self.swinViT.load_state_dict(state, strict=False)
            unexpected_swin = [k for k in unexpected if "decoder" not in k and "out." not in k]
            if unexpected_swin and len(unexpected_swin) > 20:
                raise RuntimeError(
                    f"VoCo: too many unexpected swinViT keys ({len(unexpected_swin)}); "
                    "checkpoint prefix likely mismatched"
                )

        # Native channels at the 5 levels of swinViT are feature_size·(1,2,4,8,16).
        # We use levels [0..3] (strides 2..16). Level 4 (stride 32) is discarded.
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
        # Strides 2, 4, 8, 16; we also pass x along for the conv stem.
        return [x] + hidden[0:4]

    def adapter_forward(self, native, input_shape):
        x_in, *swin_levels = native
        s1 = self.adapter.stem(x_in)
        rest = [nk(f) for nk, f in zip(self.adapter.ch_necks, swin_levels)]
        return [s1, *rest]

    def forward_features(self, x):
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])
