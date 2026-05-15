"""VoCo / SwinUNETR backbone adapter.

Uses MONAI's SwinUNETR.swinViT encoder. The encoder natively returns 5 levels;
we take levels [1..4] which are at strides {2, 4, 8, 16} with channels
{feature_size×2, ×4, ×8, ×16}, and resample down to strides {4, 8, 16, 32}.
"""
from __future__ import annotations
import torch

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from ._neck import PyramidNeck


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
        # Construct just the SwinTransformer (== SwinUNETR.swinViT) directly so
        # we don't need MONAI's full SwinUNETR signature, which changes across
        # MONAI versions (img_size was removed in 1.4+).
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
            # Most checkpoints contain weights for the SwinUNETR decoder too;
            # those land in `unexpected`. Filter for unexpected swinViT keys only.
            unexpected_swin = [k for k in unexpected if "decoder" not in k and "out." not in k]
            if unexpected_swin and len(unexpected_swin) > 20:
                # Probably the prefix stripping is wrong; surface it.
                raise RuntimeError(
                    f"VoCo: too many unexpected swinViT keys ({len(unexpected_swin)}); "
                    "checkpoint prefix likely mismatched"
                )

        # SwinUNETR depths/heads default — channels at the 5 levels of swinViT are
        # feature_size * (1, 2, 4, 8, 16). We take levels [1..4].
        in_ch = (
            feature_size * 2,
            feature_size * 4,
            feature_size * 8,
            feature_size * 16,
        )
        self.neck = PyramidNeck(in_channels=in_ch)

    def forward_features(self, x):
        hidden = list(self.swinViT(x.contiguous(), normalize=True))
        if len(hidden) < 5:
            raise RuntimeError(
                f"SwinViT expected to return >=5 levels, got {len(hidden)}"
            )
        native = hidden[1:5]
        return self.neck(native, input_shape=x.shape[2:])
