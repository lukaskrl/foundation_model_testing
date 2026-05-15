"""3DINO-ViT backbone adapter.

ViTs only have one spatial scale, so we use the UNETR projection trick: take
four intermediate blocks, all at stride patch_size, and synthesize the pyramid
by applying progressive ConvTranspose stacks to the shallower blocks.

Concretely, with patch_size=16 the ViT outputs are at stride 16. To produce
strides {4, 8, 16, 32} we:
  - feat0: upsample x4 (the shallowest block)
  - feat1: upsample x2
  - feat2: identity
  - feat3: downsample x2
And remap channels 1024 → {64, 128, 256, 512} with a 1×1×1 conv.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..registry import register_backbone
from ..seg_model import BackboneInterface

DINO_REPO = Path("/store/home/skrljl/projects/foundation_models/3DINO")


def _import_dino_vit():
    if str(DINO_REPO) not in sys.path:
        sys.path.insert(0, str(DINO_REPO))
    from dinov2.models.vision_transformer import DinoVisionTransformer3d  # type: ignore
    return DinoVisionTransformer3d


def _arch_kwargs(arch: str):
    if arch == "vit_large_3d":
        return dict(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True)
    if arch == "vit_base_3d":
        return dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True)
    raise ValueError(f"unknown 3DINO arch {arch}")


class _Up(nn.Module):
    def __init__(self, in_ch, out_ch, factor: int):
        super().__init__()
        layers = []
        cur = in_ch
        f = factor
        # progressive ×2 transposed convs
        while f > 1:
            nxt = max(out_ch, cur // 2)
            layers += [
                nn.ConvTranspose3d(cur, nxt, kernel_size=2, stride=2, bias=False),
                nn.GroupNorm(min(8, nxt), nxt),
                nn.ReLU(inplace=True),
            ]
            cur = nxt
            f //= 2
        layers.append(nn.Conv3d(cur, out_ch, kernel_size=1, bias=False))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch, factor: int):
        super().__init__()
        layers = []
        cur = in_ch
        f = factor
        while f > 1:
            nxt = min(out_ch, cur * 2) if cur < out_ch else cur
            layers += [
                nn.Conv3d(cur, nxt, kernel_size=2, stride=2, bias=False),
                nn.GroupNorm(min(8, nxt), nxt),
                nn.ReLU(inplace=True),
            ]
            cur = nxt
            f //= 2
        layers.append(nn.Conv3d(cur, out_ch, kernel_size=1, bias=False))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class _Identity1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.proj(x)


@register_backbone("dino3d")
class DinoBackbone(BackboneInterface):
    def __init__(
        self,
        weights: str,
        arch: str = "vit_large_3d",
        img_size: int = 96,
        patch_size: int = 16,
        in_chans: int = 1,
        extract_blocks=(5, 11, 17, 23),
    ):
        super().__init__()
        if patch_size != 16:
            # The pyramid math below assumes ViT stride == 16 (giving {4,8,16,32}
            # after the pyramid projection).
            raise ValueError("dino3d adapter assumes patch_size=16")
        DinoVisionTransformer3d = _import_dino_vit()
        self.vit = DinoVisionTransformer3d(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            **_arch_kwargs(arch),
        )
        self.extract_blocks = tuple(extract_blocks)
        if len(self.extract_blocks) != 4:
            raise ValueError("dino3d requires exactly 4 extract_blocks")

        if weights:
            state = torch.load(weights, map_location="cpu", weights_only=False)
            # 3DINO ckpts are typically nested under "teacher" / "student"
            if isinstance(state, dict):
                for k in ("teacher", "student", "model"):
                    if k in state and isinstance(state[k], dict):
                        state = state[k]
                        break
            # Strip common prefixes
            new = {}
            for k, v in state.items():
                nk = k
                for pref in ("module.", "backbone.", "teacher_backbone."):
                    if nk.startswith(pref):
                        nk = nk[len(pref):]
                new[nk] = v
            self.vit.load_state_dict(new, strict=False)

        # ViT output channel
        embed_dim = _arch_kwargs(arch)["embed_dim"]
        # Pyramid projection: 4× up, 2× up, identity, 2× down. Channels: {64,128,256,512}.
        self.proj0 = _Up(embed_dim, 64, factor=4)    # stride 16 -> 4
        self.proj1 = _Up(embed_dim, 128, factor=2)   # stride 16 -> 8
        self.proj2 = _Identity1x1(embed_dim, 256)    # stride 16 -> 16
        self.proj3 = _Down(embed_dim, 512, factor=2) # stride 16 -> 32

    def forward_features(self, x):
        # get_intermediate_layers handles cls token stripping + reshape to 3D grid.
        b0, b1, b2, b3 = self.vit.get_intermediate_layers(
            x,
            n=self.extract_blocks,
            reshape=True,
            return_class_token=False,
            norm=True,
        )
        return [
            self.proj0(b0),
            self.proj1(b1),
            self.proj2(b2),
            self.proj3(b3),
        ]
