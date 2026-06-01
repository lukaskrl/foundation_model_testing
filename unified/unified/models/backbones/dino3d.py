"""3DINO-ViT backbone adapter with a SpatialPriorModule for fine levels.

A plain ViT has only one spatial scale (stride 16 with ``patch_size=16``).
Upsampling tokens cannot invent fine-grained detail, so the adapter splits
the contract pyramid:

  * ``feats[0..3]`` (strides 1, 2, 4, 8) come from a lightweight
    SpatialPriorModule3D — a small conv stem that runs on the *raw input
    volume*. The module's design is ported (and simplified) from
    ``3DINO/dinov2/eval/segmentation_3d/adapter_modules.py:SpatialPriorModule``
    but uses ``GroupNorm`` so it trains at small batch size on one GPU.
    Deformable Injector/Extractor blocks are intentionally omitted — they
    belong to the future deformable-head path.

  * ``feats[4]`` (stride 16) comes from the ViT's last (or last-of-extract)
    intermediate layer with a 1×1 channel projection (``embed_dim → 512``).
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


class SpatialPriorModule3D(nn.Module):
    """Conv stem on raw input that emits features at strides {1, 2, 4, 8}.

    Adapted from 3DINO's ``SpatialPriorModule`` (with the first conv changed
    from stride 2 to stride 1 so we keep a stride-1 level, and ``GroupNorm``
    in place of ``SyncBatchNorm``). Parameter count is dominated by the two
    deepest stages — keep ``inplanes`` small.
    """

    def __init__(
        self,
        in_channels: int = 1,
        inplanes: int = 16,
        out_channels=(32, 64, 128, 256),
    ):
        super().__init__()
        c0, c1, c2, c3 = out_channels

        def gn(ch):
            g = min(8, ch)
            while ch % g != 0:
                g -= 1
            return nn.GroupNorm(g, ch)

        # stride 1 stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            gn(inplanes), nn.ReLU(inplace=True),
            nn.Conv3d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            gn(inplanes), nn.ReLU(inplace=True),
            nn.Conv3d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            gn(inplanes), nn.ReLU(inplace=True),
        )
        # stride 1 -> 2
        self.down1 = nn.Sequential(
            nn.Conv3d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            gn(2 * inplanes), nn.ReLU(inplace=True),
        )
        # stride 2 -> 4
        self.down2 = nn.Sequential(
            nn.Conv3d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            gn(4 * inplanes), nn.ReLU(inplace=True),
        )
        # stride 4 -> 8
        self.down3 = nn.Sequential(
            nn.Conv3d(4 * inplanes, 8 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            gn(8 * inplanes), nn.ReLU(inplace=True),
        )

        self.proj0 = nn.Conv3d(inplanes,     c0, kernel_size=1, bias=False)
        self.proj1 = nn.Conv3d(2 * inplanes, c1, kernel_size=1, bias=False)
        self.proj2 = nn.Conv3d(4 * inplanes, c2, kernel_size=1, bias=False)
        self.proj3 = nn.Conv3d(8 * inplanes, c3, kernel_size=1, bias=False)

    def forward(self, x):
        c0 = self.stem(x)
        c1 = self.down1(c0)
        c2 = self.down2(c1)
        c3 = self.down3(c2)
        return [self.proj0(c0), self.proj1(c1), self.proj2(c2), self.proj3(c3)]


class _DinoAdapter(nn.Module):
    """Bundle: SPM (strides 1..8) + 1×1 channel projection on ViT tokens (stride 16)."""

    def __init__(self, embed_dim: int, contract_channels, spm_inplanes: int = 16):
        super().__init__()
        # Fine levels come from a SpatialPriorModule on raw input.
        self.spm = SpatialPriorModule3D(
            in_channels=1,
            inplanes=spm_inplanes,
            out_channels=contract_channels[:4],
        )
        # Semantic level (stride 16) is the ViT's tokens, channel-projected.
        c_top = contract_channels[4]

        def gn(ch):
            g = min(8, ch)
            while ch % g != 0:
                g -= 1
            return nn.GroupNorm(g, ch)
        self.vit_proj = nn.Sequential(
            nn.Conv3d(embed_dim, c_top, kernel_size=1, bias=False),
            gn(c_top),
            nn.ReLU(inplace=True),
        )


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
        spm_inplanes: int = 16,
    ):
        super().__init__()
        if patch_size != 16:
            raise ValueError("dino3d adapter assumes patch_size=16")
        DinoVisionTransformer3d = _import_dino_vit()
        self.vit = DinoVisionTransformer3d(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            **_arch_kwargs(arch),
        )
        # We only need the last extract_block for the stride-16 semantic level,
        # but we still accept the historical 4-block list (the others are
        # ignored). This keeps existing per-model configs compatible.
        self.extract_blocks = tuple(extract_blocks)
        if not self.extract_blocks:
            raise ValueError("dino3d requires at least one extract_block")

        if weights:
            state = torch.load(weights, map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                for k in ("teacher", "student", "model"):
                    if k in state and isinstance(state[k], dict):
                        state = state[k]
                        break
            new = {}
            for k, v in state.items():
                nk = k
                for pref in ("module.", "backbone.", "teacher_backbone."):
                    if nk.startswith(pref):
                        nk = nk[len(pref):]
                new[nk] = v
            self.vit.load_state_dict(new, strict=False)

        embed_dim = _arch_kwargs(arch)["embed_dim"]
        self.adapter = _DinoAdapter(
            embed_dim=embed_dim,
            contract_channels=self.EXPECTED_CHANNELS,
            spm_inplanes=spm_inplanes,
        )

    def encoder_forward(self, x):
        # We use just the deepest extracted layer for the stride-16 level.
        # `get_intermediate_layers` returns a tuple in the order of
        # `extract_blocks`; the last one is the deepest.
        layers = self.vit.get_intermediate_layers(
            x,
            n=self.extract_blocks,
            reshape=True,
            return_class_token=False,
            norm=True,
        )
        tokens = layers[-1]
        # Pass raw input alongside the ViT tokens so the trainable SPM can
        # consume it in the adapter step.
        return [x, tokens]

    def adapter_forward(self, native, input_shape):
        x_in, tokens = native
        fine = self.adapter.spm(x_in)            # 4 tensors @ strides 1..8
        s16 = self.adapter.vit_proj(tokens)      # stride 16
        return [*fine, s16]

    def forward_features(self, x):
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])
