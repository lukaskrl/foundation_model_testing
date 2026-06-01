"""SAM-Med3D backbone adapter.

SAM-Med3D is a 3D adaptation of Segment Anything. We use just the image
encoder (``ImageEncoderViT3D``) — the prompt encoder and mask decoder
require interactive prompts and are discarded.

The ``sam_med3d_turbo.pth`` checkpoint matches the **``vit_b_ori``** variant:

    embed_dim=768, depth=12, num_heads=12,
    image_size=128, patch_size=16, window_size=14,
    global_attn_indexes=(2, 5, 8, 11),
    out_chans=384 (the neck projects from 768 to 384)

→ pretraining token grid is 8×8×8, bottleneck shape is
  ``(B, 384, 8, 8, 8)`` for a 128³ input.

To handle arbitrary input patch sizes without dealing with absolute
positional-embedding interpolation, the adapter resizes the raw input to
128³ before the ViT pass, then trilinearly resamples the bottleneck to the
contract's stride-16 target ``(D/16, H/16, W/16)``. Mirrors the canvas-
resize pattern used by ctclip. The rel-pos buffers are size-flexible
(upstream ``get_rel_pos`` interpolates) so window/global attention adapts
to the runtime token grid automatically.

Adapter layout (mirrors dino3d / ctclip):

  * SpatialPriorModule3D on raw input → contract strides 1, 2, 4, 8
  * 1×1 conv on ViT bottleneck (384 → 512) → contract stride 16
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from .dino3d import SpatialPriorModule3D

SAM_REPO = Path("/store/home/skrljl/projects/foundation_models/SAM-Med3D")


def _import_image_encoder():
    import importlib.util
    src = SAM_REPO / "segment_anything" / "modeling" / "image_encoder3D.py"
    if not src.exists():
        raise FileNotFoundError(
            f"{src}: SAM-Med3D repo not available at {SAM_REPO}."
        )
    spec = importlib.util.spec_from_file_location("_sam_med3d_image_encoder3D", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ImageEncoderViT3D


def _arch_kwargs(arch: str):
    """SAM-Med3D variant configs. The turbo checkpoint matches ``vit_b_ori``."""
    if arch == "vit_b_ori":
        return dict(
            embed_dim=768, depth=12, num_heads=12,
            global_attn_indexes=(2, 5, 8, 11),
            image_size=128,
        )
    if arch == "vit_b":
        return dict(
            embed_dim=384, depth=12, num_heads=12,
            global_attn_indexes=(2, 5, 8, 11),
            image_size=256,
        )
    if arch == "vit_l":
        return dict(
            embed_dim=1024, depth=24, num_heads=16,
            global_attn_indexes=(5, 11, 17, 23),
            image_size=256,
        )
    if arch == "vit_h":
        return dict(
            embed_dim=1280, depth=32, num_heads=16,
            global_attn_indexes=(7, 15, 23, 31),
            image_size=256,
        )
    raise ValueError(f"unknown SAM-Med3D arch {arch}")


def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class _SAMMed3DAdapter(nn.Module):
    """SPM (strides 1..8) + 1×1 projection on ViT bottleneck (stride 16)."""

    def __init__(self, bottleneck_dim: int, contract_channels, spm_inplanes: int = 16):
        super().__init__()
        self.spm = SpatialPriorModule3D(
            in_channels=1,
            inplanes=spm_inplanes,
            out_channels=contract_channels[:4],
        )
        c_top = contract_channels[4]
        self.vit_proj = nn.Sequential(
            nn.Conv3d(bottleneck_dim, c_top, kernel_size=1, bias=False),
            _gn(c_top),
            nn.ReLU(inplace=True),
        )


@register_backbone("sam_med3d")
class SAMMed3DBackbone(BackboneInterface):
    def __init__(
        self,
        weights: str,
        arch: str = "vit_b_ori",
        patch_size: int = 16,
        in_chans: int = 1,
        window_size: int = 14,
        out_chans: int = 384,
        use_rel_pos: bool = True,
        spm_inplanes: int = 16,
    ):
        super().__init__()
        ImageEncoderViT3D = _import_image_encoder()
        ak = _arch_kwargs(arch)
        self.image_size = ak["image_size"]
        self.encoder = ImageEncoderViT3D(
            img_size=ak["image_size"],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=ak["embed_dim"],
            depth=ak["depth"],
            num_heads=ak["num_heads"],
            mlp_ratio=4.0,
            out_chans=out_chans,
            qkv_bias=True,
            use_abs_pos=True,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=True,
            window_size=window_size,
            global_attn_indexes=tuple(ak["global_attn_indexes"]),
        )

        if weights:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            state = ckpt
            if isinstance(ckpt, dict):
                for k in ("model_state_dict", "state_dict", "model"):
                    if k in ckpt and isinstance(ckpt[k], dict):
                        state = ckpt[k]
                        break
            # Keep only image_encoder.* keys and strip the prefix.
            needle = "image_encoder."
            encoder_state = {
                k[len(needle):]: v for k, v in state.items()
                if k.startswith(needle)
            }
            if not encoder_state:
                encoder_state = state
            missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
            if len(missing) > 5 and not any("rel_pos" in m or "pos_embed" in m for m in missing):
                raise RuntimeError(
                    f"SAM-Med3D: {len(missing)} encoder keys missing — "
                    f"checkpoint prefix mismatch? Sample: {missing[:5]}"
                )

        self.adapter = _SAMMed3DAdapter(
            bottleneck_dim=out_chans,
            contract_channels=self.EXPECTED_CHANNELS,
            spm_inplanes=spm_inplanes,
        )

    def _run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Resize input to the pretrained canvas, run encoder, return bottleneck.

        SAM-Med3D's absolute positional embedding is sized for the pretrain
        token grid; resizing the input to ``image_size³`` sidesteps the
        need to interpolate ``pos_embed`` at every forward.
        """
        B, C, D, H, W = x.shape
        if C != 1:
            raise ValueError("sam_med3d adapter expects 1-channel CT input")
        target = (self.image_size, self.image_size, self.image_size)
        if (D, H, W) != target:
            x_canvas = F.interpolate(x, size=target, mode="trilinear", align_corners=False)
        else:
            x_canvas = x
        return self.encoder(x_canvas)   # (B, out_chans, 8, 8, 8)

    def encoder_forward(self, x):
        bottleneck = self._run_encoder(x)
        return [x, bottleneck]

    def adapter_forward(self, native, input_shape):
        x_in, bottleneck = native
        D, H, W = input_shape
        fine = self.adapter.spm(x_in)                  # strides 1..8
        s16 = self.adapter.vit_proj(bottleneck)        # 512 ch on canvas grid
        target = (D // 16, H // 16, W // 16)
        if tuple(s16.shape[2:]) != target:
            s16 = F.interpolate(s16, size=target, mode="trilinear", align_corners=False)
        return [*fine, s16]

    def forward_features(self, x):
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])
