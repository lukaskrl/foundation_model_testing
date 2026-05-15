"""CT-CLIP adapter (stub).

CT-CLIP's image encoder is CTViT — a 3D VAE-ish ViT producing a single
bottleneck `(B, 512, T/4, H/16, W/16)` (with default config). There are no
intermediate skip features, so the adapter takes the bottleneck and synthesizes
the four-scale pyramid via learned upsampling.

This adds more 'fresh' parameters at the pyramid stage than other backbones'
necks; CT-CLIP results carry that asterisk (see docs/HEAD_DESIGN.md).
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_backbone
from ..seg_model import BackboneInterface

CTCLIP_REPO = Path("/store/home/skrljl/projects/foundation_models/CT-CLIP")


def _import_ctvit():
    """Load CTViT bypassing transformer_maskgit/__init__.py.

    The package __init__.py pulls in transformers, T5, and a lot of other
    things we don't need for the image encoder. We load ctvit.py and its only
    sibling dependency (attention.py) by file path.
    """
    pkg_dir = CTCLIP_REPO / "transformer_maskgit" / "transformer_maskgit"
    if not pkg_dir.exists():
        raise NotImplementedError(
            f"CT-CLIP source missing at {pkg_dir}. Ensure the CT-CLIP repo is "
            "cloned at /store/home/skrljl/projects/foundation_models/CT-CLIP."
        )
    import importlib.util
    # Pre-register a stub `transformer_maskgit` package so `from transformer_maskgit.attention import ...`
    # in ctvit.py resolves to our hand-loaded module.
    import types
    pkg_name = "transformer_maskgit"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_dir)]
        sys.modules[pkg_name] = pkg

    def _load(modname, filename):
        full = f"{pkg_name}.{modname}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, pkg_dir / filename)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("attention", "attention.py")
    ctvit_mod = _load("ctvit", "ctvit.py")
    return ctvit_mod.CTViT


class _PyramidFromBottleneck(nn.Module):
    """Take a single (B, C, d, h, w) bottleneck and emit 4 pyramid features."""

    def __init__(self, in_ch: int, out_channels=(64, 128, 256, 512)):
        super().__init__()
        c0, c1, c2, c3 = out_channels
        # Each level is a 1×1×1 channel projection + a (potentially identity)
        # spatial resize done at runtime in forward().
        self.proj0 = nn.Conv3d(in_ch, c0, kernel_size=1, bias=False)
        self.proj1 = nn.Conv3d(in_ch, c1, kernel_size=1, bias=False)
        self.proj2 = nn.Conv3d(in_ch, c2, kernel_size=1, bias=False)
        self.proj3 = nn.Conv3d(in_ch, c3, kernel_size=1, bias=False)

    def forward(self, bottleneck, input_shape):
        D, H, W = input_shape
        t0 = (D // 4, H // 4, W // 4)
        t1 = (D // 8, H // 8, W // 8)
        t2 = (D // 16, H // 16, W // 16)
        t3 = (D // 32, H // 32, W // 32)

        def up(p, t):
            x = p(bottleneck)
            if tuple(x.shape[2:]) != tuple(t):
                x = F.interpolate(x, size=t, mode="trilinear", align_corners=False)
            return x

        return [up(self.proj0, t0), up(self.proj1, t1), up(self.proj2, t2), up(self.proj3, t3)]


@register_backbone("ctclip")
class CTClipBackbone(BackboneInterface):
    def __init__(
        self,
        weights: str,
        dim: int = 512,
        codebook_size: int = 8192,
        image_size: int = 480,
        patch_size: int = 20,
        temporal_patch_size: int = 10,
        spatial_depth: int = 4,
        temporal_depth: int = 4,
        dim_head: int = 32,
        heads: int = 8,
        use_image_encoder_only: bool = True,
    ):
        super().__init__()
        CTViT = _import_ctvit()
        self.encoder = CTViT(
            dim=dim,
            codebook_size=codebook_size,
            image_size=image_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_depth=spatial_depth,
            temporal_depth=temporal_depth,
            dim_head=dim_head,
            heads=heads,
        )

        if weights:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            # CT-CLIP checkpoints store both visual and text encoders; we take
            # just the visual side.
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            visual = {
                k.split("visual_transformer.", 1)[1]: v
                for k, v in state.items()
                if "visual_transformer." in k
            }
            if not visual:
                visual = state
            self.encoder.load_state_dict(visual, strict=False)

        # CTViT's ContinuousPositionBias.forward has a hard-coded device=cuda
        # at the inner torch.arange call. Replace it with a version that
        # respects the passed `device` argument. (Upstream bug.)
        self._patch_pos_bias(self.encoder.spatial_rel_pos_bias)

        self.pyramid = _PyramidFromBottleneck(in_ch=dim)

    @staticmethod
    def _patch_pos_bias(pos_bias_module):
        import types
        from einops import rearrange

        def forward(self, *dimensions, device=None):
            if device is None:
                device = torch.device("cpu")
            if not getattr(self, "rel_pos", None) is not None or not self.cache_rel_pos:
                positions = [torch.arange(d, device=device) for d in dimensions]
                grid = torch.stack(torch.meshgrid(*positions, indexing="ij"))
                grid = rearrange(grid, "c ... -> (...) c")
                rel_pos = rearrange(grid, "i c -> i 1 c") - rearrange(grid, "j c -> 1 j c")
                if self.log_dist:
                    rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)
                self.register_buffer("rel_pos", rel_pos, persistent=False)
            rel_pos = self.rel_pos.to(torch.float32).to(device)
            for layer in self.net:
                rel_pos = layer(rel_pos.float())
            return rearrange(rel_pos, "i j h -> h i j")

        pos_bias_module.forward = types.MethodType(forward, pos_bias_module)

    def _encode(self, tokens):
        """Re-implementation of CTViT.encode that uses the tensor's device
        instead of CTViT's hard-coded `torch.device('cuda')` (lines 292, 332 of
        upstream ctvit.py). Identical otherwise.
        """
        from einops import rearrange
        b = tokens.shape[0]
        h, w = self.encoder.patch_height_width
        video_shape = tuple(tokens.shape[:-1])
        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        attn_bias = self.encoder.spatial_rel_pos_bias(h, w, device=tokens.device)
        tokens = self.encoder.enc_spatial_transformer(
            tokens, attn_bias=attn_bias, video_shape=video_shape
        )
        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)
        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")
        tokens = self.encoder.enc_temporal_transformer(tokens, video_shape=video_shape)
        tokens = rearrange(tokens, "(b h w) t d -> b t h w d", b=b, h=h, w=w)
        return tokens

    def forward_features(self, x):
        # CTViT was pretrained on (B, 1, T, 480, 480) with T divisible by
        # temporal_patch_size (10). Resize in-plane up to 480×480.
        B, C, D, H, W = x.shape
        if C != 1:
            raise ValueError("ct-clip adapter expects 1-channel CT input")
        target_hw = (self.encoder.image_size, self.encoder.image_size) \
            if isinstance(self.encoder.image_size, int) \
            else tuple(self.encoder.image_size)
        flat = x.reshape(B * D, 1, H, W)
        flat = F.interpolate(flat, size=target_hw, mode="bilinear", align_corners=False)
        x_resized = flat.reshape(B, 1, D, *target_hw)

        # Bypass the VQ codebook — we only need the encoder features.
        tokens = self.encoder.to_patch_emb(x_resized)  # (b, t', h', w', d)
        tokens = self._encode(tokens)                   # (b, t', h', w', d)
        bottleneck = tokens.permute(0, 4, 1, 2, 3).contiguous()
        return self.pyramid(bottleneck, input_shape=x.shape[2:])
