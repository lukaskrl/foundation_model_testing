"""CT-CLIP adapter.

CT-CLIP's image encoder is CTViT — a 3D VAE-ish ViT producing a single
bottleneck `(B, dim, T/temporal_patch, H/patch, W/patch)`. CTViT was
pretrained on **480×480** in-plane patches with temporal patches of
``temporal_patch_size`` frames, so the encoder is run on a resized,
depth-padded copy of the input and the bottleneck is anisotropic
(different temporal vs spatial strides relative to the resized input).

There are no intermediate skip features. Mirroring the dino3d adapter:

  * Fine levels (strides 1, 2, 4, 8) come from a lightweight
    ``SpatialPriorModule3D`` running on the **raw input volume** — a fresh
    conv branch, the only source of native-resolution detail.
  * Stride-16 (the contract's coarsest level) comes from the CTViT
    bottleneck: a 1×1×1 channel projection to 512 ch, followed by a single
    trilinear resample to the cubic stride-16 target. This is the *only*
    spatial resample of pretrained features in any adapter; it's required
    because the CTViT bottleneck is anisotropic and operates on a resized
    canvas. The CT-CLIP comparison therefore carries an asterisk relative
    to backbones whose native pyramids fit the contract cleanly.

CT-CLIP results carry that asterisk (see docs/HEAD_DESIGN.md).
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from .dino3d import SpatialPriorModule3D
from ._neck import UpsampleNeck

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


def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class _CTClipAdapter(nn.Module):
    """Adapt the CTViT bottleneck onto the contract pyramid.

    ``pyramid_mode``: ``"spm"`` (default) = SPM on raw input (strides 1..8) +
    1×1 projection of the CTViT bottleneck (stride 16); ``"upsample"`` = every
    level is a channel-projected, trilinearly resampled copy of the bottleneck
    (``UpsampleNeck``), no raw-input path — for frozen-encoder probes.
    """

    def __init__(self, ctvit_dim: int, contract_channels, spm_inplanes: int = 16,
                 pyramid_mode: str = "spm"):
        super().__init__()
        self.pyramid_mode = pyramid_mode
        if pyramid_mode == "spm":
            self.spm = SpatialPriorModule3D(
                in_channels=1,
                inplanes=spm_inplanes,
                out_channels=contract_channels[:4],
            )
            c_top = contract_channels[4]
            self.vit_proj = nn.Sequential(
                nn.Conv3d(ctvit_dim, c_top, kernel_size=1, bias=False),
                _gn(c_top),
                nn.ReLU(inplace=True),
            )
        elif pyramid_mode == "upsample":
            self.upsample_neck = UpsampleNeck(
                in_channels=ctvit_dim,
                contract_channels=contract_channels,
            )
        else:
            raise ValueError(
                f"unknown pyramid_mode {pyramid_mode!r}; expected 'spm' or 'upsample'"
            )


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
        spm_inplanes: int = 16,
        pyramid_mode: str = "spm",
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
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            visual = {
                k.split("visual_transformer.", 1)[1]: v
                for k, v in state.items()
                if "visual_transformer." in k
            }
            if not visual:
                visual = state
            self.encoder.load_state_dict(visual, strict=False)

        # CTViT's ContinuousPositionBias has a hard-coded device=cuda; patch
        # so it respects the passed `device` argument.
        self._patch_pos_bias(self.encoder.spatial_rel_pos_bias)

        self.adapter = _CTClipAdapter(
            ctvit_dim=dim,
            contract_channels=self.EXPECTED_CHANNELS,
            spm_inplanes=spm_inplanes,
            pyramid_mode=pyramid_mode,
        )

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
        """Reimplementation of CTViT.encode that respects tensor device.

        Identical to upstream except the hard-coded ``torch.device('cuda')``
        on lines 292, 332 of upstream ctvit.py.
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

    def _run_ctvit(self, x):
        """Resize input to the pretrained canvas, encode, return bottleneck."""
        B, C, D, H, W = x.shape
        if C != 1:
            raise ValueError("ct-clip adapter expects 1-channel CT input")
        target_hw = (
            (self.encoder.image_size, self.encoder.image_size)
            if isinstance(self.encoder.image_size, int)
            else tuple(self.encoder.image_size)
        )
        flat = x.reshape(B * D, 1, H, W)
        flat = F.interpolate(flat, size=target_hw, mode="bilinear", align_corners=False)
        x_resized = flat.reshape(B, 1, D, *target_hw)

        tps = self.encoder.temporal_patch_size
        pad_d = (tps - (D % tps)) % tps
        if pad_d > 0:
            x_resized = F.pad(x_resized, (0, 0, 0, 0, 0, pad_d))

        tokens = self.encoder.to_patch_emb(x_resized)   # (B, t', h', w', d)
        tokens = self._encode(tokens)
        return tokens.permute(0, 4, 1, 2, 3).contiguous()  # (B, dim, t', h', w')

    def encoder_forward(self, x):
        bottleneck = self._run_ctvit(x)
        # Pass raw input alongside the bottleneck so the trainable SPM in the
        # adapter can consume it.
        return [x, bottleneck]

    def adapter_forward(self, native, input_shape):
        x_in, bottleneck = native
        if self.adapter.pyramid_mode == "upsample":
            return self.adapter.upsample_neck(bottleneck, input_shape)
        D, H, W = input_shape
        fine = self.adapter.spm(x_in)                       # strides 1..8
        s16 = self.adapter.vit_proj(bottleneck)             # 512 ch, anisotropic
        target = (D // 16, H // 16, W // 16)
        if tuple(s16.shape[2:]) != target:
            s16 = F.interpolate(s16, size=target, mode="trilinear", align_corners=False)
        return [*fine, s16]

    def forward_features(self, x):
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])
