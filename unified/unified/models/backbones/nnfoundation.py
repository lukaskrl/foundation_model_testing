"""nnFoundation ViT and CNN (arXiv 2609.26924; ``MIC-DKFZ/nnFoundation{ViT,CNN}`` on Hugging Face).

A matched pair: both were MAE-pretrained by nnssl on the same 2.1M CT/MRI/PET volumes, on 192^3
crops, z-score normalized per image over all voxels, not resampled, read with SimpleITK (array axes
z, y, x, so the encoder frame is SPL, as for CT-FM). The architectures come from
``dynamic_network_architectures`` (pip, 0.4.x) and are built exactly as nnssl builds them
(``nnssl/architectures/get_network_by_name.py`` and ``architecture_registry.py``).

* ``nnfoundation_vit``: Primus (EVA blocks with RoPE and an absolute position embedding), depth 40,
  dim 1056, 16 heads, 8^3 patch tokens, 674M parameters. It was pretrained as nnssl's ``EvaMAE``;
  only ``down_projection`` and ``eva`` (same names in ``Primus``) are kept, the MAE decoder is
  dropped. The model is built for the input canvas (``canvas``, default 96^3), which sets the RoPE
  reference grid to ``canvas / 8`` as nnssl does for a new patch size, and the absolute position
  embedding is resized from the pretraining grid (24^3) by trilinear interpolation, which is what
  nnFoundation does when the downstream patch size differs from 192^3.
* ``nnfoundation_cnn``: the ResEnc-L encoder (6 stages, 32-64-128-256-320-320 channels, strides
  1, 2, 4, 8, 16, 32; downsampling by stride-2 3x3x3 convs with padding 1, so cell i of a stride-s
  stage is centred on input voxel s * i). ``max_stride`` drops the coarser stages (default 16, the
  coarsest grid of CT-FM and dino3d; at 96^3 the stride-32 stage is only 3^3).

Only ``encoder_forward`` is implemented: the frozen probe (``unified/upsampler/encoders.py``) and
the pyramid-free I1/I2 interface read the encoder's native outputs. The 5-level contract of Arms
N/S/B/W is not supported.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_backbone

VIT_ARCH = dict(embed_dim=1056, eva_depth=40, eva_numheads=16, patch_embed_size=(8, 8, 8),
                init_values=0.1, scale_attn_inner=True)
VIT_PRETRAIN_CANVAS = (192, 192, 192)


def _network_weights(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return ck["network_weights"] if "network_weights" in ck else ck


class _Frozen(nn.Module):
    """Shared plumbing: the backbone is used only as a frozen feature extractor."""

    def forward(self, x):  # pragma: no cover - not part of the 5-level contract
        raise NotImplementedError(f"{type(self).__name__} supports encoder_forward only")

    def forward_features(self, x):  # pragma: no cover
        raise NotImplementedError(f"{type(self).__name__} supports encoder_forward only")


@register_backbone("nnfoundation_vit")
class NNFoundationViT(_Frozen):
    def __init__(self, weights: str = None, canvas: Sequence[int] = (96, 96, 96), **_):
        super().__init__()
        from dynamic_network_architectures.architectures.primus import Primus
        self.canvas = tuple(int(c) for c in canvas)
        ps = VIT_ARCH["patch_embed_size"]
        if any(c % p for c, p in zip(self.canvas, ps)):
            raise ValueError(f"canvas {self.canvas} is not a multiple of the patch size {ps}")
        self.grid = tuple(c // p for c, p in zip(self.canvas, ps))
        self.net = Primus(input_channels=1, num_classes=1, input_shape=self.canvas, **VIT_ARCH)
        self.net.up_projection = nn.Identity()          # pretraining decoder head, unused
        if weights:
            self.load_pretrained(weights)

    def load_pretrained(self, path: str):
        sd = _network_weights(path)
        keep = {k: v for k, v in sd.items() if k.startswith(("down_projection.", "eva."))}
        pe = keep.get("eva.pos_embed")
        if pe is not None:
            g0 = tuple(c // p for c, p in zip(VIT_PRETRAIN_CANVAS, VIT_ARCH["patch_embed_size"]))
            if pe.shape[1] != g0[0] * g0[1] * g0[2]:
                raise ValueError(f"eva.pos_embed has {pe.shape[1]} tokens, expected {g0}")
            if g0 != self.grid:
                C = pe.shape[-1]
                grid = pe.reshape(1, *g0, C).permute(0, 4, 1, 2, 3).float()   # (w h d) order
                grid = F.interpolate(grid, size=self.grid, mode="trilinear", align_corners=False)
                keep["eva.pos_embed"] = grid.permute(0, 2, 3, 4, 1).reshape(1, -1, C).to(pe.dtype)
        missing, unexpected = self.net.load_state_dict(keep, strict=False)
        missing = [k for k in missing if k.startswith(("down_projection.", "eva."))]
        if missing or unexpected:
            raise RuntimeError(f"nnFoundation ViT weights: missing {missing[:5]}, "
                               f"unexpected {unexpected[:5]}")

    def encoder_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if tuple(x.shape[2:]) != self.canvas:
            raise ValueError(f"input {tuple(x.shape[2:])} != canvas {self.canvas}: the RoPE "
                             "reference grid and the position embedding are built for the canvas")
        t = self.net.down_projection(x)                  # (B, C, w, h, d), stride 8
        B, C, W, H, D = t.shape
        t = t.flatten(2).transpose(1, 2)                 # b (w h d) c
        t, _ = self.net.eva(t)                           # final LayerNorm included
        return [t.transpose(1, 2).reshape(B, C, W, H, D)]


@register_backbone("nnfoundation_cnn")
class NNFoundationCNN(_Frozen):
    def __init__(self, weights: str = None, max_stride: int = 16, **_):
        super().__init__()
        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
        n = 6
        net = ResidualEncoderUNet(            # == nnssl architecture_registry.get_res_enc_l
            input_channels=1, n_stages=n, features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d, kernel_sizes=[[3, 3, 3]] * n,
            strides=[[1, 1, 1]] + [[2, 2, 2]] * (n - 1), n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            num_classes=1, n_conv_per_stage_decoder=[1] * (n - 1), conv_bias=True,
            norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True}, deep_supervision=False)
        if weights:
            sd = _network_weights(weights)
            enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
            missing, unexpected = net.encoder.load_state_dict(enc, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"nnFoundation CNN weights: missing {missing[:5]}, "
                                   f"unexpected {unexpected[:5]}")
        self.encoder = net.encoder                       # decoder dropped
        self.strides = [2 ** i for i in range(n)]
        self.n_keep = sum(s <= int(max_stride) for s in self.strides)

    def encoder_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(self.encoder(x))[: self.n_keep]      # finest first, strides 1 .. max_stride
