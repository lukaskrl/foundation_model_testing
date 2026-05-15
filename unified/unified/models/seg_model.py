"""The training-time model: pretrained backbone + uniform decoder."""
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn

from .head import UnifiedSegHead


class BackboneInterface(nn.Module):
    """Contract every foundation-model adapter must implement.

    Subclasses must return a list of EXACTLY 4 feature tensors with shapes:

        feat[0]: (B,  64, D/4,  H/4,  W/4)
        feat[1]: (B, 128, D/8,  H/8,  W/8)
        feat[2]: (B, 256, D/16, H/16, W/16)
        feat[3]: (B, 512, D/32, H/32, W/32)

    Where (D, H, W) is the input patch size and stride values above refer to
    spatial down-sampling relative to the input.
    """

    EXPECTED_CHANNELS = (64, 128, 256, 512)
    EXPECTED_STRIDES = (4, 8, 16, 32)
    NUM_LEVELS = 4

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def assert_contract(self, x: torch.Tensor, feats: List[torch.Tensor]) -> None:
        """Cheap structural check — call from forward() while debugging."""
        if len(feats) != self.NUM_LEVELS:
            raise ValueError(
                f"backbone returned {len(feats)} feature maps, expected {self.NUM_LEVELS}"
            )
        _, _, D, H, W = x.shape
        for i, (f, c_exp, s_exp) in enumerate(
            zip(feats, self.EXPECTED_CHANNELS, self.EXPECTED_STRIDES)
        ):
            b, c, d, h, w = f.shape
            if c != c_exp:
                raise ValueError(
                    f"level {i}: expected {c_exp} channels, got {c}"
                )
            if (d, h, w) != (D // s_exp, H // s_exp, W // s_exp):
                raise ValueError(
                    f"level {i}: expected stride {s_exp} "
                    f"(shape {(D//s_exp, H//s_exp, W//s_exp)}), got {(d,h,w)}"
                )


class SegModel(nn.Module):
    """Backbone + uniform UNETR-style decoder."""

    def __init__(self, backbone: BackboneInterface, head: UnifiedSegHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        return self.head(x, feats)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
