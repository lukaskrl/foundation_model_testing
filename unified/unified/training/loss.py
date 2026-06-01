"""Loss factory.

When ``head.deep_supervision`` is true, wraps DiceCELoss with MONAI's
DeepSupervisionLoss — model returns a list of multi-scale predictions during
training, the wrapper resamples the label to each prediction's resolution and
sums weighted losses.
"""
from __future__ import annotations
from typing import List, Union

import torch
import torch.nn as nn


def build_loss(cfg):
    name = cfg["train"]["loss"]["name"]
    if name != "dice_ce":
        raise ValueError(
            f"loss {name!r} not supported — base.yaml fixes loss=dice_ce for fair comparison"
        )
    from monai.losses import DiceCELoss
    p = cfg["train"]["loss"]
    base = DiceCELoss(
        include_background=p.get("include_background", False),
        softmax=p.get("softmax", True),
        to_onehot_y=p.get("to_onehot_y", True),
    )

    if cfg["head"].get("deep_supervision", False):
        return _DeepSupervisionWrapper(base, cfg["head"].get(
            "ds_weights", [1.0, 0.5, 0.25, 0.125],
        ))
    return _SingleLogitWrapper(base)


class _SingleLogitWrapper(nn.Module):
    """Tolerates a 1-element list from the head (smoke runs where DS is off but
    the head still returns a list during training)."""
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, pred, target):
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        return self.base(pred, target)


class _DeepSupervisionWrapper(nn.Module):
    """Multi-resolution Dice-CE.

    Accepts either a list of predictions (training-time, with deep supervision)
    or a single tensor (eval-time, sliding-window inference). For a single
    tensor it behaves like the base loss.
    """

    def __init__(self, base, weights):
        super().__init__()
        self.base = base
        self.weights = list(weights)

    def forward(
        self,
        pred: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(pred, torch.Tensor):
            return self.base(pred, target)
        # pred is a list; resample target to each prediction's spatial size.
        # target shape: (B, 1, D, H, W) of int labels; resample with nearest.
        n = min(len(pred), len(self.weights))
        total = pred[0].new_zeros(())
        for i in range(n):
            p = pred[i]
            if p.shape[2:] == target.shape[2:]:
                t = target
            else:
                t = torch.nn.functional.interpolate(
                    target.float(), size=p.shape[2:], mode="nearest"
                ).to(target.dtype)
            total = total + self.weights[i] * self.base(p, t)
        return total
