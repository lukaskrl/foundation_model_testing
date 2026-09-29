"""Multi-encoder backbone: several frozen encoders sharing ONE head.

Used to pretrain a head across encoders (the H2 pilot: does a head trained with
several encoders accept an unseen encoder better than a head trained with one?).

Each member is a complete contract backbone with its OWN adapter; only the head
downstream is shared. During training the members alternate per forward call
(= per micro-batch), so with ``grad_accum_steps: 2`` and two members every
optimizer step averages one micro-batch from each encoder and both adapters are
updated every step. Per-call alternation is deliberate: coarser alternation
(per epoch / per block) lets the shared head drift toward whichever encoder it
saw last and carries one encoder's AdamW momentum into the other's phase.

In eval mode a single fixed member (``eval_member``) is used, so sliding-window
inference never mixes encoders inside one case.

Constraint: members are fed the SAME preprocessed input (one data pipeline), so
they must share preprocessing (HU window / normalization and axcodes). The two
SuPreM encoders do; mixing e.g. a SuPreM encoder with dino3d does not.

Config::

    model:
      name: multi
      kwargs:
        eval_member: 0
        members:
          - {name: suprem_unet, weights: /path/a.pth, kwargs: {...}}
          - {name: suprem_swinunetr, weights: /path/b.pt, kwargs: {...}}
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from ..registry import build_backbone, register_backbone
from ..seg_model import BackboneInterface


@register_backbone("multi")
class MultiBackbone(BackboneInterface):
    def __init__(self, members: Sequence[dict], eval_member: int = 0,
                 pretrained: bool = True, **_ignored):
        super().__init__()
        if len(members) < 2:
            raise ValueError("multi backbone needs at least two members")
        built = []
        for spec in members:
            spec = dict(spec)
            name = spec.pop("name")
            weights = spec.pop("weights", None) if pretrained else None
            spec.pop("weights", None)
            built.append(build_backbone(name, weights=weights, **spec.pop("kwargs", {})))
            if spec:
                raise ValueError(f"unknown keys in member spec {name!r}: {sorted(spec)}")
        self.members = nn.ModuleList(built)
        self.member_names = [m["name"] for m in members]
        if not 0 <= int(eval_member) < len(self.members):
            raise ValueError(f"eval_member {eval_member} out of range")
        self.eval_member = int(eval_member)
        self._next = 0          # round-robin pointer for training calls
        self._active = 0        # member chosen by the last encoder_forward
        # Deliberately NO self.adapter: each member keeps its own, and the two
        # overrides below route freezing / train-mode to every member.

    # ------------------------------------------------------------- freezing
    def freeze_encoder(self) -> None:
        for m in self.members:
            m.freeze_encoder()

    def trainable_modules(self) -> List[nn.Module]:
        mods: List[nn.Module] = []
        for m in self.members:
            get = getattr(m, "trainable_modules", None)
            mods.extend(get() if get is not None else [getattr(m, "adapter", None)])
        return [x for x in mods if x is not None]

    # -------------------------------------------------------------- routing
    def _training_call(self) -> bool:
        # SegModel keeps a frozen backbone in eval() even while training, so the
        # wrapper's own .training flag is always False there. The adapters are
        # the modules SegModel switches with the run's mode — read those.
        mods = self.trainable_modules()
        return any(mod.training for mod in mods) if mods else self.training

    def _pick(self) -> int:
        if self._training_call():
            idx = self._next
            self._next = (self._next + 1) % len(self.members)
        else:
            idx = self.eval_member
        self._active = idx
        return idx

    def encoder_forward(self, x: torch.Tensor):
        m = self.members[self._pick()]
        if not hasattr(m, "encoder_forward"):
            raise TypeError(f"member {type(m).__name__} lacks the encoder/adapter split")
        return m.encoder_forward(x)

    def adapter_forward(self, native, input_shape):
        return self.members[self._active].adapter_forward(native, input_shape)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.members[self._pick()].forward_features(x)

    def set_eval_member(self, idx: Optional[int]) -> None:
        if idx is not None:
            self.eval_member = int(idx)
