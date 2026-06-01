"""The training-time model: pretrained backbone + uniform decoder.

Pyramid contract (a stable, documented interface that both adapters and heads
depend on):

    feats = [f_0, f_1, f_2, f_3, f_4]   # finest first

    f_0: (B,  32, D,    H,    W)        stride 1
    f_1: (B,  64, D/2,  H/2,  W/2)      stride 2
    f_2: (B, 128, D/4,  H/4,  W/4)      stride 4
    f_3: (B, 256, D/8,  H/8,  W/8)      stride 8
    f_4: (B, 512, D/16, H/16, W/16)     stride 16

(D, H, W) is the input patch size; strides are spatial down-sampling relative
to the input. Ordering is finest-first to match the deep-supervision
``ds_weights = [1.0, 0.5, 0.25, 0.125]`` convention (predictions at strides
1, 2, 4, 8 — finest first).

Adapter / encoder separation (required of every backbone):
- ``encoder``: the pretrained, frozen feature extractor. Each backbone keeps
  its native encoder under any attribute name (``encoder``, ``vit``,
  ``swinViT``, ``unet``, ``net``, ...).
- ``self.adapter``: a single ``nn.Module`` containing the trainable channel
  projections and per-backbone synthesizers that lift the native features
  onto the contract pyramid.

``SegModel.freeze_backbone=True`` freezes every parameter inside the backbone
**except** those reachable through ``backbone.adapter``. This is the only
way the channel-adapter 1×1 convs (and any SPM / synthesizer modules) train.
"""
from __future__ import annotations
from typing import List, Union

import torch
import torch.nn as nn


class BackboneInterface(nn.Module):
    """Contract every foundation-model adapter must implement.

    Subclasses must:
      1. Provide ``forward_features(x)`` returning ``NUM_LEVELS`` tensors that
         satisfy ``EXPECTED_STRIDES`` / ``EXPECTED_CHANNELS``.
      2. Register all trainable post-encoder modules under ``self.adapter``
         (an ``nn.Module``). Everything else in the backbone is treated as
         the frozen encoder.
    """

    EXPECTED_STRIDES = (1, 2, 4, 8, 16)
    EXPECTED_CHANNELS = (32, 64, 128, 256, 512)
    NUM_LEVELS = 5

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

    def freeze_encoder(self) -> None:
        """Freeze all backbone params EXCEPT those in ``self.adapter``.

        Adapter modules — the trainable channel projections / synthesizers
        / SPM — stay learnable so the head can be paired with frozen
        pretrained features through a learnable interface.
        """
        adapter = getattr(self, "adapter", None)
        if adapter is None:
            # Sub-classes that genuinely have no adapter (e.g. dev stubs) can
            # opt out by leaving ``self.adapter`` unset; then the whole
            # backbone is frozen, matching the pre-refactor behavior.
            for p in self.parameters():
                p.requires_grad_(False)
            return
        adapter_param_ids = {id(p) for p in adapter.parameters()}
        for p in self.parameters():
            if id(p) not in adapter_param_ids:
                p.requires_grad_(False)


class SegModel(nn.Module):
    """Backbone + uniform decoder.

    ``freeze_backbone=True`` freezes every parameter inside the backbone
    except those under ``backbone.adapter``. The pretrained encoder runs
    under ``torch.no_grad`` and its outputs are detached before the trainable
    adapter sees them; the adapter and head receive gradients.
    """

    def __init__(
        self,
        backbone: BackboneInterface,
        head: nn.Module,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.freeze_encoder()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.freeze_backbone:
            # Encoder stays in eval (Dropout / running-stat tracking off);
            # adapter stays in train mode so its norms behave correctly.
            self.backbone.eval()
            adapter = getattr(self.backbone, "adapter", None)
            if adapter is not None:
                adapter.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.freeze_backbone:
            feats = self._forward_frozen(x)
        else:
            feats = self.backbone.forward_features(x)
        return self.head(x, feats)

    def _forward_frozen(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run the encoder under no_grad, then the trainable adapter.

        Adapters that follow the encoder/adapter split provide
        ``encoder_forward`` + ``adapter_forward``. Adapters that don't (legacy
        / stub backbones) fall back to ``forward_features`` under no_grad —
        that path remains fully-frozen, matching the old behaviour.
        """
        encoder_forward = getattr(self.backbone, "encoder_forward", None)
        adapter_forward = getattr(self.backbone, "adapter_forward", None)
        if encoder_forward is None or adapter_forward is None:
            with torch.no_grad():
                feats = self.backbone.forward_features(x)
            return [f.detach() for f in feats]
        with torch.no_grad():
            native = encoder_forward(x)
        native = [t.detach() for t in native]
        return adapter_forward(native, x.shape[2:])

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
