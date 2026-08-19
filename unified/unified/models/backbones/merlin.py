"""Merlin backbone adapter (Stanford MIMI, Nature 2026).

Merlin is a 3D CT vision-language model: an **inflated-2D (I3D) ResNet-152**
image tower trained against structured EHR codes and radiology reports through a
Clinical-Longformer text tower. For this framework it is the second
report-supervised encoder alongside ``ctclip`` — and, unlike ``ctclip``, it is
*hierarchical*, so "report-supervised" stops being a synonym for "columnar ViT".

Native pyramid (measured at a 96³ input, reported as (C, D, H, W)):

===========  ==================  ================  =============
tap          shape               in-plane stride   depth stride
===========  ==================  ================  =============
``conv1``    (64,  96, 48, 48)   2                 1
``layer1``   (256, 48, 24, 24)   4                 2
``layer2``   (512, 24, 12, 12)   8                 4
``layer3``   (1024, 12, 6, 6)    16                8
``layer4``   (2048, 6,  3,  3)   32                16
===========  ==================  ================  =============

**The depth stride is exactly half the in-plane stride at every level.** That is
not an accident: ``conv1`` is inflated with ``time_stride=1`` while its spatial
stride is 2, and Merlin was pretrained at 1.5 × 1.5 × **3.0** mm. Two anisotropies
cancel, so at its own spacing one Merlin feature voxel is physically isotropic.
This framework resamples to 1.5 mm **isotropic**, so the two no longer cancel and
the mismatch has to be resolved explicitly — hence ``input_mode``.

``input_mode``
    ``"native_spacing"`` (default)
        Average-pool the input depth by 2 before the encoder, so it sees the
        1.5 × 1.5 × 3.0 mm geometry it was pretrained on. Every native level then
        lands *exactly* on a cubic contract stride with no per-level resampling.
        Full-resolution depth still reaches the decoder through the stride-1 stem.
    ``"isotropic"``
        Run the encoder on the framework's 1.5 mm isotropic patch and average-pool
        each level's depth by 2 to make it cubic. The encoder sees depth at twice
        the resolution it was pretrained for.

Both modes are parameter-identical (pooling has no parameters) and differ only in
*where* the factor-2 depth reduction happens, so they are a clean single-variable
A/B — the same relationship ``ctclip``'s ``upsample`` / ``layerwise`` /
``multiscale`` modes have to each other.

Contract sourcing (identical in both modes):

* stride 1  ← ``ConvStem`` on the raw input. No pretrained feature exists at
  stride 1, and we never upsample pretrained features. Same component and
  parameter count as the Swin family's stem, so the added capacity is comparable.
* strides 2, 4, 8, 16 ← ``conv1``, ``layer1``, ``layer2``, ``layer3``, one 1×1
  ``ChannelNeck`` each.
* ``layer4`` is **discarded**: its in-plane stride is 32, and reaching stride 16
  from it would require upsampling a pretrained feature. Exactly the trade the
  Swin family makes. Note this is the tap the contrastive objective acted on
  (``layer4`` → ``avgpool`` → ``contrastive_head``), so the coarsest level the
  decoder sees is one stage shallower than the one the VLM loss shaped — worth
  stating in any write-up.

``layer4``, ``avgpool``, ``classifier`` and ``contrastive_head`` are kept as
modules (so the checkpoint loads strictly, and so the CT-RATE classification
track can still pool the tap the loss shaped) but are never executed by
``encoder_forward``, so they cost idle parameters and no compute.

Tensor layout, and the orientation it depends on. Merlin's own
``I3ResNet.forward`` takes ``(B, C, H, W, D)`` and permutes to put depth on dim 2
before the inflated convs, which treat dim 2 as the "time" axis. This adapter
traverses the layers itself and skips that permute, consuming dim 2 of the
``(B, 1, d0, d1, d2)`` tensor the framework hands it as the time axis.

**That is only correct if d0 is the axial slice axis, which is a property of
``model.preprocessing.axcodes``, not of this file.** Upstream reorients to RAS
and then permutes, so its time axis is S and its in-plane axes are R, A. Under
``axcodes: RAS`` this framework's d0 is R, so the time axis would be left-right
— the pretraining geometry rotated 90°, with ``native_spacing``'s depth pooling
producing 3.0 mm across the patient rather than along the scan axis. The configs
therefore set ``axcodes: SRA``, which lands S on d0 and keeps in-plane (R, A):
the same reorientation upstream achieves with its forward-time permute. Any new
Merlin config must do the same. We also skip
upstream's outer ``checkpoint.checkpoint`` calls: ``Bottleneck3d`` already
checkpoints internally whenever its input requires grad, and wrapping again would
nest recomputation.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from ._neck import ChannelNeck, ConvStem

MERLIN_REPO = Path("/store/home/skrljl/projects/foundation_models/Merlin")

# Key prefix the released full-model checkpoint uses for the image tower.
TOWER_PREFIX = "encode_image.i3_resnet."

INPUT_MODES = ("native_spacing", "isotropic")


def _load_leaf_module(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {mod_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_i3res():
    """Import Merlin's ``i3res`` without pulling in the text tower's deps.

    ``merlin/models/__init__.py`` imports ``load.py`` → ``build.py`` →
    ``transformers`` + ``nltk``, neither of which is installed here (installing
    transformers would downgrade ``huggingface_hub``). ``i3res.py`` itself needs
    only torch, but it does ``from merlin.models import inflate``, which would
    execute that ``__init__``. So register stub packages in ``sys.modules`` and
    load the two leaf files by path.
    """
    models_dir = MERLIN_REPO / "merlin" / "models"
    if not models_dir.exists():
        raise ImportError(
            f"Merlin source not found at {MERLIN_REPO}. Clone it:\n"
            "    git clone --depth 1 https://github.com/StanfordMIMI/Merlin.git "
            f"{MERLIN_REPO}"
        )
    if "merlin.models.i3res" not in sys.modules:
        for name in ("merlin", "merlin.models"):
            if name not in sys.modules:
                stub = types.ModuleType(name)
                stub.__path__ = []          # mark as a package
                sys.modules[name] = stub
        inflate = _load_leaf_module("merlin.models.inflate", models_dir / "inflate.py")
        sys.modules["merlin.models"].inflate = inflate
        _load_leaf_module("merlin.models.i3res", models_dir / "i3res.py")
    return sys.modules["merlin.models.i3res"]


def _extract_tower_state(state, source: str):
    """Accept either the full CLIP checkpoint or a pre-slimmed image tower.

    Raises naming the prefixes actually present, rather than letting a wrong file
    reach ``load_state_dict`` and surface as a wall of missing keys.
    """
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith(TOWER_PREFIX) for k in state):
        return {k[len(TOWER_PREFIX):]: v for k, v in state.items()
                if k.startswith(TOWER_PREFIX)}
    tower = dict(state)
    # A slimmed tower always carries the inflated stem at the top level.
    if "conv1.weight" not in tower:
        found = sorted({k.split(".")[0] for k in tower})[:8]
        raise RuntimeError(
            f"Merlin: {source} does not look like an I3ResNet image tower. Expected "
            f"either bare I3ResNet keys (starting 'conv1.weight') or keys prefixed "
            f"{TOWER_PREFIX!r}; found top-level groups {found}."
        )
    return tower


def _pool_depth_by_2(t: torch.Tensor) -> torch.Tensor:
    """Average-pool dim 2 (depth) by a factor of 2, leaving H and W alone."""
    return F.avg_pool3d(t, kernel_size=(2, 1, 1), stride=(2, 1, 1))


def _resize_to(t: torch.Tensor, size) -> torch.Tensor:
    """Safety net for input shapes that are not cleanly divisible.

    A no-op at the locked 96³ patch (and at 128³): every level already lands on
    its contract shape. It exists so an odd best-effort FOV degrades to one
    trilinear resample instead of a contract assertion failure.
    """
    if tuple(t.shape[2:]) == tuple(size):
        return t
    return F.interpolate(t, size=tuple(size), mode="trilinear", align_corners=False)


class _MerlinAdapter(nn.Module):
    """Trainable post-encoder modules: one stride-1 stem + four 1×1 projections."""

    def __init__(self, native_channels, contract_channels, stem_hidden: int = 32):
        super().__init__()
        if len(native_channels) != len(contract_channels) - 1:
            raise ValueError(
                f"Merlin adapter expects {len(contract_channels) - 1} native levels, "
                f"got {len(native_channels)}"
            )
        self.stem = ConvStem(out_ch=contract_channels[0], in_ch=1,
                             hidden_ch=stem_hidden)
        self.ch_necks = nn.ModuleList(
            [ChannelNeck(ic, oc)
             for ic, oc in zip(native_channels, contract_channels[1:])]
        )


@register_backbone("merlin")
class MerlinBackbone(BackboneInterface):
    """I3D-ResNet-152 image tower of Merlin, lifted onto the 5-level contract."""

    # conv1, layer1, layer2, layer3 — layer4 is not used (see module docstring).
    NATIVE_CHANNELS = (64, 256, 512, 1024)

    def __init__(
        self,
        weights: str = None,
        input_mode: str = "native_spacing",
        class_nb: int = 1692,
        stem_hidden: int = 32,
    ):
        super().__init__()
        if input_mode not in INPUT_MODES:
            raise ValueError(
                f"unknown input_mode {input_mode!r}; expected one of {INPUT_MODES}"
            )
        self.input_mode = input_mode

        import torchvision                                       # lazy
        i3res = _import_i3res()

        # weights=None on the 2D source: every parameter is overwritten by the
        # Merlin checkpoint below, and this avoids an ImageNet download. When
        # `weights` is None (the scratch twin) the random init IS the control.
        resnet2d = torchvision.models.resnet152(weights=None)
        self.encoder = i3res.I3ResNet(
            copy.deepcopy(resnet2d),
            class_nb=class_nb,
            conv_class=True,
            return_skips=False,     # we traverse the layers ourselves
        )

        if weights:
            state = torch.load(weights, map_location="cpu", weights_only=False)
            tower = _extract_tower_state(state, source=str(weights))
            # strict=True on purpose. A silent partial load here would look like a
            # slightly-worse Dice rather than a bug (see docs/ADDING_A_MODEL.md).
            self.encoder.load_state_dict(tower, strict=True)

        self.adapter = _MerlinAdapter(
            native_channels=self.NATIVE_CHANNELS,
            contract_channels=self.EXPECTED_CHANNELS,
            stem_hidden=stem_hidden,
        )

    def encoder_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return ``[raw_input, conv1, layer1, layer2, layer3]``.

        ``x`` is ``(B, 1, D, H, W)`` — depth on dim 2, which is the axis the
        inflated convs treat as time, so no permute is needed. This holds because
        the configs orient to ``SRA`` (d0 = S = the axial slice axis); see the
        module docstring for why RAS would silently rotate the geometry 90°.
        Merlin's own
        forward also replicates the single CT channel to 3 (its ``conv1`` came
        from a 3-channel 2D ResNet); we do the same here.
        """
        enc = self.encoder
        h = x if self.input_mode == "isotropic" else _pool_depth_by_2(x)
        h = h.expand(-1, 3, -1, -1, -1)          # 1 → 3 channels, no copy

        h = enc.relu(enc.bn1(enc.conv1(h)))
        t_conv1 = h
        h = enc.maxpool(h)
        h = enc.layer1(h)
        t_layer1 = h
        h = enc.layer2(h)
        t_layer2 = h
        h = enc.layer3(h)
        t_layer3 = h
        # enc.layer4 deliberately not run: its in-plane stride is 32.
        return [x, t_conv1, t_layer1, t_layer2, t_layer3]

    def adapter_forward(self, native, input_shape) -> List[torch.Tensor]:
        x_in, *taps = native
        d, h, w = (int(s) for s in input_shape)

        s1 = _resize_to(self.adapter.stem(x_in), (d, h, w))

        out = [s1]
        for level, (tap, neck) in enumerate(zip(taps, self.adapter.ch_necks), start=1):
            stride = self.EXPECTED_STRIDES[level]
            if self.input_mode == "isotropic":
                # Native depth stride is half the in-plane stride; pool it down.
                tap = _pool_depth_by_2(tap)
            feat = neck(tap)
            out.append(_resize_to(feat, (d // stride, h // stride, w // stride)))
        return out

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])
