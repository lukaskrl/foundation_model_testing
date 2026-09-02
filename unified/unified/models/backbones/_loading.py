"""Uniform post-load verification for pretrained encoder checkpoints.

Every adapter loads its checkpoint with ``strict=False``, and it has to: each
upstream release carries keys this framework has no place for — segmentation
decoders, SSL projection heads, text towers, classification necks — so a strict
load would reject a perfectly good file. But ``strict=False`` also swallows the
failure that actually matters: a *key-layout* mismatch that leaves part (or all)
of the encoder at random init while the run is still labelled
``pretrained: true``.

That has bitten this framework before. See ``dino3d._assert_vit_loaded`` for the
``block_chunks`` incident, where the ViT was built expecting ``blocks.0.{0..23}``
while the checkpoint stored ``blocks.{0..3}.{global_idx}``, so 18 of 24
transformer blocks stayed randomly initialized and training proceeded on a
26 %-pretrained encoder.

Guard strength used to vary per adapter — from ``strict=True`` (merlin) down to
no check at all (ctclip, vista3d) — which meant the encoders most likely to fail
silently were ranked in the same table as the ones that could not. This module is
the single standard: **every parameter and persistent buffer of the constructed
encoder must be populated by the checkpoint.**

Measured against the real released checkpoints, all twelve encoders load 100 % of
their keys, so this costs no legitimate load:

    ctfm 83/83 · vista3d 83/83 · voco_b 134/134 · voco_h 134/134
    suprem_unet 104/104 · suprem_segresnet 33/33 · suprem_swinunetr 126/126
    sam_med3d 189/189 · dino3d 343/343 · merlin 934/934
    biomedparse 504/504 · ctclip 157/157

Unexpected keys are deliberately NOT an error: discarding a decoder / SSL head /
text tower is the normal case, and the counts are large and legitimate
(vista3d 60, suprem_segresnet 50, voco 33, dino3d 16). Only *missing* keys mean
something was left random.
"""
from __future__ import annotations

import torch.nn as nn


def assert_encoder_loaded(
    tag: str,
    module: nn.Module,
    result,
    *,
    benign_missing=(),
) -> None:
    """Raise unless ``result`` shows the checkpoint populated every key of ``module``.

    Parameters
    ----------
    tag : str
        Backbone name, used in the error message.
    module : nn.Module
        The encoder that was just loaded — its ``state_dict()`` defines the set
        of keys that had to be filled.
    result
        The ``_IncompatibleKeys`` named tuple returned by ``load_state_dict``.
    benign_missing : sequence of str
        Key *prefixes* that a released checkpoint may legitimately omit. Keep
        this empty unless a specific upstream release justifies an entry —
        every entry is a hole in the guard, which is the thing this module
        exists to close.
    """
    benign = tuple(benign_missing)
    missing = [k for k in result.missing_keys if not k.startswith(benign)]
    if not missing:
        return
    total = len(module.state_dict())
    loaded = total - len(missing)
    raise RuntimeError(
        f"{tag}: checkpoint populated {loaded}/{total} encoder keys "
        f"({100.0 * loaded / max(1, total):.1f}%) — {len(missing)} left at random "
        f"init. This is a key-layout mismatch (wrong prefix, wrong arch variant, "
        f"wrong file), not a benign partial load: a run started from here would "
        f"report `pretrained: true` while training on random weights. "
        f"Sample missing: {missing[:5]}"
    )
