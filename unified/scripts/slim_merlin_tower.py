"""Slice Merlin's I3D-ResNet image tower out of the released CLIP checkpoint.

The HuggingFace release (``stanfordmimi/Merlin``) ships one 1.01 GB checkpoint
holding both towers; the segmentation adapter only ever uses the image side
(keys prefixed ``encode_image.i3_resnet.``). Writing the slimmed tower once
keeps every run from paging in the Clinical-Longformer weights it discards.

    python -m scripts.slim_merlin_tower          # uses the default weights root

The adapter accepts either file, so this is an optimization, not a requirement.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from unified.models.backbones.merlin import TOWER_PREFIX  # noqa: E402
from unified.utils.paths import weights as weights_path  # noqa: E402

FULL_CKPT = "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(weights_path("Merlin", FULL_CKPT)))
    ap.add_argument("--dst", default=str(weights_path("Merlin", "merlin_i3resnet_image_tower.pt")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists() and not args.force:
        print(f"{dst} exists — nothing to do (--force to rewrite)")
        return
    if not src.exists():
        sys.exit(f"{src}: released Merlin checkpoint not found "
                 f"(run scripts/download_weights.sh merlin)")

    state = torch.load(src, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    tower = {k[len(TOWER_PREFIX):]: v for k, v in state.items()
             if k.startswith(TOWER_PREFIX)}
    if not tower:
        found = sorted({k.split(".")[0] for k in state})[:8]
        sys.exit(f"{src}: no keys prefixed {TOWER_PREFIX!r}; top-level groups {found}")
    if "conv1.weight" not in tower:
        sys.exit(f"{src}: tower is missing 'conv1.weight' — wrong checkpoint?")

    torch.save(tower, dst)
    n = sum(v.numel() for v in tower.values() if hasattr(v, "numel"))
    print(f"wrote {dst}  ({len(tower)} tensors, {n/1e6:.1f} M params, "
          f"{dst.stat().st_size/2**20:.0f} MiB from {src.stat().st_size/2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
