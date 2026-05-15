"""Quick smoke test: build the model from a config, run a dummy forward pass.

Doesn't touch the dataset — just verifies the encoder + head wire up correctly
and the output is the right shape.

    python -m scripts.verify_setup --config configs/models/voco_b.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from unified.utils import load_config  # noqa: E402
from unified.models import build_backbone, UnifiedSegHead, SegModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--load-weights", action="store_true",
                    help="actually load the pretrained checkpoint (default: skip, fast smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["model"]
    weights = mcfg.get("weights") if args.load_weights else None

    backbone = build_backbone(mcfg["name"], weights=weights, **mcfg.get("kwargs", {}))
    head = UnifiedSegHead(
        num_classes=cfg["head"]["num_classes"],
        feature_channels=cfg["head"]["feature_channels"],
        feature_strides=cfg["head"]["feature_strides"],
        decoder_channels=cfg["head"]["decoder_channels"],
        norm=cfg["head"]["norm"],
    )
    model = SegModel(backbone, head).to(args.device)

    x = torch.randn(args.batch, 1, args.patch, args.patch, args.patch, device=args.device)

    # Cheap structural check: assert pyramid contract.
    feats = backbone(x.to(args.device).contiguous()) if False else backbone.forward_features(x)
    backbone.assert_contract(x, feats)

    logits = model(x)
    expected = (args.batch, cfg["head"]["num_classes"], args.patch, args.patch, args.patch)
    assert tuple(logits.shape) == expected, \
        f"unexpected logits shape {tuple(logits.shape)} != {expected}"

    n_params = model.num_trainable_params()
    print(f"OK: {mcfg['name']}  params={n_params:,}  output_shape={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
