"""Lightweight smoke test: builds backbone+head, runs forward/backward on a
random tensor. No dataset access, no transforms. Verifies that:

  * The adapter constructs
  * Pretrained weights load (unless --no-weights)
  * The pyramid-contract assertion passes for the input patch size
  * Loss is finite and backward works
  * Optimizer step doesn't NaN

Usage:
    python -m scripts.smoke_compute --config configs/models/ctfm.yaml
    python -m scripts.smoke_compute --config configs/models/suprem_unet.yaml --patch 64
"""
from __future__ import annotations
import argparse
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from unified.utils import load_config, setup_logging, get_logger  # noqa: E402
from unified.models import build_backbone, build_head, SegModel  # noqa: E402
from unified.training import build_loss, build_optimizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--patch", type=int, default=64,
                    help="cubic patch side length (default 64 — small to fit on CPU)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="force freeze_backbone=True (overrides config)")
    args = ap.parse_args()

    setup_logging(None)
    log = get_logger("smoke_compute")

    cfg = load_config(args.config)
    name = cfg["model"]["name"]
    log.info("=== smoke_compute: %s ===", name)
    log.info("config=%s patch=%d batch=%d steps=%d weights=%s",
             args.config, args.patch, args.batch, args.steps, not args.no_weights)

    # Override patch size (some models can't run at the default 96)
    cfg["data"]["patch_size"] = [args.patch] * 3

    # Build backbone (loads pretrained weights here).
    t0 = time.time()
    mcfg = cfg["model"]
    weights = None if args.no_weights else mcfg.get("weights")
    backbone = build_backbone(mcfg["name"], weights=weights, **mcfg.get("kwargs", {}))
    log.info("backbone built in %.2fs (%d params)",
             time.time() - t0,
             sum(p.numel() for p in backbone.parameters()))

    head = build_head(
        cfg["head"].get("name", "unified_seg_head"),
        num_classes=cfg["head"]["num_classes"],
        feature_channels=cfg["head"]["feature_channels"],
        feature_strides=cfg["head"]["feature_strides"],
        decoder_channels=cfg["head"]["decoder_channels"],
        norm=cfg["head"]["norm"],
        deep_supervision=cfg["head"].get("deep_supervision", False),
    )
    freeze = args.freeze_backbone or bool(mcfg.get("freeze_backbone", False))
    model = SegModel(backbone, head, freeze_backbone=freeze)
    log.info("model total params: %d (trainable: %d, freeze_backbone=%s)",
             model.num_total_params(),
             model.num_trainable_params(),
             model.freeze_backbone)

    if args.device == "auto":
        target = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target = args.device

    try:
        model.to(target)
        device = target
    except RuntimeError as e:
        log.warning("%s: device=%s failed (%s); falling back to CPU", name, target, e)
        model.to("cpu")
        device = "cpu"
    log.info("device: %s", device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(cfg, trainable_params)
    loss_fn = build_loss(cfg)

    # Random data shaped like a single batch.
    p = args.patch
    nc = cfg["data"]["num_classes"]
    image = torch.randn(args.batch, 1, p, p, p, device=device)
    label = torch.randint(0, nc, (args.batch, 1, p, p, p), device=device, dtype=torch.long)

    model.train()
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        try:
            t = time.time()
            logits = model(image)
            if isinstance(logits, list):
                shapes = [tuple(l.shape) for l in logits]
            else:
                shapes = tuple(logits.shape)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
            dt = time.time() - t
            log.info("step %d loss=%.4f shapes=%s dt=%.2fs",
                     step, loss.item(), shapes, dt)
            losses.append(loss.item())
        except torch.cuda.OutOfMemoryError:
            log.warning("CUDA OOM at step %d, retrying on CPU", step)
            torch.cuda.empty_cache()
            model.to("cpu")
            device = "cpu"
            image = image.to("cpu")
            label = label.to("cpu")
            optimizer = build_optimizer(cfg, model.parameters())

    if losses and all(l == l and abs(l) < 1e6 for l in losses):
        log.info("SMOKE OK: %s — losses %s", name, [f"{l:.4f}" for l in losses])
        return 0
    log.error("SMOKE FAIL: %s — losses %s", name, losses)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
