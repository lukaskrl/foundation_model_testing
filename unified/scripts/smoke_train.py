"""Per-model smoke training: 3 subjects, N steps, prints loss per step.

Use this to verify a backbone wires up end-to-end (forward + backward + optimizer
step) without committing to a real training run. NOT for fair comparison —
just for catching wiring bugs.

    python -m scripts.smoke_train --config configs/models/voco_b.yaml --steps 5

Falls back to CPU if CUDA OOMs.
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
from unified.data import (
    TotalSegmentatorDataset, load_classes, build_train_transforms,
)  # noqa: E402
from unified.models import build_backbone, UnifiedSegHead, SegModel  # noqa: E402
from unified.training import build_loss, build_optimizer  # noqa: E402


def _read_split(path: Path):
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--patch-override", type=int, default=None,
                    help="override patch size (some models need this, e.g. ct-clip needs >=160)")
    ap.add_argument("--no-weights", action="store_true",
                    help="skip loading pretrained weights (faster smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.patch_override is not None:
        cfg["data"]["patch_size"] = [args.patch_override] * 3
        cfg["eval"]["sliding_window"]["roi_size"] = [args.patch_override] * 3
    setup_logging(None)
    log = get_logger("smoke")

    name = cfg["model"]["name"]
    log.info("smoke training %s, patch=%s", name, cfg["data"]["patch_size"])

    # Tiny dataset.
    classes = load_classes()
    ids = _read_split(REPO / "unified" / "data" / "splits" / "train.txt")[: args.subjects]
    raw = TotalSegmentatorDataset(cfg["data"]["dataset_root"], ids, classes)
    tf = build_train_transforms(cfg)

    class Composed(torch.utils.data.Dataset):
        def __init__(self, base, t):
            self.base, self.t = base, t

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            return self.t(self.base[i])

    # MONAI random-crop transforms return a list of patches per sample; we use
    # MONAI's DataLoader so it collates lists correctly.
    from monai.data import DataLoader
    loader = DataLoader(
        Composed(raw, tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
    )

    # Build model.
    mcfg = cfg["model"]
    weights = None if args.no_weights else mcfg.get("weights")
    backbone = build_backbone(mcfg["name"], weights=weights, **mcfg.get("kwargs", {}))
    head = UnifiedSegHead(
        num_classes=cfg["head"]["num_classes"],
        feature_channels=cfg["head"]["feature_channels"],
        feature_strides=cfg["head"]["feature_strides"],
        decoder_channels=cfg["head"]["decoder_channels"],
        norm=cfg["head"]["norm"],
    )
    model = SegModel(backbone, head)
    log.info("trainable params: %d", model.num_trainable_params())

    def _try_device(device):
        model.to(device)
        optimizer = build_optimizer(cfg, model.parameters())
        loss_fn = build_loss(cfg)
        return device, optimizer, loss_fn

    if args.device == "auto":
        target = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target = args.device

    device, optimizer, loss_fn = _try_device(target)
    log.info("device: %s", device)

    model.train()
    step = 0
    losses = []
    for epoch in range(100):  # loop until we hit args.steps
        for batch in loader:
            if step >= args.steps:
                break
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            t0 = time.time()
            try:
                logits = model(image)
                loss = loss_fn(logits, label)
                loss.backward()
                optimizer.step()
            except torch.cuda.OutOfMemoryError:
                if device == "cuda":
                    log.warning("CUDA OOM at step %d; falling back to CPU", step)
                    torch.cuda.empty_cache()
                    device, optimizer, loss_fn = _try_device("cpu")
                    continue
                raise
            dt = time.time() - t0
            log.info(
                "step %d loss=%.4f image=%s label=%s dt=%.2fs",
                step, loss.item(), tuple(image.shape), tuple(label.shape), dt
            )
            losses.append(loss.item())
            step += 1
        if step >= args.steps:
            break

    if losses and not any(l != l for l in losses):  # no NaN
        log.info("SMOKE OK: %s — final loss %.4f (started at %.4f)",
                 name, losses[-1], losses[0])
        return 0
    else:
        log.error("SMOKE FAIL: %s — losses %s", name, losses)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
