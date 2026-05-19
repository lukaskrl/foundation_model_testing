"""End-to-end perf test for the training step.

Runs N training steps with the real backbone + head + loss + optimizer and
separates the time the GPU spends *waiting on data* from the time it spends
*computing*. This is the metric to watch when "GPUs are filled but not
utilized": if data-wait dominates, the loader is the bottleneck; if compute
dominates, the model is.

Stages timed per step:
  - data_wait_s : wall time from "ask loader for next batch" until it arrives
                  (worker IO + transforms + collate + H2D copy queue setup)
  - h2d_s       : host->device copy of the batch
  - compute_s   : forward + loss + backward + optimizer.step + cuda.synchronize

Reports per-stage min/median/max and samples/s; tells you which stage owns
the step time.

    python -m scripts.bench_step --config configs/models/voco_b.yaml \\
        --subjects 16 --steps 20 --no-weights

Use --no-cache to force the slow 117-file label merge (baseline) and compare
against the cached path.
"""
from __future__ import annotations
import argparse
import statistics
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


def _stats(xs):
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), statistics.median(xs), max(xs), sum(xs) / len(xs))


def _fmt_row(name, xs):
    lo, mid, hi, mean = _stats(xs)
    return f"  {name:>14}: min={lo*1000:7.1f}ms  med={mid*1000:7.1f}ms  max={hi*1000:7.1f}ms  mean={mean*1000:7.1f}ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2,
                    help="warmup steps excluded from stats")
    ap.add_argument("--subjects", type=int, default=16,
                    help="how many subjects to pull from (shuffled)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override cfg.train.batch_size")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="override cfg.data.num_workers")
    ap.add_argument("--nspv", type=int, default=None,
                    help="override cfg.data.num_samples_per_volume (patches per loaded volume)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore subject_dir/label.nii.gz (forces 117-file merge per sample)")
    ap.add_argument("--no-weights", action="store_true",
                    help="skip loading pretrained backbone weights")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.nspv is not None:
        cfg["data"]["num_samples_per_volume"] = args.nspv
    setup_logging(None)
    log = get_logger("bench_step")

    name = cfg["model"]["name"]
    nw = cfg["data"]["num_workers"]
    bs = cfg["train"]["batch_size"]
    nspv = cfg["data"]["num_samples_per_volume"]
    log.info("=" * 78)
    log.info("bench_step model=%s batch=%d nspv=%d workers=%d cache=%s",
             name, bs, nspv, nw, "off" if args.no_cache else "on")
    log.info("=" * 78)

    classes = load_classes()
    ids = _read_split(REPO / "unified" / "data" / "splits" / "train.txt")[: args.subjects]
    raw = TotalSegmentatorDataset(
        cfg["data"]["dataset_root"], ids, classes,
        use_merged_label=(not args.no_cache),
    )
    log.info("merged labels available for %d/%d subjects",
             raw.num_merged_available, len(raw))
    tf = build_train_transforms(cfg)

    class Composed(torch.utils.data.Dataset):
        def __init__(self, base, t):
            self.base, self.t = base, t

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            return self.t(self.base[i])

    from monai.data import DataLoader
    loader = DataLoader(
        Composed(raw, tf),
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=(4 if nw > 0 else None),
        drop_last=True,
    )

    # Model.
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    optimizer = build_optimizer(cfg, model.parameters())
    loss_fn = build_loss(cfg)
    scaler = torch.cuda.amp.GradScaler() if cfg["train"].get("amp", True) and device.type == "cuda" else None
    log.info("device=%s amp=%s trainable=%d",
             device, scaler is not None, model.num_trainable_params())

    # Iterate.
    data_wait, h2d, compute, total = [], [], [], []
    step = 0
    loader_iter = iter(loader)
    t_step = time.time()
    while step < args.steps + args.warmup:
        try:
            t_w = time.time()
            batch = next(loader_iter)
            dt_wait = time.time() - t_w
        except StopIteration:
            loader_iter = iter(loader)
            continue

        t_h = time.time()
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_h2d = time.time() - t_h

        t_c = time.time()
        optimizer.zero_grad(set_to_none=True)
        try:
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(image)
                    loss = loss_fn(logits, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(image)
                loss = loss_fn(logits, label)
                loss.backward()
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            log.error("OOM at step %d image=%s", step, tuple(image.shape))
            raise
        dt_c = time.time() - t_c
        dt_total = time.time() - t_step
        t_step = time.time()

        tag = "warmup" if step < args.warmup else "step  "
        log.info(
            "%s %d  wait=%.0fms h2d=%.0fms compute=%.0fms total=%.0fms  loss=%.4f img=%s",
            tag, step, dt_wait * 1000, dt_h2d * 1000, dt_c * 1000, dt_total * 1000,
            loss.item(), tuple(image.shape),
        )
        if step >= args.warmup:
            data_wait.append(dt_wait); h2d.append(dt_h2d)
            compute.append(dt_c); total.append(dt_total)
        step += 1

    log.info("=" * 78)
    log.info("results over %d measured steps (warmup=%d skipped):", len(total), args.warmup)
    log.info(_fmt_row("data_wait", data_wait))
    log.info(_fmt_row("h2d", h2d))
    log.info(_fmt_row("compute", compute))
    log.info(_fmt_row("step_total", total))
    mean_total = sum(total) / max(1, len(total))
    mean_compute = sum(compute) / max(1, len(compute))
    mean_wait = sum(data_wait) / max(1, len(data_wait))
    samples_per_s = bs / mean_total if mean_total > 0 else 0.0
    log.info("  samples/s: %.2f   (batch=%d, num_samples_per_volume=%d)",
             samples_per_s, bs, nspv)
    log.info("  GPU busy fraction: compute / step_total = %.1f%%",
             100 * mean_compute / max(1e-9, mean_total))
    log.info("  Bottleneck: %s",
             "DATA" if mean_wait > mean_compute else "COMPUTE")
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
