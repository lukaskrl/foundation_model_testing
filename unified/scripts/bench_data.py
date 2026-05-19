"""Profile the data pipeline to find the dataloading bottleneck.

Stages timed (over N subjects):
  1) raw __getitem__       — nib.load CT + 117 mask merge (no transforms)
  2) full train pipeline   — raw + Spacingd + CropForegroundd + RandCrop + augs
  3) DataLoader throughput — same as (2) but through workers; reports
                             samples/s for num_workers ∈ {0, 4, 8}

Run:
    CUDA_VISIBLE_DEVICES=1 python -m scripts.bench_data --config configs/models/voco_b.yaml --subjects 4
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from unified.utils import load_config, setup_logging, get_logger  # noqa: E402
from unified.data import (
    TotalSegmentatorDataset, load_classes,
    build_train_transforms,
)  # noqa: E402


def _read_split(path: Path):
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def _percentiles(xs):
    xs = sorted(xs)
    if not xs:
        return (0.0, 0.0, 0.0)
    n = len(xs)
    return (xs[0], xs[n // 2], xs[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subjects", type=int, default=4,
                    help="how many distinct subjects to profile")
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8],
                    help="num_workers values to compare for DataLoader stage")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(None)
    log = get_logger("bench")

    classes = load_classes()
    ids = _read_split(REPO / "unified" / "data" / "splits" / "train.txt")[: args.subjects]
    log.info("profiling on %d subjects: %s", len(ids), ids)

    raw = TotalSegmentatorDataset(cfg["data"]["dataset_root"], ids, classes)

    # --- 1) raw __getitem__ ----------------------------------------------------
    log.info("=== stage 1: raw __getitem__ (CT + 117 mask merge) ===")
    raw_times = []
    for i in range(len(ids)):
        t = time.time()
        item = raw[i]
        dt = time.time() - t
        raw_times.append(dt)
        log.info("  subj %s: %.2fs  image=%s label=%s",
                 ids[i], dt, tuple(item["image"].shape), tuple(item["label"].shape))
    lo, mid, hi = _percentiles(raw_times)
    log.info("  raw __getitem__: min=%.2fs median=%.2fs max=%.2fs  total=%.2fs",
             lo, mid, hi, sum(raw_times))

    # --- 2) full train pipeline (no DataLoader) --------------------------------
    log.info("=== stage 2: raw + train transforms (single process) ===")
    tf = build_train_transforms(cfg)
    tf_times = []
    for i in range(len(ids)):
        item = raw[i]  # we time transforms separately so re-fetch the raw
        t = time.time()
        out = tf(item)
        dt = time.time() - t
        tf_times.append(dt)
        # RandCropByPosNegLabel returns a list of patches
        if isinstance(out, list):
            shapes = [tuple(o["image"].shape) for o in out]
        else:
            shapes = [tuple(out["image"].shape)]
        log.info("  subj %s: transforms %.2fs  patches=%s", ids[i], dt, shapes)
    lo, mid, hi = _percentiles(tf_times)
    log.info("  transforms only: min=%.2fs median=%.2fs max=%.2fs  total=%.2fs",
             lo, mid, hi, sum(tf_times))
    log.info("  raw+transforms total: %.2fs (%.2fs/subj avg)",
             sum(raw_times) + sum(tf_times),
             (sum(raw_times) + sum(tf_times)) / max(1, len(ids)))

    # --- 3) DataLoader throughput ---------------------------------------------
    log.info("=== stage 3: DataLoader throughput (full pipeline through workers) ===")

    class Composed(torch.utils.data.Dataset):
        def __init__(self, base, t):
            self.base, self.t = base, t

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            return self.t(self.base[i])

    from monai.data import DataLoader
    composed = Composed(raw, tf)

    for nw in args.workers:
        loader = DataLoader(
            composed,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
        )
        # iterate two epochs so we measure both cold + warm worker startup
        for epoch in range(2):
            t0 = time.time()
            n_samples = 0
            t_batches = []
            t_prev = time.time()
            for batch in loader:
                dt = time.time() - t_prev
                t_batches.append(dt)
                n_samples += batch["image"].shape[0]
                t_prev = time.time()
            total = time.time() - t0
            lo, mid, hi = _percentiles(t_batches)
            log.info(
                "  workers=%d epoch=%d: total=%.2fs samples=%d  batch_dt min=%.2fs med=%.2fs max=%.2fs  thr=%.2f samp/s",
                nw, epoch, total, n_samples, lo, mid, hi, n_samples / max(1e-9, total),
            )

    log.info("done")


if __name__ == "__main__":
    main()
