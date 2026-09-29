"""Populate the model-independent disk cache ahead of training, on CPU.

``CachedDataset`` fills itself lazily, so the first run at a new
``preprocessing_fingerprint`` pays the whole NIfTI-load + resample + foreground-crop
cost inside its first epoch. When the GPUs are busy but the CPUs are not — e.g. after
flipping ``data.use_source_affine`` on, which forks a fresh fingerprint — building it
up front costs nothing and takes that hit off the first run.

Writes exactly what training writes (same dataset, same det transforms, same cache
directory), so it is a warm-up, not a second code path: anything already cached is
skipped, and a later training run cannot tell the difference.

Usage:
    python -m scripts.prebuild_cache --config configs/interface/ctfm_i1_frz_pt_f100.yaml
    python -m scripts.prebuild_cache --config ... --workers 16 --splits train,val
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from unified.utils import load_config, setup_logging  # noqa: E402
from unified.data import (  # noqa: E402
    CachedDataset, TotalSegmentatorDataset, build_cache_det_transforms,
    load_classes, preprocessing_fingerprint,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--splits-dir", default=str(REPO / "unified" / "data" / "splits"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    setup_logging(None)
    cfg = load_config(args.config)
    cache_cfg = cfg["data"].get("cache", {})
    if not cache_cfg.get("enabled", False):
        raise SystemExit("data.cache.enabled is false for this config")

    fp = preprocessing_fingerprint(cfg)
    cache_root = Path(cache_cfg["dir"]) / fp
    src_affine = bool(cfg["data"].get("use_source_affine", False))
    print(f"config           {args.config}")
    print(f"use_source_affine {src_affine}  -> "
          f"{'1.5mm (native)' if src_affine else '2.25mm (Spacingd re-resamples)'}")
    print(f"cache            {cache_root}", flush=True)

    classes = load_classes()
    det = build_cache_det_transforms(cfg)
    ids: list[str] = []
    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        ids += [l.strip() for l in (Path(args.splits_dir) / f"{sp}.txt").read_text().splitlines()
                if l.strip()]
    if args.limit:
        ids = ids[: args.limit]

    raw = TotalSegmentatorDataset(cfg["data"]["dataset_root"], ids, classes,
                                  use_source_affine=src_affine)
    ds = CachedDataset(raw, det_transforms=det, cache_dir=cache_root, post_transforms=None)

    todo = [i for i in range(len(ds)) if not ds._cache_path(i).exists()]
    print(f"{len(ds)} subjects, {len(todo)} to build, {len(ds) - len(todo)} already cached",
          flush=True)
    if not todo:
        return

    t0 = time.time()
    done = 0

    def build(i):
        ds[i]          # side effect: writes cache_dir/<sid>.pt
        return i

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in ex.map(build, todo):
            done += 1
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"  {done}/{len(todo)}  {el/done:.1f}s/subject  "
                      f"eta {(len(todo)-done)*el/done/60:.0f} min", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {cache_root}")


if __name__ == "__main__":
    main()
