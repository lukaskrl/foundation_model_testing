"""Cache CT-RATE volumes for the classification benchmark.

Streams volumes from the gated HuggingFace repo one at a time, preprocesses to
the framework grid, caches the small float16 tensor, and deletes the raw NIfTI
(see unified.data.ctrate). Resumable and disk-floor guarded — safe to run in the
background and safe to re-run.

Default plan: 2000 distinct-patient train volumes (one reconstruction each, for
diversity) + the full validation split (all 3039 reconstructions, as published).

  python -m scripts.ctrate_cache --cache-root /home/lukas/data/CTRATE \
      --train-n 2000 --workers 6 --min-free-gb 100
"""
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

# Allow `python scripts/ctrate_cache.py` as well as `-m scripts.ctrate_cache`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download
from unified.data.ctrate import (
    _CSV_PATHS, load_label_table, load_metadata, stream_and_cache,
    PreprocessConfig,
)


def _one_per_patient(names, n, seed):
    """Pick ``n`` distinct patients, one (deterministic) reconstruction each."""
    by_pat = {}
    for nm in sorted(names):
        pat = "_".join(nm.split("_")[:2])       # e.g. train_1234
        by_pat.setdefault(pat, nm)              # first sorted recon per patient
    pats = sorted(by_pat)
    random.Random(seed).shuffle(pats)
    if n and n < len(pats):
        pats = pats[:n]
    return [by_pat[p] for p in pats]


def cache_split(split, n, cache_root, workers, min_free_gb, seed, one_per_patient):
    labels = load_label_table(
        hf_hub_download("ibrahimhamamci/CT-RATE", repo_type="dataset",
                        filename=_CSV_PATHS[split]["labels"]))
    meta = load_metadata(
        hf_hub_download("ibrahimhamamci/CT-RATE", repo_type="dataset",
                        filename=_CSV_PATHS[split]["meta"]))
    names = list(labels.keys())
    if one_per_patient:
        names = _one_per_patient(names, n, seed)
    elif n:
        names = sorted(names)[:n]
    else:
        names = sorted(names)
    cache_dir = Path(cache_root) / split
    print(f"[{split}] selecting {len(names)} volumes "
          f"(one_per_patient={one_per_patient}) -> {cache_dir}", flush=True)
    stats = stream_and_cache(
        names, split, cache_dir, meta, PreprocessConfig(),
        workers=workers, min_free_gb=min_free_gb, progress_every=50)
    print(f"[{split}] DONE cached={stats.cached} skip={stats.skipped_existing} "
          f"fail={len(stats.failed)} wrote={stats.bytes_written/1e9:.1f}GB", flush=True)
    if stats.failed:
        print(f"[{split}] first failures: {stats.failed[:5]}", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="/home/lukas/data/CTRATE")
    ap.add_argument("--train-n", type=int, default=2000,
                    help="distinct-patient train volumes (0 = all)")
    ap.add_argument("--val-n", type=int, default=0,
                    help="val volumes (0 = all 3039 reconstructions)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-free-gb", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    args = ap.parse_args()

    for split in args.splits:
        n = args.train_n if split == "train" else args.val_n
        cache_split(split, n, args.cache_root, args.workers, args.min_free_gb,
                    args.seed, one_per_patient=(split == "train"))
    print("ALL SPLITS DONE", flush=True)


if __name__ == "__main__":
    main()
