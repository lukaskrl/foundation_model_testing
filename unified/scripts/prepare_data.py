"""One-shot dataset prep: write train/val/test splits and pre-merge segmentations.

Two steps, both idempotent:

1. **Splits** — read ``meta.csv`` and write ``unified/data/splits/{train,val,test}.txt``
   (one subject ID per line). Same split, reproducible across runs.

2. **Merged labels** — for each subject, merge the 117 per-organ binary masks in
   ``segmentations/`` into a single uint8 ``label.nii.gz`` written *next to*
   ``ct.nii.gz`` in the subject directory. At train time the dataset reads this
   one file instead of opening + gunzipping 118 files per sample, which is the
   single biggest cause of GPU starvation during training.

Class order matches ``classes.txt`` exactly (background = 0, organ_i = i+1),
so merged and on-the-fly labels are bit-equivalent.

Usage:
    python -m scripts.prepare_data                              # splits + merge all
    python -m scripts.prepare_data --skip-merge                 # just splits
    python -m scripts.prepare_data --skip-splits --jobs 8       # just merge, parallel
    python -m scripts.prepare_data --split train --limit 32     # quick subset
    python -m scripts.prepare_data --overwrite                  # rebuild merged labels
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from unified.data import build_subject_lists, load_classes  # noqa: E402
from unified.data.totalsegmentator import LABEL_FILENAME  # noqa: E402


def _read_split(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def _write_splits(meta_csv: Path, out_dir: Path) -> dict[str, list[str]]:
    splits = build_subject_lists(meta_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        path = out_dir / f"{name}.txt"
        path.write_text("\n".join(sorted(ids)) + "\n")
        print(f"  split {name}: {len(ids)} subjects -> {path}")
    return splits


def _merge_one(args):
    sid, dataset_root, classes, overwrite = args
    import nibabel as nib

    sdir = Path(dataset_root) / sid
    ct_path = sdir / "ct.nii.gz"
    seg_dir = sdir / "segmentations"
    out = sdir / LABEL_FILENAME

    if not ct_path.exists() or not seg_dir.is_dir():
        return sid, 0.0, "missing"
    if out.exists() and not overwrite:
        return sid, 0.0, "skip"

    t0 = time.time()
    ct_nii = nib.load(str(ct_path))
    shape = ct_nii.shape
    affine = ct_nii.affine

    label = np.zeros(shape, dtype=np.uint8)
    for i, name in enumerate(classes, start=1):
        f = seg_dir / f"{name}.nii.gz"
        if not f.exists():
            continue
        mask = np.asanyarray(nib.load(str(f)).dataobj)
        label[mask > 0] = i

    out_nii = nib.Nifti1Image(label, affine)
    out_nii.set_data_dtype(np.uint8)
    # Atomic write: tmp file then rename, so a killed worker can't leave a partial.
    # Prefix the temp name so the ".nii.gz" extension stays intact for nibabel.
    tmp = out.parent / f".tmp.{os.getpid()}.{out.name}"
    nib.save(out_nii, str(tmp))
    os.replace(tmp, out)
    return sid, time.time() - t0, "ok"


def _merge_labels(
    dataset_root: Path,
    ids: list[str],
    classes: list[str],
    jobs: int,
    overwrite: bool,
) -> None:
    print(
        f"  merging labels for {len(ids)} subjects -> "
        f"{dataset_root}/<sid>/{LABEL_FILENAME}  (jobs={jobs}, overwrite={overwrite})"
    )
    tasks = [(sid, str(dataset_root), classes, overwrite) for sid in ids]

    t0 = time.time()
    n_ok = n_skip = n_miss = 0
    dts: list[float] = []

    def _record(result):
        nonlocal n_ok, n_skip, n_miss
        sid, dt, status = result
        if status == "ok":
            n_ok += 1; dts.append(dt)
        elif status == "skip":
            n_skip += 1
        else:
            n_miss += 1

    if jobs <= 1:
        for i, task in enumerate(tasks, 1):
            _record(_merge_one(task))
            if i % 25 == 0 or i == len(tasks):
                print(f"    {i}/{len(ids)} ok={n_ok} skip={n_skip} miss={n_miss}")
    else:
        with mp.Pool(jobs) as pool:
            for i, res in enumerate(pool.imap_unordered(_merge_one, tasks), 1):
                _record(res)
                if i % 25 == 0 or i == len(tasks):
                    print(f"    {i}/{len(ids)} ok={n_ok} skip={n_skip} miss={n_miss}")

    total = time.time() - t0
    avg = sum(dts) / max(1, len(dts))
    print(
        f"  merge done in {total:.1f}s — ok={n_ok} skip={n_skip} miss={n_miss}  "
        f"avg {avg:.2f}s/subj"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/store/Datasets/TotalSegmentatorDataset")
    ap.add_argument(
        "--meta-csv",
        default="/store/Datasets/TotalSegmentatorDataset/meta.csv",
    )
    ap.add_argument(
        "--splits-dir",
        default=str(REPO / "unified" / "data" / "splits"),
    )
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="all",
                    help="which split(s) to merge labels for")
    ap.add_argument("--limit", type=int, default=None,
                    help="merge only the first N subjects (handy for benchmarking)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite label.nii.gz even if it already exists")
    ap.add_argument("--skip-splits", action="store_true",
                    help="don't (re)write split files; use existing splits")
    ap.add_argument("--skip-merge", action="store_true",
                    help="don't merge labels; just (re)write split files")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    splits_dir = Path(args.splits_dir)

    # --- step 1: splits -----------------------------------------------------
    if not args.skip_splits:
        print("[1/2] writing splits")
        _write_splits(Path(args.meta_csv), splits_dir)
    else:
        print("[1/2] skipping splits (--skip-splits)")

    # --- step 2: merged labels ---------------------------------------------
    if args.skip_merge:
        print("[2/2] skipping label merge (--skip-merge)")
        return

    print("[2/2] merging per-organ masks into label.nii.gz")
    classes = load_classes()
    names = ["train", "val", "test"] if args.split == "all" else [args.split]
    ids: list[str] = []
    for n in names:
        p = splits_dir / f"{n}.txt"
        if not p.exists():
            print(f"  skipping {n}: no split file at {p}")
            continue
        ids.extend(_read_split(p))
    if args.limit is not None:
        ids = ids[: args.limit]
    if not ids:
        print("  nothing to merge")
        return
    _merge_labels(dataset_root, ids, classes, args.jobs, args.overwrite)


if __name__ == "__main__":
    main()
