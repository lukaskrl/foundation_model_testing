"""One-shot: emit the train/val/test subject lists from meta.csv.

Writes ``unified/data/splits/{train,val,test}.txt`` with one subject ID per line.
This step doesn't modify the dataset directory; it just snapshots which subjects
go where so the same split is reproducible across runs.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from unified.data import build_subject_lists  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--meta-csv",
        default="/store/Datasets/TotalSegmentatorDataset/meta.csv",
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO / "unified" / "data" / "splits"),
    )
    args = ap.parse_args()

    splits = build_subject_lists(args.meta_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        path = out_dir / f"{name}.txt"
        path.write_text("\n".join(sorted(ids)) + "\n")
        print(f"{name}: {len(ids)} subjects → {path}")


if __name__ == "__main__":
    main()
