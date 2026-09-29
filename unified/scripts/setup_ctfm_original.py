"""One-shot setup for CT-FM's own lighter pipeline (head-to-head arm A).

Two steps, both idempotent:

1. **Patch lighter.** ``lighter==0.0.3a18`` (pinned in ``CT-FM/requirements.txt``)
   binds ``on_train_start -> mode="train"`` and ``on_validation_end -> mode=None``.
   ``on_train_start`` fires once per ``fit()``, so after the first validation epoch
   ``self.mode`` stays ``None`` and the next training step dies in
   ``_prepare_batch`` with ``getattr(): attribute name must be string`` -- one crash
   every ``check_val_every_n_epoch`` epochs. The patch re-asserts the mode at the
   start of every training epoch. Bookkeeping only; the training math is untouched.
   It lives in the venv, so **any reinstall of that venv reverts it** -- re-run this
   script afterwards (``run_ctfm_original.sh`` refuses to start without it).

2. **Dataset view.** CT-FM's loader wants a comma-separated ``meta.csv``; the official
   v2.0.1 one is ``;``-separated with a UTF-8 BOM. This builds
   ``$DATA_ROOT/TotalSegmentatorDataset_ctfm``: one absolute symlink per subject into
   the real dataset, plus the re-delimited ``meta.csv`` (pandas round-trip, same
   rows and split column). CT-FM reads each subject's merged ``label.nii.gz``, so
   ``python -m scripts.prepare_data`` must have run first.

Must run under the CT-FM venv's interpreter (it needs that venv's lighter + pandas):

    ../CT-FM/.venv/bin/python scripts/setup_ctfm_original.py
    ../CT-FM/.venv/bin/python scripts/setup_ctfm_original.py --skip-patch
    DATA_ROOT=/mnt/big ../CT-FM/.venv/bin/python scripts/setup_ctfm_original.py

Create the venv first if it does not exist:

    python3.10 -m venv ../CT-FM/.venv
    ../CT-FM/.venv/bin/pip install -r ../CT-FM/requirements.txt
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "data"))

PINNED_LIGHTER = "0.0.3a18"
ANCHOR = "            self.on_train_start = lambda: self._on_mode_start(Mode.TRAIN)\n"
MARKER = "self.on_train_epoch_start = lambda: self._on_mode_start(Mode.TRAIN)"
PATCH = (
    "            # PATCH (local, lighter==0.0.3a18): on_train_start fires once per fit(), but\n"
    "            # on_validation_end resets self.mode to None. The first training step of the\n"
    "            # epoch after any validation epoch then dies in _prepare_batch with\n"
    "            # \"getattr(): attribute name must be string\". Re-assert the mode at the start\n"
    "            # of every training epoch. Bookkeeping only; does not touch the training math.\n"
    f"            {MARKER}\n"
)


def patch_lighter(system_py: Path) -> str:
    """Insert the mode re-assert after the ``on_train_start`` binding. Keeps a ``.orig``."""
    src = system_py.read_text()
    if MARKER in src:
        return "already patched"
    if src.count(ANCHOR) != 1:
        raise SystemExit(f"{system_py}: expected exactly one anchor line\n  {ANCHOR.strip()}\n"
                         f"found {src.count(ANCHOR)} -- not lighter=={PINNED_LIGHTER}?")
    backup = system_py.with_name(system_py.name + ".orig")
    if not backup.exists():
        shutil.copy2(system_py, backup)
    system_py.write_text(src.replace(ANCHOR, ANCHOR + PATCH))
    return "patched"


def build_dataset(src: Path, dst: Path) -> str:
    """Symlink every subject of ``src`` into ``dst`` and write a comma-separated meta.csv."""
    import pandas as pd

    subjects = sorted(p for p in src.iterdir() if p.is_dir() and p.name.startswith("s"))
    if not subjects:
        raise SystemExit(f"no subject directories under {src} -- run download_totalsegmentator.sh")
    # CT-FM's get_ts_datalist reads <subject>/label.nii.gz: the merged label this repo writes.
    unmerged = [s.name for s in subjects if not (s / "label.nii.gz").exists()]
    if unmerged:
        raise SystemExit(f"{len(unmerged)} subjects lack label.nii.gz (e.g. {unmerged[0]}) "
                         "-- run python -m scripts.prepare_data first")
    dst.mkdir(parents=True, exist_ok=True)
    made = 0
    for subj in subjects:
        link = dst / subj.name
        if link.is_symlink():
            if Path(os.readlink(link)) != subj.resolve():
                raise SystemExit(f"{link} points to {os.readlink(link)}, expected {subj.resolve()}")
            continue
        link.symlink_to(subj.resolve())
        made += 1
    meta = pd.read_csv(src / "meta.csv", sep=";", encoding="utf-8-sig")
    meta.to_csv(dst / "meta.csv", index=False)
    return f"{len(subjects)} subjects ({made} new links), meta.csv {len(meta)} rows"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=DATA_ROOT / "TotalSegmentatorDataset")
    ap.add_argument("--dst", type=Path, default=DATA_ROOT / "TotalSegmentatorDataset_ctfm")
    ap.add_argument("--skip-patch", action="store_true")
    ap.add_argument("--skip-dataset", action="store_true")
    args = ap.parse_args()

    if not args.skip_patch:
        try:
            import lighter.system
            from importlib.metadata import version
        except ImportError:
            raise SystemExit("lighter not importable -- run this with ../CT-FM/.venv/bin/python")
        if version("lighter") != PINNED_LIGHTER:
            raise SystemExit(f"lighter {version('lighter')} installed, patch is for {PINNED_LIGHTER}")
        print("lighter:", patch_lighter(Path(lighter.system.__file__)))
    if not args.skip_dataset:
        print("dataset:", build_dataset(args.src, args.dst), "->", args.dst)


if __name__ == "__main__":
    main()
