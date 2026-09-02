"""Reclaim disk from finished runs in a matrix directory.

The trainer finalizes a run automatically when it finishes cleanly (see
``checkpoint:`` in base.yaml). This script is the sweep for everything that
route misses: runs killed by the OOM killer, runs from before finalization
existed, and runs whose ``.done`` was written by an older launcher.

Only runs marked complete (a ``.done`` marker) are touched, so a run that is
still training or still retryable keeps the epoch_*.pt it needs to resume.

    python -m scripts.cleanup_runs --dry-run          # report, change nothing
    python -m scripts.cleanup_runs                    # reclaim
    python -m scripts.cleanup_runs --root runs/lowshot --all   # ignore .done
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from unified.utils import finalize_run  # noqa: E402
from unified.utils.config import load_config  # noqa: E402


def _cfg_for(run_dir: Path):
    """The config a run was launched from, or None if it can't be identified.

    Without it finalize_run can still drop optimizer state, but not the frozen
    backbone — that needs the model block to rebuild a reference and to confirm
    the run was frozen+pretrained in the first place.
    """
    cand = REPO / "configs" / "lowshot" / f"{run_dir.name}.yaml"
    if not cand.exists():
        return None
    try:
        return load_config(str(cand))
    except Exception:
        return None


def _gb(n: int) -> float:
    return n / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/lowshot",
                    help="matrix directory holding one subdir per run")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be reclaimed, change nothing")
    ap.add_argument("--all", action="store_true",
                    help="also finalize runs with no .done marker (DESTROYS "
                         "their resume state — only for abandoned runs)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = REPO / root
    if not root.is_dir():
        sys.exit(f"no such directory: {root}")

    total_freed = 0
    n_runs = 0
    skipped = 0
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        done = (run_dir / ".done").exists()
        if not done and not args.all:
            if list(run_dir.glob("epoch_*.pt")):
                skipped += 1
            continue

        if args.dry_run:
            freed = sum(c.stat().st_size for c in run_dir.glob("epoch_*.pt"))
            best = run_dir / "best.pt"
            if best.exists():
                # optimizer+scheduler state is ~2/3 of a fine-tune checkpoint
                freed += int(best.stat().st_size * 2 / 3)
            if freed:
                print(f"  would reclaim {_gb(freed):7.2f} GB  {run_dir.name}")
        else:
            cfg = _cfg_for(run_dir)
            before = (run_dir / "best.pt").stat().st_size if (run_dir / "best.pt").exists() else 0
            summary = finalize_run(run_dir, cfg=cfg)
            freed = summary["freed_bytes"]
            after = (run_dir / "best.pt").stat().st_size if (run_dir / "best.pt").exists() else 0
            if freed:
                note = "" if cfg is not None else "  [no config — backbone kept]"
                print(f"  reclaimed {_gb(freed):7.2f} GB  {run_dir.name} "
                      f"({summary['removed']} epoch ckpt(s), best.pt "
                      f"{before/1048576:.0f}->{after/1048576:.0f} MB){note}")
        total_freed += freed
        n_runs += 1

    verb = "would reclaim" if args.dry_run else "reclaimed"
    print(f"\n{verb} {_gb(total_freed):.2f} GB across {n_runs} finished run(s)")
    if skipped:
        print(f"skipped {skipped} unfinished run(s) holding resume state "
              f"(use --all to force)")


if __name__ == "__main__":
    main()
