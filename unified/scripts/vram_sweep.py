"""Sweep training VRAM vs batch_size for every model config and recommend a
batch_size that fills the GPU without crashing it.

For each config, runs scripts/vram_probe.py in a FRESH process per batch size
(so an OOM can't poison later trials), increasing batch_size until the model
either OOMs or exceeds the reserved-memory budget. Recommends the largest tested
batch_size whose reserved memory stays under --budget-gb.

    python scripts/vram_sweep.py --budget-gb 44 --gpu 0

Writes runs/vram_sweep/results.jsonl (one row per trial) and prints a summary.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Sweep order; stops early per model on OOM / over-budget. num_samples=2, so the
# effective per-step batch is 2x these.
BATCH_SEQUENCE = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

# ct-clip's ViT expects a >=160 cube (image_size 480 / patch 20); the 96 patch
# the rest use does not tile. Probe it at a size it can actually run.
PATCH_OVERRIDE = {"ctclip": 160}


def run_probe(cfg_path, bs, gpu, patch=None, timeout=420):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONWARNINGS="ignore")
    cmd = [sys.executable, str(REPO / "scripts" / "vram_probe.py"),
           "--config", str(cfg_path), "--batch-size", str(bs)]
    if patch is not None:
        cmd += ["--patch", str(patch)]
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return {"status": "error", "msg": (p.stderr.splitlines()[-1] if p.stderr else "no RESULT")[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, default=44.0,
                    help="max reserved GB to consider 'fitting' (headroom below 48)")
    ap.add_argument("--hard-gb", type=float, default=46.0,
                    help="stop sweeping a model once reserved exceeds this")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="specific config basenames (no .yaml); default = all")
    args = ap.parse_args()

    cfg_dir = REPO / "configs" / "models"
    if args.configs:
        cfgs = [cfg_dir / f"{c}.yaml" for c in args.configs]
    else:
        cfgs = sorted(cfg_dir.glob("*.yaml"))

    out_dir = REPO / "runs" / "vram_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    rf = results_path.open("w")

    summary = []
    for cfg_path in cfgs:
        model = cfg_path.stem
        patch = PATCH_OVERRIDE.get(model)
        print(f"\n=== {model} (patch={patch or 96}) ===", flush=True)
        trials = []
        best = None
        for bs in BATCH_SEQUENCE:
            r = run_probe(cfg_path, bs, args.gpu, patch=patch)
            r["_bs_requested"] = bs
            rf.write(json.dumps(r) + "\n"); rf.flush()
            st = r.get("status")
            if st == "ok":
                resv = r["reserved_gb"]
                mark = "OK " if resv <= args.budget_gb else "OVER"
                print(f"  bs={bs:<3d} eff={r['eff']:<3d} peak={r['peak_gb']:>6.2f}  "
                      f"reserved={resv:>6.2f}  {r['amp']}  step={r['step_s']:.2f}s  [{mark}]",
                      flush=True)
                trials.append(r)
                if resv <= args.budget_gb:
                    best = r
                if resv > args.hard_gb:
                    print(f"  -> reserved {resv:.1f} > hard cap {args.hard_gb}; stop", flush=True)
                    break
            elif st == "oom":
                print(f"  bs={bs:<3d} OOM; stop", flush=True)
                break
            elif st == "timeout":
                print(f"  bs={bs:<3d} TIMEOUT; stop", flush=True)
                break
            else:
                print(f"  bs={bs:<3d} ERROR: {r.get('msg')}; stop", flush=True)
                break

        if best is not None:
            summary.append({
                "model": model, "rec_bs": best["bs"], "eff": best["eff"],
                "peak_gb": best["peak_gb"], "reserved_gb": best["reserved_gb"],
                "amp": best["amp"], "step_s": best["step_s"],
                "frozen": best.get("frozen"), "total_M": best.get("total_M"),
            })
        else:
            note = trials[-1].get("status") if trials else (
                "oom@bs1" if False else "no-fit")
            # If even bs=1 didn't fit / errored, capture why.
            first = run_probe(cfg_path, 1, args.gpu, patch=patch) if not trials else {}
            summary.append({"model": model, "rec_bs": None,
                            "note": trials[0].get("status") if trials else first.get("status", "fail"),
                            "msg": (trials[0].get("msg") if trials else first.get("msg"))})

    rf.close()

    print("\n" + "=" * 92)
    print(f"{'model':<20} {'rec_bs':>6} {'eff':>4} {'peak_GB':>8} {'resv_GB':>8} {'amp':>5} {'step_s':>7}  notes")
    print("-" * 92)
    for s in summary:
        if s.get("rec_bs"):
            print(f"{s['model']:<20} {s['rec_bs']:>6} {s['eff']:>4} {s['peak_gb']:>8.2f} "
                  f"{s['reserved_gb']:>8.2f} {s['amp']:>5} {s['step_s']:>7.2f}  "
                  f"{'frozen' if s.get('frozen') else 'full'} {s.get('total_M')}M")
        else:
            print(f"{s['model']:<20} {'-':>6} {'-':>4} {'-':>8} {'-':>8} {'-':>5} {'-':>7}  "
                  f"FAILED: {s.get('note')} {s.get('msg') or ''}")
    print("=" * 92)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nresults -> {results_path}\nsummary -> {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
