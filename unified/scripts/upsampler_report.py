"""Summarize upsampler probe results against the trilinear baseline.

For each method in a results JSON (scripts/upsampler_probe.py): mean / thin / thick
Dice and NSD at 3 mm, the paired deltas against trilinear, how many thin classes improve,
and the Spearman correlation of the per-class delta with structure thickness (a
negative value means the gain concentrates in thin structures, which is the claim).

Usage:  python -m scripts.upsampler_report results/upsampler/stage1_dino3d.json
        python -m scripts.upsampler_report --merge stage1_dino3d.json stage1_dino3d_label.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _rank(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x)); r[order] = np.arange(len(x))
    for v in np.unique(x):                       # average ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    a, b = _rank(np.asarray(a, float)), _rank(np.asarray(b, float))
    return float(np.corrcoef(a, b)[0, 1])


def main():
    thick = json.loads((REPO / "results/upsampler/class_thickness.json").read_text())
    args = sys.argv[1:]
    if "--merge" in args:           # one table from several files of the SAME encoder/banks
        groups = [[a for a in args if a != "--merge"]]
    else:
        groups = [[a] for a in args]
    for paths in groups:
        res = {}
        for path in paths:
            res.update({k: v for k, v in json.loads(Path(path).read_text()).items()
                        if k not in res})
        path = " + ".join(paths)
        base = res.get("trilinear")
        print(f"\n{path}")
        print(f"  {'method':44s} {'dice':>6s} {'thin':>6s} {'thick':>6s} {'nsd3':>6s} |"
              f" {'d_dice':>7s} {'d_thin':>7s} {'d_thick':>7s} {'d_nsd3':>7s} {'d_n3thin':>8s}"
              f"  thin+/n  rho(d,mm)")
        for k, v in res.items():
            if k.startswith("_"):
                continue
            nsd = v.get("nsd2_mean", float("nan"))
            row = (f"  {k:44s} {v['dice_mean']:6.3f} {v['dice_thin']:6.3f} {v['dice_thick']:6.3f}"
                   f" {nsd:6.3f} |")
            if base and k != "trilinear":
                pc, pb = v["per_class"], base["per_class"]
                common = [c for c in pc if c in pb and c in thick]
                d = np.array([pc[c] - pb[c] for c in common])
                mm = np.array([thick[c]["thick_mm"] for c in common])
                thin = mm < 8.0
                dn = v.get("nsd2_mean", np.nan) - base.get("nsd2_mean", np.nan)
                dnt = v.get("nsd2_thin", np.nan) - base.get("nsd2_thin", np.nan)
                row += (f" {v['dice_mean'] - base['dice_mean']:+7.4f}"
                        f" {v['dice_thin'] - base['dice_thin']:+7.4f}"
                        f" {v['dice_thick'] - base['dice_thick']:+7.4f} {dn:+7.4f} {dnt:+8.4f}"
                        f"  {int((d[thin] > 0).sum()):3d}/{int(thin.sum()):<3d} {spearman(d, mm):+.2f}")
            print(row)


if __name__ == "__main__":
    main()
