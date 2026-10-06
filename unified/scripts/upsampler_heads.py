"""The head experiment side by side (docs/UPSAMPLER_PLAN.md, "Inside a trained head").

Default: validation curves of I1, I1 + trilinear skip, I1 + guided skip and I2 at every
validation epoch any of them has reached.

``--test``: the test split, from scripts/upsampler_test_eval.py's per-case JSONs. Macro
mean = mean over classes of the per-class mean over the cases where the class is present
(as scripts/evaluate.py). Differences are paired over the cases both runs have, with a
95% bootstrap interval over cases. Thin classes: 2V/S thickness under 8 mm
(results/upsampler/class_thickness.json). NSD at 1.5 mm and 3 mm (1 and 2 voxels).

Usage:  python -m scripts.upsampler_heads [--test] [--boot 2000]
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = {"I1": "dino3d_i1_frz_pt_f100", "I1+tri": "dino3d_i1t_frz_pt_f100",
        "I1+guided": "dino3d_i1g_frz_pt_f100", "I2": "dino3d_i2_frz_pt_f100"}
TEST_DIR = REPO / "results/upsampler/test_heads"
TEST_RUNS = {
    "dino3d I1": "dino3d_i1_frz_pt_f100", "dino3d I1+tri": "dino3d_i1t_frz_pt_f100",
    "dino3d I1+guided": "dino3d_i1g_frz_pt_f100", "dino3d I2": "dino3d_i2_frz_pt_f100",
    "dino3d I2+guided": "dino3d_i2g_frz_pt_f100",
    "SAM-Med3D I1": "samMed3d_i1_frz_pt_f100", "SAM-Med3D I1+tri": "samMed3d_i1t_frz_pt_f100",
    "SAM-Med3D I1+guided": "samMed3d_i1g_frz_pt_f100",
    "CT-FM I1": "ctfm_i1_frz_pt_f100",       # CNN reference: its native pyramid, strides 16..1
}
PAIRS = [  # a − b
    ("dino3d I1+guided", "dino3d I1+tri"), ("dino3d I1+tri", "dino3d I1"),
    ("dino3d I1+guided", "dino3d I1"), ("dino3d I2", "dino3d I1"),
    ("dino3d I2+guided", "dino3d I2"), ("dino3d I1+guided", "dino3d I2"),
    ("SAM-Med3D I1+guided", "SAM-Med3D I1+tri"), ("SAM-Med3D I1+guided", "SAM-Med3D I1"),
    ("SAM-Med3D I1+tri", "SAM-Med3D I1"),
    ("CT-FM I1", "dino3d I1"), ("CT-FM I1", "dino3d I1+guided"), ("CT-FM I1", "dino3d I2"),
    ("CT-FM I1", "dino3d I2+guided"), ("CT-FM I1", "SAM-Med3D I1"), ("CT-FM I1", "SAM-Med3D I1+guided"),
]
# ViT-to-CNN gap closed: (X − ViT I1) / (CNN I1 − ViT I1), with the same decoder for all
GAPS = [("dino3d I1", ["dino3d I1+tri", "dino3d I1+guided", "dino3d I2", "dino3d I2+guided"], "CT-FM I1"),
        ("SAM-Med3D I1", ["SAM-Med3D I1+tri", "SAM-Med3D I1+guided"], "CT-FM I1")]
METRICS = [("dice", "Dice"), ("nsd1", "NSD 1.5mm"), ("nsd2", "NSD 3mm")]
THIN_MM = 8.0


def curve(run):
    p = REPO / "runs/interface" / run / "val_metrics.jsonl"
    if not p.exists():
        return {}
    return {d["epoch"]: d["mean_dice"] for d in map(json.loads, p.read_text().splitlines())}


def val_table():
    cs = {k: curve(v) for k, v in RUNS.items()}
    new = set(cs["I1+tri"]) | set(cs["I1+guided"])
    print(f"{'epoch':>5} " + " ".join(f"{k:>10}" for k in cs) + "   guided-tri  guided-I1   I2-I1")
    for e in sorted(new):
        row = [cs[k].get(e) for k in cs]
        f = lambda v: f"{v:10.4f}" if v is not None else f"{'-':>10}"  # noqa: E731
        i1, tri, g, i2 = row
        d = lambda a, b: f"{a - b:+10.4f}" if a is not None and b is not None else f"{'-':>10}"  # noqa: E731
        print(f"{e:5d} " + " ".join(f(v) for v in row) + f" {d(g, tri)} {d(g, i1)} {d(i2, i1)}")


def load_cases(run):
    p = TEST_DIR / f"{run}_test.json"
    return json.loads(p.read_text())["cases"] if p.exists() else {}


def matrix(cases, sids, metric, classes):
    """(cases, classes) array, NaN where the class is absent from the case."""
    V = np.full((len(sids), len(classes)), np.nan)
    for i, s in enumerate(sids):
        for j, c in enumerate(classes):
            v = cases[s][metric].get(c)
            if v is not None:
                V[i, j] = v
    return V


def class_means(V, idx=None):
    X = V if idx is None else V[idx]                      # (..., cases, classes)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return np.nanmean(X, axis=-2)


def test_table(n_boot, seed=0):
    thick = json.loads((REPO / "results/upsampler/class_thickness.json").read_text())
    data = {k: load_cases(r) for k, r in TEST_RUNS.items()}
    print("Test split, macro means over each run's own cases:")
    for k, cases in data.items():
        if not cases:
            continue
        sids = sorted(cases)
        classes = sorted({c for s in sids for c in cases[s]["dice"]})
        vals = [np.nanmean(class_means(matrix(cases, sids, m, classes))) for m, _ in METRICS]
        print(f"  {k:<20} n={len(sids):3d}  " + "  ".join(f"{n} {v:.4f}" for (_, n), v in zip(METRICS, vals)))

    rng = np.random.default_rng(seed)
    print(f"\nPaired differences a − b over shared cases, 95% bootstrap CI ({n_boot} resamples):")
    for a, b in PAIRS:
        A, B = data[a], data[b]
        sids = sorted(set(A) & set(B))
        if not sids:
            continue
        classes = sorted({c for s in sids for c in A[s]["dice"]})
        thin = np.array([c in thick and thick[c]["thick_mm"] < THIN_MM for c in classes])
        known = np.array([c in thick for c in classes])
        groups = {"all": np.ones(len(classes), bool), "thin": thin, "thick": known & ~thin}
        idx = rng.integers(0, len(sids), (n_boot, len(sids)))
        print(f"  {a} − {b}  (n={len(sids)}; {thin.sum()} thin / {(known & ~thin).sum()} thick classes)")
        for m, name in METRICS:
            VA, VB = matrix(A, sids, m, classes), matrix(B, sids, m, classes)
            d_cls = class_means(VA) - class_means(VB)                  # (classes,)
            d_boot = class_means(VA, idx) - class_means(VB, idx)       # (boot, classes)
            cells = []
            for g, mask in groups.items():
                pt = np.nanmean(d_cls[mask])
                lo, hi = np.nanpercentile(np.nanmean(d_boot[:, mask], 1), [2.5, 97.5])
                up = int((d_cls[mask] > 0).sum())
                cells.append(f"{g} {pt:+.4f} [{lo:+.4f},{hi:+.4f}] {up}/{mask.sum()} up")
            print(f"    {name:<10} " + "   ".join(cells))
    gap_table(data, thick, n_boot)


def gap_table(data, thick, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    print(f"\nViT-to-CNN gap closed, (X − ViT I1) / (CNN − ViT I1), Dice, 95% bootstrap CI:")
    for base, xs, cnn in GAPS:
        xs = [x for x in xs if data[x]]
        if not (data[base] and data[cnn] and xs):
            continue
        sids = sorted(set(data[base]) & set(data[cnn]).intersection(*[data[x] for x in xs]))
        classes = sorted({c for s in sids for c in data[base][s]["dice"]})
        thin = np.array([c in thick and thick[c]["thick_mm"] < THIN_MM for c in classes])
        idx = rng.integers(0, len(sids), (n_boot, len(sids)))
        M = {k: matrix(data[k], sids, "dice", classes) for k in [base, cnn, *xs]}
        print(f"  {base} -> {cnn}  (n={len(sids)})")
        for g, mask in {"all": np.ones(len(classes), bool), "thin": thin}.items():
            pt = {k: np.nanmean(class_means(V)[mask]) for k, V in M.items()}
            bt = {k: np.nanmean(class_means(V, idx)[:, mask], 1) for k, V in M.items()}
            gap = pt[cnn] - pt[base]
            cells = []
            for x in xs:
                r = (bt[x] - bt[base]) / (bt[cnn] - bt[base])
                lo, hi = np.nanpercentile(r, [2.5, 97.5])
                cells.append(f"{x.split(' ', 1)[1]} {100 * (pt[x] - pt[base]) / gap:.0f}% [{100 * lo:.0f},{100 * hi:.0f}]")
            print(f"    {g:<5} gap {gap:+.4f}: " + "   ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    test_table(a.boot) if a.test else val_table()


if __name__ == "__main__":
    main()
