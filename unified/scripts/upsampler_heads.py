"""Validation curves of the head experiment side by side (docs/UPSAMPLER_PLAN.md,
"Inside a trained head"): I1, I1 + trilinear skip, I1 + guided skip, I2, at every
validation epoch any of them has reached.

Usage:  python -m scripts.upsampler_heads
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = {"I1": "dino3d_i1_frz_pt_f100", "I1+tri": "dino3d_i1t_frz_pt_f100",
        "I1+guided": "dino3d_i1g_frz_pt_f100", "I2": "dino3d_i2_frz_pt_f100"}


def curve(run):
    p = REPO / "runs/interface" / run / "val_metrics.jsonl"
    if not p.exists():
        return {}
    return {d["epoch"]: d["mean_dice"] for d in map(json.loads, p.read_text().splitlines())}


def main():
    cs = {k: curve(v) for k, v in RUNS.items()}
    new = set(cs["I1+tri"]) | set(cs["I1+guided"])
    print(f"{'epoch':>5} " + " ".join(f"{k:>10}" for k in cs) + "   guided-tri  guided-I1   I2-I1")
    for e in sorted(new):
        row = [cs[k].get(e) for k in cs]
        f = lambda v: f"{v:10.4f}" if v is not None else f"{'-':>10}"  # noqa: E731
        i1, tri, g, i2 = row
        d = lambda a, b: f"{a - b:+10.4f}" if a is not None and b is not None else f"{'-':>10}"  # noqa: E731
        print(f"{e:5d} " + " ".join(f(v) for v in row) + f" {d(g, tri)} {d(g, i1)} {d(i2, i1)}")


if __name__ == "__main__":
    main()
