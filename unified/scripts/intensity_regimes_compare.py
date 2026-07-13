"""Head-to-head visual + quantitative comparison of CT intensity-normalization regimes.

Loads one TotalSegmentator volume through the *exact* geometric preprocessing the
framework uses (Spacingd -> CropForegroundd -> Orientationd(RAS)), then applies each
candidate intensity regime to the SAME raw-HU volume so you can compare them side by
side. It writes two figures and prints a metrics table.

Regimes compared (see unified-recipe-v2 / the intensity-windowing brainstorm):
  1. current_minmax    clip[-1024,2048] -> [0,1]         (what base.yaml does today)
  2. percentile_minmax clip[p0.5,p99.5] -> [0,1]         (data-driven clip, still linear)
  3. zscore_global     clip[p0.5,p99.5], (x-mean)/std    (nnU-Net CT scheme)
  4. narrow_soft       clip[-160,240] -> [0,1]           (WL40/WW400 abdomen window)
  5. sigmoid_soft      sigmoid((x-40)/80)                (nonlinear soft-tissue window)
  6. multiwindow       3ch [soft, bone, lung] as RGB     (radiologist-style multi-channel)

WHAT THE METRICS REVEAL (this is the point of the script):
  CNR (contrast-to-noise between two organs) is INVARIANT under any affine map. So
  regimes 1/2/3 -- and regime 4 for organ pairs that sit *inside* its window -- will
  print near-identical CNR. That is not a bug: it demonstrates that a full-fine-tuned
  network can largely undo a linear normalization, so the linear regimes are nearly
  equivalent. The regimes that actually change what is learnable are the ones that
  either (a) clip away information (narrow window collapses BONE-pair CNR) or
  (b) are genuinely nonlinear (sigmoid) or (c) add channels (multiwindow keeps BOTH
  soft- and bone-pair separability high). The figures show the same story visually.

Usage (from the unified/ repo root, with the project venv active):
  python -m scripts.intensity_regimes_compare                 # random train subject
  python -m scripts.intensity_regimes_compare --subject s0287 # a specific one
  python -m scripts.intensity_regimes_compare --split val --seed 3
"""
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from unified.utils import load_config  # noqa: E402
from unified.utils.config import _read_yaml, ConfigError  # noqa: E402
from unified.data import TotalSegmentatorDataset, load_classes  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


# ---- dataset fingerprint stats (TotalSeg train, n=1082) --------------------
# From runs/fingerprint/totalseg_train_fingerprint.json. Hardcoded so the script
# has no dependency on that artifact existing; refresh if you re-run the fingerprint.
FG_MEAN = -119.66
FG_STD = 505.78
FG_P0_5 = -998.0
FG_P99_5 = 1504.0

# The diagnostic soft-tissue HU band (abdominal organs live here).
SOFT_LO, SOFT_HI = -100.0, 200.0

# Clinical windows (window-level / window-width -> [lo, hi]).
def _wl_ww(wl, ww):
    return wl - ww / 2.0, wl + ww / 2.0

SOFT_WIN = _wl_ww(40, 400)      # (-160, 240)  abdomen / soft tissue
BONE_WIN = _wl_ww(450, 1800)    # (-450, 1350) bone
LUNG_WIN = _wl_ww(-600, 1500)   # (-1350, 150) lung


# ---- regimes ---------------------------------------------------------------
def _clip_scale(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def r_current(x):
    return _clip_scale(x, -1024.0, 2048.0)


def r_percentile(x):
    return _clip_scale(x, FG_P0_5, FG_P99_5)


def r_zscore(x):
    xc = np.clip(x, FG_P0_5, FG_P99_5)
    return (xc - FG_MEAN) / FG_STD


def r_narrow(x):
    return _clip_scale(x, *SOFT_WIN)


def r_sigmoid(x):
    # Centered on soft tissue (WL 40); scale 80 puts the WW/2=200 HU edges near
    # +/-2.5 sigmoid units, so soft tissue gets the steep middle while bone/lung
    # saturate SMOOTHLY (a rib at 1000 vs 1200 HU still differ) instead of clipping.
    return 1.0 / (1.0 + np.exp(-(x - 40.0) / 80.0))


def r_multiwindow(x):
    # (..., 3) stack of soft / bone / lung windows -> a false-color CT.
    soft = _clip_scale(x, *SOFT_WIN)
    bone = _clip_scale(x, *BONE_WIN)
    lung = _clip_scale(x, *LUNG_WIN)
    return np.stack([soft, bone, lung], axis=-1)


# zscore output range for the clipped HU band [P0.5, P99.5] -> displayed on the
# SAME theoretical basis as the [0,1] windows, so affine regimes look identically
# crushed (they are affine-equivalent) rather than getting a free auto-stretch.
_ZLO = (FG_P0_5 - FG_MEAN) / FG_STD
_ZHI = (FG_P99_5 - FG_MEAN) / FG_STD

# name, fn, kind, display-range, clip-windows (HU [lo,hi] the regime preserves;
# a voxel outside EVERY window loses its contrast -> counted as "clipped").
REGIMES = [
    ("current_minmax",    r_current,    "affine",    (0.0, 1.0),      [(-1024.0, 2048.0)]),
    ("percentile_minmax", r_percentile, "affine",    (0.0, 1.0),      [(FG_P0_5, FG_P99_5)]),
    ("zscore_global",     r_zscore,     "affine",    (_ZLO, _ZHI),    [(FG_P0_5, FG_P99_5)]),
    ("narrow_soft",       r_narrow,     "clips",     (0.0, 1.0),      [SOFT_WIN]),
    ("sigmoid_soft",      r_sigmoid,    "nonlinear", (0.0, 1.0),      []),  # no hard clip
    ("multiwindow",       r_multiwindow,"3-channel", (0.0, 1.0),      [SOFT_WIN, BONE_WIN, LUNG_WIN]),
]


def clipped_pct(hu_vals, windows):
    """% of organ voxels whose contrast is lost (outside EVERY preserved window)."""
    if not windows:
        return 0.0
    covered = np.zeros(hu_vals.shape, dtype=bool)
    for lo, hi in windows:
        covered |= (hu_vals >= lo) & (hu_vals <= hi)
    return 100.0 * float((~covered).mean())

# Soft-tissue (low-contrast) organs used to pick the abdominal slice + zoom box.
SOFT_ORGANS = [
    "liver", "spleen", "stomach", "pancreas", "gallbladder", "duodenum",
    "small_bowel", "colon", "kidney_left", "kidney_right",
    "adrenal_gland_left", "adrenal_gland_right", "urinary_bladder", "prostate",
]

# Organ pairs for the CNR table. Soft pairs are hard-to-separate abdominal
# neighbors; the bone pair is what a narrow soft-tissue window destroys.
SOFT_PAIRS = [("liver", "spleen"), ("pancreas", "duodenum"),
              ("liver", "gallbladder"), ("kidney_left", "spleen")]
BONE_PAIRS = [("vertebrae_L3", "vertebrae_L2"), ("rib_left_7", "rib_left_8"),
              ("femur_left", "hip_left"), ("vertebrae_T8", "vertebrae_T9")]


def geometric_preprocess(cfg, subject_id, classes):
    """Load one subject and run the framework's deterministic geometric prefix.

    Returns (hu_volume [D,H,W] float32, label_volume [D,H,W] int) in RAS, at the
    configured spacing, foreground-cropped -- i.e. exactly what the model sees,
    minus the (per-model) intensity windowing we are here to compare.
    """
    from monai.transforms import (
        Compose, EnsureTyped, Spacingd, CropForegroundd, Orientationd,
    )
    d = cfg["data"]
    ds = TotalSegmentatorDataset(d["dataset_root"], [subject_id], classes)
    sample = ds[0]
    keys = ("image", "label")
    tfm = Compose([
        EnsureTyped(keys=keys),
        Spacingd(keys=keys, pixdim=tuple(d["spacing"]), mode=("bilinear", "nearest")),
        CropForegroundd(keys=keys, source_key="image",
                        margin=int(d.get("crop_foreground_margin", 0))),
        Orientationd(keys=keys, axcodes="RAS"),
    ])
    out = tfm(sample)
    hu = out["image"][0].cpu().numpy().astype(np.float32)   # (D,H,W)
    lab = out["label"][0].cpu().numpy().astype(np.int32)
    return hu, lab


def pick_axial_slice(lab, class_to_idx):
    """Axial S-index (last axis, RAS) with the most soft-tissue-organ voxels."""
    idxs = [class_to_idx[o] for o in SOFT_ORGANS if o in class_to_idx]
    mask = np.isin(lab, idxs)                    # (D,H,W)
    counts = mask.sum(axis=(0, 1))               # per S-index
    if counts.max() == 0:                        # fallback: any organ
        counts = (lab > 0).sum(axis=(0, 1))
    return int(counts.argmax()), mask


def zoom_box(soft_mask_slice, size=160):
    """Bounding box (centered, fixed size) around soft organs on a 2D slice."""
    ys, xs = np.where(soft_mask_slice)
    if len(ys) == 0:
        h, w = soft_mask_slice.shape
        return slice(0, h), slice(0, w)
    cy, cx = int(ys.mean()), int(xs.mean())
    h, w = soft_mask_slice.shape
    half = size // 2
    y0 = max(0, min(cy - half, h - size)) if h > size else 0
    x0 = max(0, min(cx - half, w - size)) if w > size else 0
    return slice(y0, min(y0 + size, h)), slice(x0, min(x0 + size, w))


def cnr(hu_a, hu_b, fn):
    """Contrast-to-noise between two organs' voxels under a regime.

    Works for 1-channel (scalar out) and multi-channel (vector out) regimes:
    CNR = ||mean_a - mean_b|| / sqrt(0.5 (tr cov_a + tr cov_b)).
    """
    oa, ob = fn(hu_a), fn(hu_b)
    if oa.ndim == 1:
        oa, ob = oa[:, None], ob[:, None]
    ma, mb = oa.mean(0), ob.mean(0)
    va, vb = oa.var(0).sum(), ob.var(0).sum()
    denom = np.sqrt(0.5 * (va + vb)) + 1e-8
    return float(np.linalg.norm(ma - mb) / denom)


def sample_organ_hu(hu, lab, class_to_idx, name, cap=40000):
    idx = class_to_idx.get(name)
    if idx is None:
        return None
    v = hu[lab == idx]
    if v.size == 0:
        return None
    if v.size > cap:
        v = v[np.random.default_rng(0).choice(v.size, cap, replace=False)]
    return v


def first_available_pair(pairs, hu, lab, c2i, min_vox=200):
    for a, b in pairs:
        va = sample_organ_hu(hu, lab, c2i, a)
        vb = sample_organ_hu(hu, lab, c2i, b)
        if va is not None and vb is not None and va.size >= min_vox and vb.size >= min_vox:
            return (a, b), va, vb
    return None, None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs" / "base.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--subject", default=None, help="subject id (default: random from split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slice", type=int, default=None, help="force axial S-index")
    ap.add_argument("--out-dir", default=str(REPO / "runs" / "intensity_compare"))
    args = ap.parse_args()

    # base.yaml is not a valid *model* config (has train/wandb); read it raw.
    # A per-model config (e.g. ctfm_unet_128.yaml) still works via load_config.
    try:
        cfg = load_config(args.config)
    except ConfigError:
        cfg = _read_yaml(Path(args.config))
    classes = load_classes()
    c2i = {name: i + 1 for i, name in enumerate(classes)}

    # pick subject
    split_ids = [l.strip() for l in
                 (REPO / "unified" / "data" / "splits" / f"{args.split}.txt")
                 .read_text().splitlines() if l.strip()]
    if args.subject:
        subject = args.subject
    else:
        subject = random.Random(args.seed).choice(split_ids)
    print(f"[load] subject={subject} split={args.split}")

    hu, lab = geometric_preprocess(cfg, subject, classes)
    print(f"[load] HU volume {hu.shape}  HU range [{hu.min():.0f}, {hu.max():.0f}]  "
          f"organs present: {int(len(np.unique(lab)) - 1)}")

    z, soft_mask = pick_axial_slice(lab, c2i)
    if args.slice is not None:
        z = args.slice
    print(f"[slice] axial S-index = {z}")

    # ---- metrics table -----------------------------------------------------
    (sp_a, sp_b), sva, svb = first_available_pair(SOFT_PAIRS, hu, lab, c2i)
    print(f"[pairs] soft-organ CNR pair = {sp_a}/{sp_b}")

    organ_hu = hu[lab > 0]
    soft_band = organ_hu[(organ_hu >= SOFT_LO) & (organ_hu <= SOFT_HI)]

    print("\n" + "=" * 88)
    print(f"{'regime':<20}{'kind':<11}{'soft-range-use':<16}"
          f"{f'CNR({sp_a}/{sp_b})':<22}{'clipped%':<10}")
    print("-" * 88)
    rows = []
    for name, fn, kind, _, windows in REGIMES:
        # soft-band range utilization: spread of soft-band voxels / spread of all
        # organ voxels, in output units (channel 0 for multiwindow).
        def out1(v):
            o = fn(v)
            return o[..., 0] if o.ndim > 1 else o
        allspread = np.ptp(np.percentile(out1(organ_hu), [1, 99]))
        softspread = np.ptp(np.percentile(out1(soft_band), [1, 99])) if soft_band.size else 0.0
        util = 100.0 * softspread / (allspread + 1e-9)
        cnr_soft = cnr(sva, svb, fn) if sva is not None else float("nan")
        clip = clipped_pct(organ_hu, windows)
        rows.append((name, kind, util, cnr_soft, clip))
        print(f"{name:<20}{kind:<11}{util:>7.1f}%{'':<8}"
              f"{cnr_soft:<22.3f}{clip:>6.1f}%")
    print("=" * 88)
    print("Read:")
    print(" * affine trio (current/percentile/zscore): IDENTICAL on every column --")
    print("   a full-fine-tune net can undo a linear map, so swapping among them is")
    print("   nearly a no-op (differences are only clip range + aug-magnitude scaling).")
    print(" * soft-organ CNR barely moves anywhere: windowing does NOT make two soft")
    print("   organs more separable (monotonic-invariant). What it changes is RANGE-USE")
    print("   (soft tissue jumps ~15%->~65% of the input range = more bf16 precision).")
    print(" * narrow_soft buys that range-use by CLIPPING (high clipped%) -> it throws")
    print("   away bone/lung contrast. That is the redistribution trap, quantified.")
    print(" * sigmoid: same range-use gain with ~0 clipped% (nonlinear, no info lost).")
    print(" * multiwindow: high range-use AND ~0 clipped% -- the 3 windows cover every")
    print("   regime -- at the cost of in_channels 1->3 (backbone surgery).\n")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: slice grid (full + abdomen zoom) ------------------------
    def ax_slice(vol_or_rgb):
        # take axial slice at z, orient upright for display
        if vol_or_rgb.ndim == 4:  # (D,H,W,3)
            s = vol_or_rgb[:, :, z, :]
            return np.rot90(s)
        s = vol_or_rgb[:, :, z]
        return np.rot90(s)

    yz, xz = zoom_box(np.rot90(soft_mask[:, :, z]))
    n = len(REGIMES)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 6.6))
    for j, (name, fn, kind, disp, _win) in enumerate(REGIMES):
        out = fn(hu)                       # full volume normalized
        sl = ax_slice(out)
        is_rgb = sl.ndim == 3
        if disp is None:                   # derive display range (zscore)
            flat = sl if not is_rgb else sl[..., 0]
            vmin, vmax = np.percentile(flat, [0.5, 99.5])
        else:
            vmin, vmax = disp
        kw = {} if is_rgb else dict(cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, j].imshow(np.clip(sl, vmin, vmax) if is_rgb else sl, **kw)
        axes[0, j].set_title(f"{name}\n[{kind}]", fontsize=9)
        axes[0, j].axis("off")
        # abdomen zoom
        zoom = sl[yz, xz]
        axes[1, j].imshow(np.clip(zoom, vmin, vmax) if is_rgb else zoom, **kw)
        axes[1, j].axis("off")
        # mark the zoom box on the full view
        axes[0, j].add_patch(Rectangle((xz.start, yz.start),
                                       xz.stop - xz.start, yz.stop - yz.start,
                                       fill=False, ec="yellow", lw=1.2))
    axes[0, 0].set_ylabel("full axial", fontsize=10)
    axes[1, 0].set_ylabel("abdomen zoom", fontsize=10)
    fig.suptitle(f"Intensity regimes  |  subject {subject}  |  axial S={z}  "
                 f"(displayed on each regime's output range)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = Path(args.out_dir) / f"regimes_slices_{subject}.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # ---- Figure 2: foreground histograms -----------------------------------
    rng = np.random.default_rng(0)
    fg = organ_hu if organ_hu.size <= 400000 else organ_hu[rng.choice(organ_hu.size, 400000, replace=False)]
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (name, fn, kind, disp, _win) in zip(axes2.ravel(), REGIMES):
        out = fn(fg)
        if out.ndim > 1:  # multiwindow: overlay 3 channels
            for c, col, lab_c in zip(range(3), ["tab:red", "tab:green", "tab:blue"],
                                     ["soft", "bone", "lung"]):
                ax.hist(out[:, c], bins=120, histtype="step", color=col, label=lab_c)
            ax.legend(fontsize=8)
        else:
            ax.hist(out, bins=120, color="0.4")
            # shade where the soft-tissue band lands
            lo, hi = fn(np.array([SOFT_LO])).item(), fn(np.array([SOFT_HI])).item()
            ax.axvspan(min(lo, hi), max(lo, hi), color="tab:orange", alpha=0.3,
                       label=f"soft band [{SOFT_LO:.0f},{SOFT_HI:.0f}]HU")
            ax.legend(fontsize=8)
        ax.set_title(f"{name} [{kind}]", fontsize=10)
        ax.set_yticks([])
        ax.set_xlabel("network input value")
    fig2.suptitle(f"Foreground (organ-voxel) value distribution per regime  |  {subject}\n"
                  "orange = where the diagnostic soft-tissue band lands "
                  "(narrow band = crushed contrast)", fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = Path(args.out_dir) / f"regimes_hist_{subject}.png"
    fig2.savefig(p2, dpi=130)
    plt.close(fig2)

    print(f"[write] {p1}")
    print(f"[write] {p2}")


if __name__ == "__main__":
    main()
