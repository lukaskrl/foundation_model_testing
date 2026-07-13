"""Dataset fingerprint for TotalSegmentator — nnU-Net-style, CPU-only.

Computes the dataset statistics the fixed `base.yaml` recipe currently GUESSES,
so recipe choices become data-driven. Fingerprints the TRAIN split only (never
peek at val/test when setting the recipe). Three products, each feeding a
specific recipe lever discussed in the Tier-1 plan:

  1. Per-class voxel counts / presence  -> `train.loss.ce_class_weights`
     (inverse-frequency weighting for the small/rare-class laggards).
  2. Foreground intensity distribution  -> per-encoder intensity window / z-score
     ablation (the fixed [-1024,2048] window crushes soft-tissue contrast).
  3. Per-class physical bounding-box extent -> patch/FOV sizing. Directly tests
     the "96^3 @ 1.5mm = 144mm patch can't see a whole ribcage" hypothesis.

Also reports raw spacing/shape distributions (to sanity-check the 1.5mm iso
resample) and emits a ready-to-paste `ce_class_weights` YAML block.

Usage:
    python -m scripts.dataset_fingerprint                 # all train cases
    python -m scripts.dataset_fingerprint --max-cases 50  # quick smoke
    python -m scripts.dataset_fingerprint --intensity-cases 200 --out runs/fingerprint/fp.json

CPU-only and read-only w.r.t. the dataset — safe to run while GPUs are busy.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_yaml(p):
    import yaml
    with open(p) as f:
        return yaml.safe_load(f)


def build_subject_ids(meta_csv, split):
    ids = []
    with open(meta_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row.get("split", "").strip().lower() == split:
                sid = row.get("image_id", "").strip()
                if sid:
                    ids.append(sid)
    return ids


def load_label(subject_dir, num_classes):
    """Return (label int array, spacing (sx,sy,sz) mm, shape) or None."""
    import nibabel as nib
    merged = subject_dir / "label.nii.gz"
    ct = subject_dir / "ct.nii.gz"
    if not merged.exists() or not ct.exists():
        return None
    lab_nii = nib.load(str(merged))
    label = np.asanyarray(lab_nii.dataobj)          # native dtype, no float64 cast
    spacing = tuple(float(z) for z in lab_nii.header.get_zooms()[:3])
    return label, spacing, tuple(int(s) for s in label.shape)


def sample_fg_intensity(subject_dir, label, cap):
    """Sample up to `cap` foreground (label>0) HU intensities from the CT."""
    import nibabel as nib
    ct = np.asanyarray(nib.load(str(subject_dir / "ct.nii.gz")).dataobj)
    if ct.shape != label.shape:
        return None
    fg = label > 0
    n = int(fg.sum())
    if n == 0:
        return np.empty(0, dtype=np.float32)
    vals = ct[fg].astype(np.float32, copy=False)
    if n > cap:
        # Deterministic stride subsample (no RNG dependence for reproducibility).
        vals = vals[:: max(1, n // cap)][:cap]
    return vals


def main():
    base = _load_yaml(REPO / "configs" / "base.yaml")["data"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", default=base["dataset_root"])
    ap.add_argument("--meta-csv", default=base["meta_csv"])
    ap.add_argument("--classes-file", default=str(REPO / base["classes_file"]))
    ap.add_argument("--num-classes", type=int, default=int(base["num_classes"]))
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-cases", type=int, default=0, help="0 = all")
    ap.add_argument("--intensity-cases", type=int, default=200,
                    help="# evenly-spaced cases to sample CT intensities from")
    ap.add_argument("--intensity-per-case", type=int, default=20000)
    ap.add_argument("--patch-mm", type=float, default=96 * 1.5,
                    help="physical patch size (mm) to compare class extents against")
    ap.add_argument("--weight-cap", type=float, default=5.0)
    ap.add_argument("--out", default=str(REPO / "runs" / "fingerprint"
                                         / "totalseg_train_fingerprint.json"))
    args = ap.parse_args()

    root = Path(args.dataset_root)
    NC = args.num_classes
    class_names = [l.strip() for l in open(args.classes_file) if l.strip()]
    assert len(class_names) == NC - 1, f"{len(class_names)} names != num_classes-1 ({NC-1})"

    ids = build_subject_ids(args.meta_csv, args.split)
    if args.max_cases:
        ids = ids[: args.max_cases]
    n_total = len(ids)
    intensity_idx = set(np.linspace(0, n_total - 1, min(args.intensity_cases, n_total),
                                    dtype=int).tolist()) if n_total else set()
    print(f"[fingerprint] split={args.split} cases={n_total} "
          f"intensity-sampled={len(intensity_idx)}")

    # Accumulators
    voxel_counts = np.zeros(NC, dtype=np.int64)     # total voxels per class
    present = np.zeros(NC, dtype=np.int64)          # #cases where class present
    extents_mm = [[] for _ in range(NC)]            # per-class list of max-axis extent (mm)
    volumes_mm3 = [[] for _ in range(NC)]           # per-class list of physical volume
    spacings, shapes, phys_sizes = [], [], []
    fg_samples = []
    fg_cap_total = 5_000_000
    n_ok = n_fail = 0
    t0 = time.time()

    from scipy import ndimage
    for i, sid in enumerate(ids):
        try:
            res = load_label(root / sid, NC)
            if res is None:
                n_fail += 1
                continue
            label, spacing, shape = res
            spacings.append(spacing)
            shapes.append(shape)
            phys_sizes.append(tuple(shape[a] * spacing[a] for a in range(3)))
            vox_vol = float(np.prod(spacing))

            counts = np.bincount(label.reshape(-1), minlength=NC)[:NC]
            voxel_counts += counts
            present += (counts > 0).astype(np.int64)

            # Per-class bbox in ONE pass; objs[k] is the slice tuple for label k+1.
            objs = ndimage.find_objects(label.astype(np.int32, copy=False))
            for k, sl in enumerate(objs):          # k -> class index k+1
                if sl is None:
                    continue
                ext_mm = [(sl[a].stop - sl[a].start) * spacing[a] for a in range(3)]
                extents_mm[k + 1].append(max(ext_mm))
                volumes_mm3[k + 1].append(float(counts[k + 1]) * vox_vol)

            if i in intensity_idx and len(fg_samples) < fg_cap_total:
                v = sample_fg_intensity(root / sid, label, args.intensity_per_case)
                if v is not None and len(v):
                    fg_samples.append(v)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"  ! {sid}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_total}  ({(time.time()-t0)/ (i+1):.2f}s/case)")

    print(f"[fingerprint] done: ok={n_ok} fail={n_fail} in {time.time()-t0:.0f}s")

    # ---- aggregate ------------------------------------------------------
    def med_axes(rows):
        a = np.array(rows, dtype=np.float64)
        return {"median": np.median(a, 0).round(3).tolist(),
                "min": a.min(0).round(3).tolist(), "max": a.max(0).round(3).tolist()}

    fg = np.concatenate(fg_samples) if fg_samples else np.empty(0, np.float32)
    intensity = {}
    if fg.size:
        intensity = {
            "n_samples": int(fg.size),
            "mean": float(fg.mean()), "std": float(fg.std()),
            "min": float(fg.min()), "max": float(fg.max()),
            "p0_5": float(np.percentile(fg, 0.5)),
            "p50": float(np.percentile(fg, 50)),
            "p99_5": float(np.percentile(fg, 99.5)),
        }

    total_vox = int(voxel_counts.sum())
    fg_vox = int(voxel_counts[1:].sum())
    per_class = []
    for c in range(NC):
        name = "background" if c == 0 else class_names[c - 1]
        ext = extents_mm[c]
        vol = volumes_mm3[c]
        per_class.append({
            "index": c, "name": name,
            "n_present": int(present[c]),
            "presence_freq": round(present[c] / max(1, n_ok), 4),
            "total_voxels": int(voxel_counts[c]),
            "frac_of_all": round(voxel_counts[c] / max(1, total_vox), 6),
            "frac_of_fg": (round(voxel_counts[c] / max(1, fg_vox), 6) if c else None),
            "extent_mm_median": round(float(np.median(ext)), 1) if ext else None,
            "extent_mm_p90": round(float(np.percentile(ext, 90)), 1) if ext else None,
            "volume_mm3_median": round(float(np.median(vol)), 1) if vol else None,
        })

    # ---- ce_class_weights recommendations -------------------------------
    counts_safe = np.maximum(voxel_counts.astype(np.float64), 1.0)
    def make_weights(power):
        w = counts_safe ** (-power)                 # inverse-freq^power
        fg_mean = w[1:].mean()
        w = w / fg_mean                             # normalize so mean(fg)=1
        w = np.minimum(w, args.weight_cap)          # cap extreme rare-class weights
        w[0] = 1.0                                  # keep background at 1 (FP calibration)
        return w.round(3).tolist()
    weights = {
        "inverse_sqrt_cap": make_weights(0.5),      # recommended: gentle
        "inverse_freq_cap": make_weights(1.0),      # aggressive
    }

    # ---- FOV analysis ---------------------------------------------------
    exceed = [pc for pc in per_class[1:]
              if pc["extent_mm_median"] and pc["extent_mm_median"] > args.patch_mm]
    exceed.sort(key=lambda x: -x["extent_mm_median"])

    fingerprint = {
        "split": args.split, "n_cases": n_ok, "n_fail": n_fail,
        "spacing_mm": med_axes(spacings) if spacings else {},
        "shape_voxels": med_axes(shapes) if shapes else {},
        "phys_size_mm": med_axes(phys_sizes) if phys_sizes else {},
        "foreground_intensity_hu": intensity,
        "total_voxels": total_vox, "foreground_voxels": fg_vox,
        "patch_mm": args.patch_mm,
        "n_classes_exceeding_patch_fov": len(exceed),
        "recommended_ce_class_weights": weights,
        "per_class": per_class,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(fingerprint, indent=2))
    print(f"[fingerprint] wrote {outp}")

    # ---- human-readable summary ----------------------------------------
    print("\n================ SUMMARY ================")
    if fingerprint["spacing_mm"]:
        print(f"raw spacing (mm)   median={fingerprint['spacing_mm']['median']}  "
              f"(base.yaml resamples to {base['spacing']})")
        print(f"raw shape (vox)    median={fingerprint['shape_voxels']['median']}")
        print(f"phys size (mm)     median={fingerprint['phys_size_mm']['median']}")
    if intensity:
        print(f"\nFOREGROUND HU: mean={intensity['mean']:.1f} std={intensity['std']:.1f}  "
              f"clip[p0.5,p99.5]=[{intensity['p0_5']:.0f}, {intensity['p99_5']:.0f}]")
        print(f"  -> current window [-1024,2048] wastes range; a z-score or "
              f"[{intensity['p0_5']:.0f},{intensity['p99_5']:.0f}] clip is data-driven")
    print(f"\nRAREST classes (lowest presence_freq):")
    for pc in sorted(per_class[1:], key=lambda x: x["presence_freq"])[:8]:
        print(f"  {pc['name']:28s} present {pc['n_present']:>4}/{n_ok}  "
              f"freq={pc['presence_freq']:.3f}  frac_fg={pc['frac_of_fg']:.2e}")
    print(f"\nFOV: {len(exceed)}/{NC-1} classes exceed the {args.patch_mm:.0f}mm patch "
          f"(need more context than one patch sees):")
    for pc in exceed[:12]:
        print(f"  {pc['name']:28s} median extent {pc['extent_mm_median']:.0f}mm "
              f"(p90 {pc['extent_mm_p90']:.0f}mm)")
    print(f"\nRecommended ce_class_weights (inverse_sqrt_cap{args.weight_cap:.0f}) — paste under train.loss:")
    w = weights["inverse_sqrt_cap"]
    print("  ce_class_weights: [" + ", ".join(f"{x:g}" for x in w) + "]")
    print("=========================================")


if __name__ == "__main__":
    main()
