"""Extract aligned per-encoder hypercolumn features on real TotalSegmentator voxels.

Voxel correspondence across encoders is the whole point, so:
  * one shared canonical frame: det prefix (Spacing -> CropForeground) + Orientation(RAS)
  * intensity normalization is applied to the WHOLE canonical volume before
    patching (it is orientation-invariant, and percentile/znorm modes are
    volume-level statistics that would be wrong per-patch)
  * each encoder's axcodes permute/flip is applied to the patch, and the INVERSE
    is applied to its feature maps, putting every encoder back in RAS
  * all native levels are resampled to a common stride-4 grid (24^3 at patch 96)
    and concatenated -> one hypercolumn per voxel per encoder
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from monai.data import MetaTensor
from monai.transforms import Compose, Spacingd, CropForegroundd, Orientationd

from unified.utils import load_config
from unified.data.transforms import _resolved_preprocessing, _orient_intensity_list
from unified.data.totalsegmentator import TotalSegmentatorDataset, load_classes
import unified.models.backbones  # noqa: F401
from unified.models import build_backbone

MODELS = ["ctfm", "vista3d", "voco_b", "voco_h", "suprem_unet", "suprem_segresnet",
          "suprem_swinunetr", "biomedparse", "ctclip_layerwise", "merlin",
          "sam_med3d", "dino3d_layerwise"]


def ornt_from_ras(axcodes):
    src = nib.orientations.axcodes2ornt("RAS")
    dst = nib.orientations.axcodes2ornt(axcodes)
    return nib.orientations.ornt_transform(src, dst)


def apply_ornt(t, xfm):
    """t: (B,C,D,H,W) in the source frame -> destination frame."""
    perm = [int(xfm[i, 0]) for i in range(3)]
    t = t.permute(0, 1, 2 + perm[0], 2 + perm[1], 2 + perm[2])
    flips = [2 + i for i in range(3) if xfm[i, 1] < 0]
    return torch.flip(t, flips) if flips else t.contiguous()


def inv_ornt(axcodes):
    src = nib.orientations.axcodes2ornt("RAS")
    dst = nib.orientations.axcodes2ornt(axcodes)
    return nib.orientations.ornt_transform(dst, src)


def intensity_only(cfg):
    """The encoder's intensity transform, with Orientationd dropped."""
    ts = _orient_intensity_list(cfg)
    return Compose([t for t in ts if not isinstance(t, Orientationd)])


def build_patches(cfg, n_subj, n_patch, patch, seed, split="val"):
    d = cfg["data"]
    ids = [l.strip() for l in open(REPO / "unified/data/splits" / f"{split}.txt") if l.strip()]
    ds = TotalSegmentatorDataset(d["dataset_root"], ids[:n_subj], classes=load_classes(d["classes_file"]))
    det = Compose([
        Spacingd(keys=("image", "label"), pixdim=tuple(d["spacing"]), mode=("bilinear", "nearest")),
        CropForegroundd(keys=("image", "label"), source_key="image",
                        margin=int(d.get("crop_foreground_margin", 0))),
        Orientationd(keys=("image", "label"), axcodes="RAS"),
    ])
    rng = np.random.default_rng(seed)
    vols, labs, coords, sids = [], [], [], []
    for i in range(len(ds)):
        s = ds[i]
        aff = torch.as_tensor(s["image_meta_dict"]["affine"])
        out = det({"image": MetaTensor(s["image"], affine=aff),
                   "label": MetaTensor(s["label"], affine=aff)})
        img = out["image"].as_tensor().float()          # (1,D,H,W) raw HU, RAS
        lab = out["label"].as_tensor().long()
        D, H, W = img.shape[1:]
        if min(D, H, W) < patch:
            pad = [max(0, patch - W), 0, max(0, patch - H), 0, max(0, patch - D), 0]
            pad = [pad[0], 0, pad[2], 0, pad[4], 0]
            img = F.pad(img, [0, max(0, patch - W), 0, max(0, patch - H), 0, max(0, patch - D)])
            lab = F.pad(lab, [0, max(0, patch - W), 0, max(0, patch - H), 0, max(0, patch - D)])
            D, H, W = img.shape[1:]
        fg = torch.nonzero(lab[0] > 0, as_tuple=False)
        if fg.numel() == 0:
            continue
        pick = rng.choice(len(fg), size=n_patch, replace=len(fg) < n_patch)
        cs = []
        for p in pick:
            c = fg[int(p)].tolist()
            c = [int(np.clip(c[k] - patch // 2, 0, [D, H, W][k] - patch)) for k in range(3)]
            cs.append(c)
        vols.append(img); labs.append(lab); coords.append(cs); sids.append(s["id"])
        print(f"  subj {s['id']} shape={(D,H,W)} patches={len(cs)}", flush=True)
    return vols, labs, coords, sids


def sample_voxels(lab_patch, n_vox, rng, fg_frac=0.75):
    """lab_patch: (96,96,96) long -> stride-4 grid labels + flat indices."""
    l24 = lab_patch[2::4, 2::4, 2::4].reshape(-1)          # 24^3
    fg = torch.nonzero(l24 > 0, as_tuple=False).flatten().numpy()
    bg = torch.nonzero(l24 == 0, as_tuple=False).flatten().numpy()
    n_fg = min(len(fg), int(n_vox * fg_frac))
    n_bg = min(len(bg), n_vox - n_fg)
    idx = np.concatenate([rng.choice(fg, n_fg, replace=False) if n_fg else np.array([], int),
                          rng.choice(bg, n_bg, replace=False) if n_bg else np.array([], int)])
    return idx.astype(np.int64), l24.numpy()[idx.astype(np.int64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subj", type=int, default=24)
    ap.add_argument("--n-patch", type=int, default=4)
    ap.add_argument("--n-vox", type=int, default=400)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", nargs=2, type=float, default=None,
                    help="override every encoder's HU window with a_min a_max (Arm W)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    base = load_config(str(REPO / "configs/models/ctfm.yaml"))
    print("building canonical patches...", flush=True)
    vols, labs, coords, sids = build_patches(base, a.n_subj, a.n_patch, a.patch, a.seed)

    # Fixed voxel sample, shared by every encoder.
    rng = np.random.default_rng(a.seed + 1)
    vox_idx, vox_y, patch_id = [], [], []
    P = a.patch
    for si in range(len(vols)):
        for pi, c in enumerate(coords[si]):
            lp = labs[si][0, c[0]:c[0]+P, c[1]:c[1]+P, c[2]:c[2]+P]
            idx, y = sample_voxels(lp, a.n_vox, rng)
            vox_idx.append(idx); vox_y.append(y)
            patch_id.append(np.full(len(idx), len(patch_id), dtype=np.int32))
    y_all = np.concatenate(vox_y); pid_all = np.concatenate(patch_id)
    subj_of_patch = np.concatenate([[si] * len(coords[si]) for si in range(len(vols))])
    np.savez(outdir / "meta.npz", y=y_all, patch_id=pid_all,
             subj_of_patch=subj_of_patch, sids=np.array(sids))
    print(f"voxels sampled: {len(y_all)}  fg={int((y_all>0).sum())}  "
          f"classes={len(np.unique(y_all))}", flush=True)

    for name in a.models:
        fout = outdir / f"X_{name}.npy"
        if fout.exists():
            print(f"{name}: cached"); continue
        cfg = load_config(str(REPO / f"configs/models/{name}.yaml"))
        m = cfg["model"]
        if a.window is not None:
            cfg["model"].setdefault("preprocessing", {})["intensity"] = dict(
                mode="range", a_min=a.window[0], a_max=a.window[1],
                b_min=0.0, b_max=1.0, clip=True)
        axcodes, _ = _resolved_preprocessing(cfg)
        fwd_x, inv_x = ornt_from_ras(axcodes), inv_ornt(axcodes)
        itx = intensity_only(cfg)
        t0 = time.time()
        bb = build_backbone(m["name"], weights=m.get("weights"), **m.get("kwargs", {}))
        bb = bb.to(dev).eval()
        chunks = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            k = 0
            for si in range(len(vols)):
                vi = itx({"image": MetaTensor(vols[si].clone())})["image"]
                vi = vi.as_tensor() if hasattr(vi, "as_tensor") else vi
                for c in coords[si]:
                    p = vi[None, :, c[0]:c[0]+P, c[1]:c[1]+P, c[2]:c[2]+P].to(dev).float()
                    p = apply_ornt(p, fwd_x)
                    nat = bb.encoder_forward(p) if hasattr(bb, "encoder_forward") else None
                    if nat is None:
                        nat = bb.forward_features(p)
                    feats = [t for t in nat if torch.is_tensor(t) and t.dim() == 5 and t.shape[1] > 1]
                    ups = []
                    for f in feats:
                        f = apply_ornt(f.float(), inv_x)
                        ups.append(F.interpolate(f, size=(a.grid,) * 3, mode="trilinear",
                                                 align_corners=False))
                    hc = torch.cat(ups, 1)[0]                       # (Cdim, 24,24,24)
                    hc = hc.reshape(hc.shape[0], -1).T               # (24^3, Cdim)
                    chunks.append(hc[torch.as_tensor(vox_idx[k], device=dev)].half().cpu())
                    k += 1
        X = torch.cat(chunks).numpy()
        np.save(fout, X)
        print(f"{name:20s} X={X.shape} dim={X.shape[1]:5d} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del bb; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
