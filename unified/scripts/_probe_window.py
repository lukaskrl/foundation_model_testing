"""Arm W at probe cost: what does forcing a shared HU window cost each encoder?

Between-encoder ranking from a linear probe is noisy. A WINDOW comparison is
paired — same encoder, same voxels, same readout, only the input window differs —
so the within-encoder delta is far better conditioned than the cross-encoder
ranking. Reported per encoder, and broken out by anatomy group, because the
soft-tissue window's damage is concentrated in air-filled structures.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from _probe_union import fit, MODELS

WDS = [1e-1, 1.0, 10.0, 100.0]
CLASSES = [l.strip() for l in open(REPO / "unified/data/classes.txt") if l.strip()]
BONE = ("rib_", "vertebrae_", "femur", "hip", "humerus", "clavicula", "scapula",
        "sacrum", "skull", "sternum", "costal_cartilages")


def group_of(name):
    if name.startswith("lung") or name in ("trachea",):
        return "air"
    if name.startswith(BONE):
        return "bone"
    return "soft"


def per_class_recall(pred, y, n_cls):
    out = {}
    for c in torch.unique(y).tolist():
        if c == 0:
            continue
        out[c] = float((pred[y == c] == c).float().mean())
    return out


def fit_full(Xtr, ytr, Xev, yev, n_cls, dev, wd, steps, seed):
    torch.manual_seed(seed)
    W = torch.zeros(Xtr.shape[1], n_cls, device=dev, requires_grad=True)
    b = torch.zeros(n_cls, device=dev, requires_grad=True)
    cnt = torch.bincount(ytr, minlength=n_cls).float()
    cw = torch.where(cnt > 0, 1.0 / cnt.clamp_min(1), torch.zeros_like(cnt))
    cw = cw / cw.sum() * (cnt > 0).sum()
    opt = torch.optim.AdamW([W, b], lr=3e-2, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lf = torch.nn.CrossEntropyLoss(weight=cw)
    for _ in range(steps):
        opt.zero_grad(); lf(Xtr @ W + b, ytr).backward(); opt.step(); sch.step()
    with torch.no_grad():
        pred = (Xev @ W + b).argmax(1)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="tag=dir pairs")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    roots = {r.split("=")[0]: Path(r.split("=", 1)[1]) for r in a.roots}
    meta = np.load(list(roots.values())[0] / "meta.npz", allow_pickle=True)
    y = meta["y"].astype(np.int64)
    subj = meta["subj_of_patch"][meta["patch_id"]].astype(np.int64)
    yt = torch.as_tensor(y, device=dev); usubj = np.unique(subj); n_cls = 118
    res = {}
    for seed in range(a.seeds):
        rng = np.random.default_rng(200 + seed); pm = rng.permutation(usubj); n = len(pm)
        te = set(pm[:int(.3*n)].tolist()); va = set(pm[int(.3*n):int(.5*n)].tolist())
        m_te = np.isin(subj, list(te)); m_va = np.isin(subj, list(va)); m_tr = ~(m_te | m_va)
        I = {k: torch.as_tensor(np.where(v)[0], device=dev)
             for k, v in dict(tr=m_tr, va=m_va, te=m_te).items()}
        ys = {k: yt[v] for k, v in I.items()}
        for m in MODELS:
            for tag, root in roots.items():
                f = root / f"X_{m}.npy"
                if not f.exists(): continue
                T = torch.as_tensor(np.load(f), device=dev, dtype=torch.float32)
                tr = T[I["tr"]]
                mu, sd = tr.mean(0, keepdim=True), tr.std(0, keepdim=True).clamp_min(1e-5)
                Z = {k: (T[I[k]] - mu) / sd for k in I}; del T
                best = (-1, None)
                for wd in WDS:
                    p = fit_full(Z["tr"], ys["tr"], Z["va"], ys["va"], n_cls, dev, wd, a.steps, seed)
                    pres = torch.unique(ys["va"])
                    v = float(np.mean([(p[ys["va"] == c] == c).float().mean().item() for c in pres]))
                    if v > best[0]: best = (v, wd)
                p = fit_full(Z["tr"], ys["tr"], Z["te"], ys["te"], n_cls, dev, best[1], a.steps, seed)
                rec = per_class_recall(p, ys["te"], n_cls)
                g = {"air": [], "bone": [], "soft": []}
                for c, r in rec.items():
                    if 1 <= c <= len(CLASSES):
                        g[group_of(CLASSES[c-1])].append(r)
                row = dict(overall=float(np.mean(list(rec.values()))), wd=best[1],
                           **{k: (float(np.mean(v)) if v else float("nan")) for k, v in g.items()})
                res.setdefault(m, {}).setdefault(tag, []).append(row)
                print(f"[s{seed}] {m:20s} {tag:7s} wd={best[1]:<5g} overall={row['overall']:.4f} "
                      f"air={row['air']:.4f} soft={row['soft']:.4f} bone={row['bone']:.4f}", flush=True)
                del Z; torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
