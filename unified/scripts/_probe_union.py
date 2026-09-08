"""Does concat(all teachers) beat the best single teacher, at full dimension
with per-condition tuned regularisation?

Three-way SUBJECT-wise split: train / val (picks weight decay) / test (reported).
Conditions: every single encoder; best-single + each other; all 12; and a
DUPLICATE control (best encoder concatenated with itself), which adds parameters
and zero information — the null for any "more dims helped" explanation.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

MODELS = ["ctfm", "vista3d", "voco_b", "voco_h", "suprem_unet", "suprem_segresnet",
          "suprem_swinunetr", "biomedparse", "ctclip_layerwise", "merlin",
          "sam_med3d", "dino3d_layerwise"]
WDS = [1e-2, 1e-1, 1.0, 10.0, 100.0]


def fit(Xtr, ytr, Xev, yev, n_cls, dev, wd, steps, seed):
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
    pres = torch.unique(yev)
    return float(np.mean([(pred[yev == c] == c).float().mean().item() for c in pres]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    d = Path(a.feats)
    meta = np.load(d / "meta.npz", allow_pickle=True)
    y = meta["y"].astype(np.int64); subj = meta["subj_of_patch"][meta["patch_id"]].astype(np.int64)
    Xn = {m: np.load(d / f"X_{m}.npy") for m in MODELS if (d / f"X_{m}.npy").exists()}
    names = list(Xn); usubj = np.unique(subj)
    yt = torch.as_tensor(y, device=dev); n_cls = 118
    print(f"{len(names)} encoders, {len(y)} voxels, {len(usubj)} subjects, "
          f"total dim {sum(v.shape[1] for v in Xn.values())}", flush=True)

    out = {"seeds": []}
    for seed in range(a.seeds):
        rng = np.random.default_rng(200 + seed)
        pm = rng.permutation(usubj)
        n = len(pm); te_s, va_s = set(pm[:int(.3*n)].tolist()), set(pm[int(.3*n):int(.5*n)].tolist())
        m_te = np.isin(subj, list(te_s)); m_va = np.isin(subj, list(va_s))
        m_tr = ~(m_te | m_va)
        I = {k: torch.as_tensor(np.where(v)[0], device=dev) for k, v in
             dict(tr=m_tr, va=m_va, te=m_te).items()}
        ys = {k: yt[v] for k, v in I.items()}
        Z = {}
        for m in names:
            T = torch.as_tensor(Xn[m], device=dev, dtype=torch.float32)
            tr = T[I["tr"]]
            mu, sd = tr.mean(0, keepdim=True), tr.std(0, keepdim=True).clamp_min(1e-5)
            Z[m] = {k: (T[I[k]] - mu) / sd for k in I}
            del T
        torch.cuda.empty_cache()

        def run(sel, tag, dup=False):
            cat = lambda k: torch.cat([Z[m][k] for m in sel] * (2 if dup else 1), 1)
            tr, va, te = cat("tr"), cat("va"), cat("te")
            best = (-1, None)
            for wd in WDS:
                v = fit(tr, ys["tr"], va, ys["va"], n_cls, dev, wd, a.steps, seed)
                if v > best[0]: best = (v, wd)
            t = fit(tr, ys["tr"], te, ys["te"], n_cls, dev, best[1], a.steps, seed)
            print(f"[s{seed}] {tag:34s} dim={tr.shape[1]:6d} wd={best[1]:<6g} "
                  f"val={best[0]:.4f} TEST={t:.4f}", flush=True)
            return dict(tag=tag, dim=int(tr.shape[1]), wd=best[1], val=best[0], test=t)

        res = [run([m], m) for m in names]
        base = max(res, key=lambda r: r["val"])["tag"]
        print(f"[s{seed}] best single by VAL = {base}", flush=True)
        res.append(run(names, "ALL_12"))
        res.append(run([base], f"{base}_DUPLICATED", dup=True))
        for m in names:
            if m != base:
                res.append(run([base, m], f"{base}+{m}"))
        out["seeds"].append({"seed": seed, "base": base, "results": res})
        del Z; torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
