"""Does the union of foundation-model features carry more than the best single one?

Reads aligned hypercolumns from _probe_features.py and answers three questions:

  1. CKA / unique-variance matrix  — how redundant are the encoders?
  2. Matched-capacity probe curve  — greedy forward selection over encoders, with
     every condition PCA-projected to the SAME dimension D so the readout has an
     identical parameter count. Any gain is information, not capacity.
  3. Unmatched (full-dim) curve    — the ceiling a fusion adapter could exploit.

Splits are SUBJECT-wise: voxels within a patch are heavily correlated, so the
unit of replication is the subject, not the voxel.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch


def load(dirp, models):
    d = Path(dirp)
    meta = np.load(d / "meta.npz", allow_pickle=True)
    y = meta["y"]; pid = meta["patch_id"]; sop = meta["subj_of_patch"]
    subj = sop[pid]
    X = {}
    for m in models:
        f = d / f"X_{m}.npy"
        if f.exists():
            X[m] = np.load(f)
    return X, y.astype(np.int64), subj.astype(np.int64)


def standardize(tr, te):
    mu = tr.mean(0, keepdim=True)
    sd = tr.std(0, keepdim=True).clamp_min(1e-5)
    return (tr - mu) / sd, (te - mu) / sd


def pca_project(tr, te, D):
    D = min(D, tr.shape[1], tr.shape[0] - 1)
    mu = tr.mean(0, keepdim=True)          # train mean applied to BOTH splits
    tr = tr - mu
    q = min(tr.shape[1], D + 16)
    U, S, V = torch.pca_lowrank(tr, q=q, center=False, niter=4)
    V = V[:, :D]
    return tr @ V, (te - mu) @ V


def probe(Xtr, ytr, Xte, yte, n_cls, dev, steps=600, lr=3e-2, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    Xtr, Xte = standardize(Xtr, Xte)
    W = torch.zeros(Xtr.shape[1], n_cls, device=dev, requires_grad=True)
    b = torch.zeros(n_cls, device=dev, requires_grad=True)
    # inverse-frequency weights: 118 classes at wildly different voxel counts
    cnt = torch.bincount(ytr, minlength=n_cls).float()
    cw = torch.where(cnt > 0, 1.0 / cnt.clamp_min(1), torch.zeros_like(cnt))
    cw = cw / cw.sum() * (cnt > 0).sum()
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lossf = torch.nn.CrossEntropyLoss(weight=cw)
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(Xtr @ W + b, ytr)
        loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        pred = (Xte @ W + b).argmax(1)
    # balanced accuracy over classes present in the test split
    present = torch.unique(yte)
    recalls = [(pred[yte == c] == c).float().mean().item() for c in present]
    return float(np.mean(recalls)), float((pred == yte).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--D", type=int, default=192)
    ap.add_argument("--holdout", type=float, default=0.35)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    MODELS = ["ctfm", "vista3d", "voco_b", "voco_h", "suprem_unet", "suprem_segresnet",
              "suprem_swinunetr", "biomedparse", "ctclip_layerwise", "merlin",
              "sam_med3d", "dino3d_layerwise"]
    Xn, y, subj = load(a.feats, MODELS)
    names = list(Xn)
    usubj = np.unique(subj)
    print(f"encoders={len(names)} voxels={len(y)} subjects={len(usubj)} "
          f"classes={len(np.unique(y))}", flush=True)
    for n in names:
        print(f"  {n:20s} dim={Xn[n].shape[1]}")

    n_cls = 118
    yt = torch.as_tensor(y, device=dev)
    results = {"dims": {n: int(Xn[n].shape[1]) for n in names}, "D": a.D, "runs": []}

    for seed in range(a.seeds):
        rng = np.random.default_rng(100 + seed)
        te_subj = set(rng.choice(usubj, max(2, int(len(usubj) * a.holdout)), replace=False).tolist())
        m_te = np.isin(subj, list(te_subj)); m_tr = ~m_te
        itr = torch.as_tensor(np.where(m_tr)[0], device=dev)
        ite = torch.as_tensor(np.where(m_te)[0], device=dev)
        ytr, yte = yt[itr], yt[ite]
        T = {n: torch.as_tensor(Xn[n], device=dev, dtype=torch.float32) for n in names}
        Z = {}
        for n in names:
            tr, te = standardize(T[n][itr], T[n][ite])
            Z[n] = (tr, te)

        def score(sel, D):
            tr = torch.cat([Z[n][0] for n in sel], 1)
            te = torch.cat([Z[n][1] for n in sel], 1)
            if D:
                tr, te = pca_project(tr, te, D)
            return probe(tr, ytr, te, yte, n_cls, dev, steps=a.steps, seed=seed)

        run = {"seed": seed, "n_test_subj": len(te_subj)}
        singles = {}
        for n in names:
            bal, acc = score([n], a.D)
            singles[n] = bal
            print(f"[seed {seed}] single D={a.D} {n:20s} bal={bal:.4f} acc={acc:.4f}", flush=True)
        run["singles_matched"] = singles

        # greedy forward selection, matched capacity
        order = [max(singles, key=singles.get)]
        curve = [(1, singles[order[0]], order[0])]
        rest = [n for n in names if n not in order]
        while rest:
            best, bn = -1, None
            for n in rest:
                b_, _ = score(order + [n], a.D)
                if b_ > best: best, bn = b_, n
            order.append(bn); rest.remove(bn)
            curve.append((len(order), best, bn))
            print(f"[seed {seed}] greedy K={len(order):2d} +{bn:20s} bal={best:.4f}", flush=True)
        run["greedy_matched"] = [(k, v, n) for k, v, n in curve]

        # unmatched full-dim: best single vs all
        b1, _ = score([order[0]], 0)
        ball, _ = score(names, 0)
        run["full_dim_best_single"] = b1
        run["full_dim_all"] = ball
        print(f"[seed {seed}] FULL-DIM best_single({order[0]})={b1:.4f}  all_{len(names)}={ball:.4f}",
              flush=True)
        results["runs"].append(run)
        del T, Z; torch.cuda.empty_cache()

    # CKA + unique variance on the full sample (no split needed)
    Tall = {n: torch.as_tensor(Xn[n], device=dev, dtype=torch.float32) for n in names}
    Tc = {n: (v - v.mean(0, keepdim=True)) / v.std(0, keepdim=True).clamp_min(1e-5)
          for n, v in Tall.items()}
    def cka(X, Y):
        xy = torch.linalg.matrix_norm(X.T @ Y) ** 2
        return float(xy / (torch.linalg.matrix_norm(X.T @ X) * torch.linalg.matrix_norm(Y.T @ Y)))
    M = {n: {m: cka(Tc[n], Tc[m]) for m in names} for n in names}
    results["cka"] = M
    print("\nCKA matrix:")
    print("        " + "".join(f"{n[:7]:>8s}" for n in names))
    for n in names:
        print(f"{n[:7]:>7s} " + "".join(f"{M[n][m]:8.2f}" for m in names))

    # unique variance: predict top-32 PCs of T_i from PCA-256 of the others
    uniq = {}
    for n in names:
        tgt = Tc[n]
        tgt = tgt - tgt.mean(0, keepdim=True)
        U, S, V = torch.pca_lowrank(tgt, q=48, center=False, niter=4)
        Y = tgt @ V[:, :32]
        oth = torch.cat([Tc[m] for m in names if m != n], 1)
        oth = oth - oth.mean(0, keepdim=True)
        Uo, So, Vo = torch.pca_lowrank(oth, q=272, center=False, niter=4)
        Xo = oth @ Vo[:, :256]
        Xo = torch.cat([Xo, torch.ones(len(Xo), 1, device=dev)], 1)
        lam = 1e-2 * Xo.shape[0]
        A = Xo.T @ Xo + lam * torch.eye(Xo.shape[1], device=dev)
        Wt = torch.linalg.solve(A, Xo.T @ Y)
        res = Y - Xo @ Wt
        r2 = float(1 - res.var(0).sum() / Y.var(0).sum())
        uniq[n] = dict(r2_from_others=r2, unique_frac=1 - r2)
        print(f"unique variance  {n:20s} R2(from others)={r2:.3f}  unique={1-r2:.3f}", flush=True)
    results["unique"] = uniq
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
