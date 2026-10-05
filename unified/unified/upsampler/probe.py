"""Linear-probe evaluation of upsampled features (docs/UPSAMPLER_PLAN.md §1D).

A fixed bank of 96^3 patches (class-balanced centres, shared by every encoder and
every method) is cut once; each encoder's coarse features for it are cached. For
each upsampling method one per-voxel linear classifier is fit on the train bank and
scored on the val bank at full resolution.

Every method here is linear in the features for fixed weights, so the classifier
is applied BEFORE upsampling: ``up(W F + b) == W up(F) + b``. That upsamples 118
logit channels instead of 1024 feature channels and makes stride-1 output cheap.
The unit test ``t_linear`` in ``scripts/test_upsampler.py`` checks the identity.

Optimizer note (docs/PROBES.md §4): AdamW with lr * wd near 1 erases the weights.
The defaults here give lr * wd = 1e-7. Features are divided by one per-encoder scale
and the bias starts at the log class prior (see ``fit_and_eval``).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import VolumeStore, class_balanced_starts, load_patch
from .encoders import FrozenEncoder
from .geometry import default_centers, interp_to
from .module import aggregate

BANK_ROOT = Path("/home/lukas/data/cache/upsampler/banks")
REPO = Path(__file__).resolve().parents[2]


# -------------------------------------------------------------------- banks
def bank_name(split, n_vol, n_patch, patch):
    return f"{split}_{n_vol}v{n_patch}p_{patch}"


def probe_volumes(store: VolumeStore, split: str, n_vol: int, seed: int = 0):
    """The volumes a bank draws from (also excluded from label-supervised training)."""
    rng = np.random.default_rng(seed)
    sids = store.sids(split)
    if n_vol < len(sids):
        sids = sorted(rng.choice(sids, n_vol, replace=False).tolist())
    return sids, rng


def patch_plan(store: VolumeStore, split: str, n_vol: int, n_patch: int, patch: int,
               seed: int = 0):
    sids, rng = probe_volumes(store, split, n_vol, seed)
    plan = []
    for sid in sids:
        for st in class_balanced_starts(store.label(sid), n_patch, (patch,) * 3, rng):
            plan.append((sid, [int(v) for v in st]))
    return plan


def build_ct_bank(store, split, n_vol, n_patch, patch=96, seed=0, root=BANK_ROOT):
    """CT (int16) and labels (uint8) for a fixed patch plan. Encoder-independent."""
    path = Path(root) / f"{bank_name(split, n_vol, n_patch, patch)}_ct.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    plan = patch_plan(store, split, n_vol, n_patch, patch, seed)
    cts, labs = [], []
    for sid, st in plan:
        img, lab = load_patch(store, sid, st, (patch,) * 3)
        cts.append(img.round().clamp(-32768, 32767).short()); labs.append(lab.byte())
    bank = {"plan": plan, "ct": torch.stack(cts), "lab": torch.stack(labs), "patch": patch}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, path)
    return bank


def build_feat_bank(store, enc: FrozenEncoder, ct_bank, name, batch=4, root=BANK_ROOT,
                    tag=""):
    """Coarse RAS features of every bank patch, normalized per source volume."""
    path = Path(root) / f"{name}_{enc.stem}{tag}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    plan = ct_bank["plan"]
    feats, centers = [], None
    for i in range(0, len(plan), batch):
        xs = []
        for j in range(i, min(i + batch, len(plan))):
            p = store.norm(enc.stem, enc.intensity, plan[j][0])
            xs.append(p.apply(ct_bank["ct"][j].float()[None]))
        f, cs = enc(torch.stack(xs))
        feats.append(f.half().cpu())
        centers = [c.cpu() for c in cs]
    bank = {"feats": torch.cat(feats), "centers": centers, "stem": enc.stem}
    torch.save(bank, path)
    return bank


# ------------------------------------------------------------------ methods
def full_res(logits, fc, out_shape):
    return interp_to(logits, fc, default_centers(out_shape, out_shape))


class TrilinearMethod:
    def prepare(self, bank, device):
        return None

    def __call__(self, cache, idx, ct, logits, fc):
        return full_res(logits, fc, tuple(ct.shape[2:]))


class WeightsMethod:
    """Any module with ``weights(ct, feats, fc, out_shape)``. The module is frozen
    while the probe is fit, so each patch's weights are computed once (fp16, pinned
    CPU memory) and reused every epoch. Logits are aggregated at ``out_stride``,
    then trilinearly interpolated to full resolution if the stride is > 1."""

    def __init__(self, up, out_stride: int = 2, batch: int = 4):
        self.up, self.out_stride, self.batch = up, out_stride, batch
        self.tab = None

    @torch.no_grad()
    def prepare(self, bank, device):
        fine = tuple(bank["ct"].shape[1:])
        out_shape = tuple(n // self.out_stride for n in fine)
        ws = []
        for i in range(0, bank["ct"].shape[0], self.batch):
            ct = bank["ct"][i:i + self.batch].float()[:, None].to(device)
            f = bank["feats"][i:i + self.batch].float().to(device)
            w, tab = self.up.weights(ct, f, bank["centers"], out_shape=out_shape)
            ws.append(w.half().cpu())
        self.tab = tab
        return torch.cat(ws).pin_memory()

    def __call__(self, cache, idx, ct, logits, fc):
        fine = tuple(ct.shape[2:])
        w = cache[idx].to(logits.device, non_blocking=True).float()
        y = aggregate(w, logits, self.tab.to(logits.device))
        if self.out_stride > 1:
            out_shape = tuple(y.shape[2:])
            y = full_res(y, default_centers(out_shape, fine), fine)
        return y


# -------------------------------------------------------------------- probe
def dice_ce_loss(logits, lab, n_classes):
    """Cross-entropy + soft Dice over the classes present in the batch. Written with
    gather / scatter instead of a one-hot tensor: at 96^3 x 118 classes x 4 patches
    the one-hot alone is 1.7 GB, and this loss dominated the probe step."""
    logp = logits.float().log_softmax(1)
    ce = F.nll_loss(logp, lab)
    p = logp.exp()
    flat = lab.reshape(-1)
    inter = torch.zeros(n_classes, device=p.device).scatter_add(
        0, flat, p.gather(1, lab[:, None]).reshape(-1))
    cnt = torch.bincount(flat, minlength=n_classes).float()
    denom = p.sum((0, 2, 3, 4)) + cnt
    present = cnt > 0
    dice = (2 * inter + 1e-5) / (denom + 1e-5)
    return ce + (1 - dice[present].mean())


@torch.no_grad()
def confusion(pred, lab, n_classes):
    """(tp, fp, fn) per class for one batch."""
    tp = torch.bincount(lab[pred == lab].flatten(), minlength=n_classes)
    pc = torch.bincount(pred.flatten(), minlength=n_classes)
    lc = torch.bincount(lab.flatten(), minlength=n_classes)
    return tp, pc - tp, lc - tp


NSD_TOL = (1, 2)          # voxels at 1.5 mm: 1.5 mm and 3 mm


@torch.no_grad()
def surface_counts(pred, lab, n_classes, tols=NSD_TOL):
    """Per-class counts for normalized surface Dice, summed over a batch.

    Boundary = mask minus its 3x3x3 erosion; "within tol" = inside a (2 tol + 1)^3
    dilation of the other boundary (Chebyshev distance, a close approximation at
    1-2 voxels). Returns {tol: (|dP|, |dG|, |dP near dG|, |dG near dP|)}, each (K,)."""
    out = {t: [torch.zeros(n_classes, device=pred.device) for _ in range(4)] for t in tols}
    for b in range(pred.shape[0]):
        P = F.one_hot(pred[b], n_classes).movedim(-1, 0)[None].half()
        G = F.one_hot(lab[b], n_classes).movedim(-1, 0)[None].half()
        dP = P - (-F.max_pool3d(-P, 3, 1, 1))
        dG = G - (-F.max_pool3d(-G, 3, 1, 1))
        for t in tols:
            nG = F.max_pool3d(dG, 2 * t + 1, 1, t)
            nP = F.max_pool3d(dP, 2 * t + 1, 1, t)
            c = out[t]
            f32 = dict(dtype=torch.float32)   # fp16 sums overflow (inf) past 65504 voxels
            c[0] += dP.sum((0, 2, 3, 4), **f32); c[1] += dG.sum((0, 2, 3, 4), **f32)
            c[2] += (dP * nG).sum((0, 2, 3, 4), **f32); c[3] += (dG * nP).sum((0, 2, 3, 4), **f32)
    return out


def thin_classes(threshold_mm=8.0):
    t = json.loads((REPO / "results/upsampler/class_thickness.json").read_text())
    return {k for k, v in t.items() if v["thick_mm"] < threshold_mm}, set(t)


def summarize(tp, fp, fn, class_names, thin, known, surf=None):
    present = [c for c in range(1, len(class_names) + 1) if int(tp[c] + fn[c]) > 0]
    dice = {class_names[c - 1]: float(2 * tp[c] / (2 * tp[c] + fp[c] + fn[c])) for c in present}

    def split(d, prefix):
        thin_v = [v for k, v in d.items() if k in thin]
        thick_v = [v for k, v in d.items() if k in known and k not in thin]
        return {f"{prefix}_mean": float(np.mean(list(d.values()))),
                f"{prefix}_thin": float(np.mean(thin_v)) if thin_v else math.nan,
                f"{prefix}_thick": float(np.mean(thick_v)) if thick_v else math.nan}

    r = {**split(dice, "dice"), "n_classes": len(dice),
         "n_thin": sum(1 for k in dice if k in thin), "per_class": dice}
    for t, (bp, bg, pn, gn) in (surf or {}).items():
        nsd = {class_names[c - 1]: float((pn[c] + gn[c]) / (bp[c] + bg[c]).clamp_min(1))
               for c in present}
        r.update(split(nsd, f"nsd{t}"))
        r[f"per_class_nsd{t}"] = nsd
    return r


def fit_and_eval(method, tr: dict, va: dict, class_names: List[str], device,
                 epochs=20, batch=4, lr=1e-3, wd=1e-4, seed=0, log=print):
    """tr / va: {"ct", "lab", "feats", "centers"} (CPU). Returns the val summary."""
    torch.manual_seed(seed)
    K = len(class_names) + 1
    C = tr["feats"].shape[1]
    # Probe protocol (chosen by a 2-epoch A/B on 3DINO and SAM-Med3D):
    #  * divide the features by ONE per-encoder scale from the train bank. Encoders
    #    differ ~7x in scale (SAM-Med3D std 0.14, 3DINO 1.0) and Adam's step size
    #    does not adapt to it. Per-channel standardization inflates near-constant
    #    channels (3DINO has channels at std 0.007) and cost 0.019 Dice.
    #  * do NOT centre: with Adam at lr 1e-3 the uncentred channel means act as a
    #    fast bias. Instead initialize the bias to the log class prior of the train
    #    bank, which is what the probe would otherwise spend thousands of steps on.
    # Both maps are affine, so the probe stays linear and commutes with every upsampler.
    sd = float(tr["feats"].float().std())
    probe = nn.Conv3d(C, K, 1).to(device)
    cnt = torch.bincount(tr["lab"].flatten().long(), minlength=K).float()
    with torch.no_grad():
        probe.bias.copy_(torch.log((cnt + 1) / (cnt + 1).sum()).to(device))

    def std_probe(f):
        return probe(f / sd)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    N = tr["feats"].shape[0]
    steps = epochs * math.ceil(N / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    cache_tr = method.prepare(tr, device)
    cache_va = method.prepare(va, device)
    if cache_tr is not None:
        log(f"    cached weights for {N + va['feats'].shape[0]} patches ({time.time() - t0:.0f}s)")
    feats_tr = tr["feats"].to(device)                     # fp16, small
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        order = torch.randperm(N, generator=g)
        run = []
        for i in range(0, N, batch):
            idx = order[i:i + batch]
            ct = tr["ct"][idx].float()[:, None].to(device)
            lab = tr["lab"][idx].long().to(device)
            y = method(cache_tr, idx, ct, std_probe(feats_tr[idx.to(device)].float()), tr["centers"])
            loss = dice_ce_loss(y, lab, K)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            run.append(loss.item())
        if ep % 5 == 4 or ep == epochs - 1:
            log(f"    epoch {ep + 1}/{epochs} loss {np.mean(run):.4f} ({time.time() - t0:.0f}s)")
    tp = torch.zeros(K, dtype=torch.long, device=device)
    fp, fn = tp.clone(), tp.clone()
    surf = {}
    with torch.no_grad():
        for i in range(0, va["feats"].shape[0], batch):
            idx = torch.arange(i, min(i + batch, va["feats"].shape[0]))
            ct = va["ct"][idx].float()[:, None].to(device)
            lab = va["lab"][idx].long().to(device)
            f = va["feats"][idx].float().to(device)
            pred = method(cache_va, idx, ct, std_probe(f), va["centers"]).argmax(1)
            a, b, c = confusion(pred, lab, K)
            tp += a; fp += b; fn += c
            for t, cnt in surface_counts(pred, lab, K).items():
                for acc, v in zip(surf.setdefault(t, [torch.zeros(K, device=device) for _ in range(4)]), cnt):
                    acc += v
    thin, known = thin_classes()
    surf = {t: [v.cpu() for v in vs] for t, vs in surf.items()}
    return summarize(tp.cpu(), fp.cpu(), fn.cpu(), class_names, thin, known, surf)
