"""Label-free training of the 3D guided upsampler on one frozen encoder (plan §1C).

Objectives (unified/upsampler/objectives.py): ``crop`` and ``down`` are label-free;
``label`` is the plan's fallback, trained through a throwaway linear head.

Every step reports the same loss for trilinear interpolation on the same batch, so
the log always shows whether the module beats the baseline it must replace.
Validation uses fixed crops and keeps best.pt by the validation score: cosine
similarity for the label-free objectives (val volumes, no labels), minus the loss
for ``label`` (held-out TRAIN volumes, so the probe's val split is never seen).

GPU sharing: ``--mem-frac`` caps this process's share of the GPU (default 0.25 of
80 GB), so it cannot starve the training jobs already on the card.

Usage:
    python -m scripts.upsampler_train --encoder dino3d_layerwise --objective crop \\
        --out runs/upsampler/dino3d_crop --steps 6000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from unified.upsampler.data import VolumeStore  # noqa: E402
from unified.upsampler.encoders import FrozenEncoder  # noqa: E402
from unified.upsampler.geometry import interp_to  # noqa: E402
from unified.upsampler.module import GuidedUpsampler3D  # noqa: E402
from unified.upsampler.objectives import (SAMPLERS, FeatureStats, feature_loss,  # noqa: E402
                                          label_logits)
from unified.upsampler.probe import dice_ce_loss, probe_volumes  # noqa: E402

DEFAULT_STORE = "/home/lukas/data/cache/upsampler/vol15"
N_CLASSES = 118
PROBE_TRAIN_VOLS = 50        # must match scripts/upsampler_probe.py --n-train-vol


class ScaledHead(torch.nn.Module):
    """The throwaway linear head applied to features divided by a fixed global std."""

    def __init__(self, conv, sd: float):
        super().__init__()
        self.conv = conv
        self.register_buffer("sd", torch.tensor(float(sd)))

    def forward(self, x):
        return self.conv(x / self.sd)


def module_kwargs(a):
    return dict(dim=a.dim, width=a.width, depth=a.depth, radius=a.radius,
                feature_keys=a.feature_keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dino3d_layerwise")
    ap.add_argument("--objective", default="crop", choices=sorted(SAMPLERS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--w-mse", type=float, default=1.0)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--feature-keys", action="store_true")
    ap.add_argument("--norm-feats", action="store_true",
                    help="label objective: head on features / global std, log-prior bias")
    ap.add_argument("--out-stride", type=int, default=2,
                    help="label objective: output stride before the final trilinear step")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-frac", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    log = open(out / "log.jsonl", "a")
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    store = VolumeStore(a.store)
    train_ids, val_ids = store.sids("train"), store.sids("val")[:12]
    if a.objective == "label":
        # labels are used: keep the probe bank's volumes and the val split out entirely
        held = set(probe_volumes(store, "train", PROBE_TRAIN_VOLS)[0])
        rest = [s_ for s_ in train_ids if s_ not in held]
        train_ids, val_ids = rest[:-12], rest[-12:]
    enc = FrozenEncoder(a.encoder, device=dev)
    sample = SAMPLERS[a.objective]
    kw = dict(patch=a.patch)

    def draw(r, ids):
        return sample(store, [ids[int(i)] for i in r.integers(0, len(ids), a.batch)], enc, r, **kw)

    C = enc.feature_dim()
    stats = FeatureStats(C).to(dev)
    head = None
    if a.objective == "label":
        head = torch.nn.Conv3d(C, N_CLASSES, 1).to(dev)
        if a.norm_feats:
            # As the probe (unified/upsampler/probe.py:fit_and_eval): features divided by
            # one global std, bias at the log class prior. Without it a small-scale encoder
            # (SAM-Med3D, std 0.14) leaves the throwaway head nearly untrained, and the
            # upsampler gets almost no signal.
            draws = [draw(rng, train_ids) for _ in range(8)]
            sd = float(torch.cat([s_.feats.flatten() for s_ in draws]).std())
            cnt = torch.bincount(torch.cat([s_.target.flatten() for s_ in draws]).cpu(),
                                 minlength=N_CLASSES).double()
            with torch.no_grad():
                head.bias.copy_(torch.log((cnt + 1) / (cnt + 1).sum()).float().to(dev))
            head = ScaledHead(head, sd)
            print(f"label head on features / {sd:.4f}, bias at the log class prior", flush=True)
    else:
        stats.fit([draw(rng, train_ids).feats for _ in range(8)])   # from training crops

    # fixed validation batches (same crops, same sub-crops every time)
    vr = np.random.default_rng(10_000 + a.seed)
    val = [draw(vr, val_ids) for _ in range(a.val_batches)]

    model = GuidedUpsampler3D(**module_kwargs(a)).to(dev)
    params = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps,
                                                pct_start=0.05, anneal_strategy="cos")
    start, best = 0, -float("inf")
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); stats.load_state_dict(ck["stats"])
        if head is not None:
            head.load_state_dict(ck["head"])
        start, best = ck["step"], ck["best"]
        rng = np.random.default_rng(a.seed + start)
        print(f"resumed at step {start}", flush=True)

    # score: higher is better. Feature objectives: cosine to the target features.
    # Label objective: minus the Dice+CE loss. The trilinear score uses the same
    # head, so it is only a reference, not a tuned baseline.
    metric = "neg-loss" if a.objective == "label" else "cos"

    def losses(s):
        if a.objective == "label":
            loss = dice_ce_loss(label_logits(model, head, s, a.out_stride), s.target, N_CLASSES)
            with torch.no_grad():
                l_t = dice_ce_loss(label_logits(model, head, s, trilinear=True), s.target, N_CLASSES)
            return loss, -loss.detach(), -l_t
        p = model(s.guide_hu, s.feats, s.feat_centers, out_centers=s.out_centers)
        loss, cos, _ = feature_loss(p, s.target, stats, a.w_mse)
        with torch.no_grad():
            _, cos_t, _ = feature_loss(interp_to(s.feats, s.feat_centers, s.out_centers),
                                       s.target, stats, a.w_mse)
        return loss, cos, cos_t

    def evaluate():
        model.eval()
        sc, sc_t, ls = [], [], []
        with torch.no_grad():
            for s in val:
                l, c, ct = losses(s)
                sc.append(float(c)); sc_t.append(float(ct)); ls.append(float(l))
        model.train()
        return float(np.mean(sc)), float(np.mean(sc_t)), float(np.mean(ls))

    def save(name, step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "stats": stats.state_dict(), "step": step,
                    "best": best, "kwargs": module_kwargs(a), "encoder": a.encoder,
                    "objective": a.objective,
                    "head": head.state_dict() if head is not None else None}, out / f"{name}.tmp")
        (out / f"{name}.tmp").replace(out / name)

    if start == 0:
        c_up, c_tri, l_up = evaluate()
        print(f"step 0  val {metric} {c_up:.4f} (trilinear {c_tri:.4f})", flush=True)
        log.write(json.dumps({"step": 0, "val_score": c_up, "val_score_tri": c_tri,
                              "val_loss": l_up}) + "\n"); log.flush()

    model.train()
    t0, run = time.time(), []
    for step in range(start + 1, a.steps + 1):
        s = draw(rng, train_ids)
        loss, sc, sc_t = losses(s)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        run.append((loss.item(), float(sc), float(sc_t)))
        if step % 50 == 0:
            r = np.mean(run, 0); run = []
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0
            print(f"step {step:5d}  loss {r[0]:.4f}  {metric} {r[1]:.4f} (trilinear {r[2]:.4f})  "
                  f"tau {model.log_tau.exp().item():.2f}  {(time.time() - t0) / (step - start):.2f}s/it "
                  f"peak {mem:.1f}G", flush=True)
            log.write(json.dumps({"step": step, "loss": r[0], "score": r[1], "score_tri": r[2]}) + "\n")
            log.flush()
        if step % a.val_every == 0 or step == a.steps:
            c_up, c_tri, l_up = evaluate()
            improved = c_up > best
            best = max(best, c_up)
            print(f"step {step}  val {metric} {c_up:.4f} (trilinear {c_tri:.4f})"
                  f"{'  *best' if improved else ''}", flush=True)
            log.write(json.dumps({"step": step, "val_score": c_up, "val_score_tri": c_tri,
                                  "val_loss": l_up}) + "\n"); log.flush()
            save("last.pt", step)
            if improved:
                save("best.pt", step)
    print(f"total time {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
