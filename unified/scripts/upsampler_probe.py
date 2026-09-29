"""Stage-1/2 evaluation: linear probe on upsampled features, per method (plan §1D, §2).

Methods (``--methods``):
    trilinear                 interpolate the coarse logits
    bilateral[:s:r]           classic joint bilateral upsampling (spatial sigma s cells,
                              range sigma r on the [-1000, 1000] HU window mapped to [0, 1])
    init[@stride]             the upsampler before training (a fixed Gaussian kernel)
    up:<ckpt>[@stride]        a trained upsampler
  stride = output stride of the neighbourhood methods before the final trilinear step
  (default 2, as in the plan; @1 is the ablation)

Results go to one JSON per run, one entry per method, written as each finishes, so
a rerun skips what is done.

Usage:
    python -m scripts.upsampler_probe --encoder dino3d_layerwise \\
        --methods trilinear bilateral init up:runs/upsampler/dino3d_crop/best.pt \\
        --out results/upsampler/stage1_dino3d.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from unified.data.totalsegmentator import load_classes  # noqa: E402
from unified.upsampler.data import VolumeStore  # noqa: E402
from unified.upsampler.encoders import FrozenEncoder  # noqa: E402
from unified.upsampler.module import BilateralUpsampler, GuidedUpsampler3D  # noqa: E402
from unified.upsampler.probe import (TrilinearMethod, WeightsMethod, bank_name,  # noqa: E402
                                     build_ct_bank, build_feat_bank, fit_and_eval)

DEFAULT_STORE = "/home/lukas/data/cache/upsampler/vol15"


def make_method(spec: str, device):
    name, _, stride = spec.partition("@")
    stride = int(stride) if stride else 2
    if name == "trilinear":
        return TrilinearMethod()
    if name.startswith("bilateral"):
        parts = name.split(":")[1:]
        s, r = (float(parts[0]), float(parts[1])) if parts else (0.5, 0.1)
        return WeightsMethod(BilateralUpsampler(spatial_sigma=s, range_sigma=r).to(device), stride)
    if name == "init":
        return WeightsMethod(GuidedUpsampler3D().to(device).eval(), stride)
    if name.startswith("up:"):
        ck = torch.load(name[3:], map_location=device, weights_only=False)
        m = GuidedUpsampler3D(**ck["kwargs"]).to(device).eval()
        m.load_state_dict(ck["model"])
        return WeightsMethod(m, stride)
    raise ValueError(f"unknown method {spec!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dino3d_layerwise")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--n-train-vol", type=int, default=50)
    ap.add_argument("--n-val-vol", type=int, default=20)
    ap.add_argument("--n-patch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-frac", type=float, default=0.3)
    ap.add_argument("--random-encoder", action="store_true",
                    help="random-init twin of the encoder (control)")
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    res.setdefault("_setup", vars(a))

    store = VolumeStore(a.store)
    t0 = time.time()
    tr_ct = build_ct_bank(store, "train", a.n_train_vol, a.n_patch, a.patch, seed=a.seed)
    va_ct = build_ct_bank(store, "val", a.n_val_vol, a.n_patch, a.patch, seed=a.seed)
    tag = "_random" if a.random_encoder else ""
    enc = FrozenEncoder(a.encoder, device=dev, pretrained=not a.random_encoder)
    tr_f = build_feat_bank(store, enc, tr_ct, bank_name("train", a.n_train_vol, a.n_patch, a.patch), tag=tag)
    va_f = build_feat_bank(store, enc, va_ct, bank_name("val", a.n_val_vol, a.n_patch, a.patch), tag=tag)
    del enc; torch.cuda.empty_cache()
    print(f"banks: train {len(tr_ct['plan'])} patches, val {len(va_ct['plan'])} patches, "
          f"features {tuple(tr_f['feats'].shape[1:])}  ({time.time() - t0:.0f}s)", flush=True)
    tr = {"ct": tr_ct["ct"], "lab": tr_ct["lab"], "feats": tr_f["feats"], "centers": tr_f["centers"]}
    va = {"ct": va_ct["ct"], "lab": va_ct["lab"], "feats": va_f["feats"], "centers": va_f["centers"]}
    names = load_classes(REPO / "unified/data/classes.txt")

    for spec in a.methods:
        if spec in res:
            print(f"{spec}: done already", flush=True)
            continue
        print(f"{spec}:", flush=True)
        t1 = time.time()
        r = fit_and_eval(make_method(spec, dev), tr, va, names, dev, epochs=a.epochs,
                         batch=a.batch, lr=a.lr, wd=a.wd, seed=a.seed,
                         log=lambda s: print(s, flush=True))
        r["seconds"] = round(time.time() - t1)
        res[spec] = r
        # merge with whatever other runs wrote to the same file meanwhile
        disk = json.loads(out.read_text()) if out.exists() else {}
        disk.update({k: v for k, v in res.items() if k not in disk or k == spec})
        tmp = out.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(disk, indent=1))
        os.replace(tmp, out)
        res = disk
        print(f"  {spec}: dice {r['dice_mean']:.4f}  thin {r['dice_thin']:.4f}  "
              f"thick {r['dice_thick']:.4f}  ({r['n_classes']} classes, {r['n_thin']} thin)",
              flush=True)
    base = res.get("trilinear")
    if base:
        print("\nvs trilinear:", flush=True)
        for k, v in res.items():
            if k.startswith("_") or k == "trilinear":
                continue
            print(f"  {k:60s} dice {v['dice_mean'] - base['dice_mean']:+.4f}  "
                  f"thin {v['dice_thin'] - base['dice_thin']:+.4f}  "
                  f"thick {v['dice_thick'] - base['dice_thick']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
