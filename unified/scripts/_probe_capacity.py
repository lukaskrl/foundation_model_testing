"""Feasibility probe: can K frozen encoders coexist and run at benchmark patch size?

Per encoder: param count, resident VRAM, peak VRAM for a no_grad forward at
(2,1,96,96,96), forward latency, and native feature shapes. Then the sum, which
is what a fusion backbone / multi-teacher distillation step would have to hold.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
from unified.utils import load_config
import unified.models.backbones  # noqa: F401
from unified.models import build_backbone

CONFIGS = ["ctfm", "vista3d", "voco_b", "voco_h", "suprem_unet", "suprem_segresnet",
           "suprem_swinunetr", "biomedparse", "ctclip", "merlin", "sam_med3d", "dino3d"]


def probe(name, patch, batch, device, keep):
    cfg = load_config(str(REPO / f"configs/models/{name}.yaml"))
    m = cfg["model"]
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    t0 = time.time()
    bb = build_backbone(m["name"], weights=m.get("weights"), **m.get("kwargs", {}))
    bb = bb.to(device).eval()
    load_s = time.time() - t0
    resident = (torch.cuda.memory_allocated(device) - base) / 2**30

    enc_p = sum(p.numel() for n_, p in bb.named_parameters() if not n_.startswith("adapter"))
    ad_p = sum(p.numel() for p in bb.adapter.parameters()) if hasattr(bb, "adapter") else 0

    x = torch.randn(batch, 1, patch, patch, patch, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(2):
            native = bb.encoder_forward(x) if hasattr(bb, "encoder_forward") else None
            feats = bb.forward_features(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            feats = bb.forward_features(x)
        torch.cuda.synchronize()
        lat = (time.time() - t0) / 3
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    act = peak - resident - base / 2**30

    nat = []
    if native is not None:
        nat = [tuple(t.shape) for t in native if torch.is_tensor(t) and t.dim() == 5]
    out = dict(name=name, encoder_params=enc_p, adapter_params=ad_p,
               resident_gb=round(resident, 3), peak_gb=round(peak, 3),
               activation_gb=round(act, 3), latency_s=round(lat, 4),
               load_s=round(load_s, 1),
               native_shapes=[list(s) for s in nat],
               contract_shapes=[list(t.shape) for t in feats])
    if not keep:
        del bb, x, feats, native
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--models", nargs="*", default=CONFIGS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    rows = []
    for n in a.models:
        try:
            r = probe(n, a.patch, a.batch, dev, keep=False)
            rows.append(r)
            print(f"{n:18s} enc={r['encoder_params']/1e6:8.1f}M ad={r['adapter_params']/1e6:5.2f}M "
                  f"res={r['resident_gb']:6.2f}G act={r['activation_gb']:6.2f}G "
                  f"peak={r['peak_gb']:6.2f}G lat={r['latency_s']*1000:7.1f}ms", flush=True)
        except Exception as e:
            print(f"{n:18s} FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append(dict(name=n, error=f"{type(e).__name__}: {e}"))
            torch.cuda.empty_cache()
    ok = [r for r in rows if "error" not in r]
    print("-" * 96)
    print(f"K={len(ok)}  sum resident={sum(r['resident_gb'] for r in ok):.2f} GB  "
          f"sum activation={sum(r['activation_gb'] for r in ok):.2f} GB  "
          f"max single activation={max(r['activation_gb'] for r in ok):.2f} GB")
    print(f"sequential fusion forward (resident all + one activation) ~= "
          f"{sum(r['resident_gb'] for r in ok) + max(r['activation_gb'] for r in ok):.2f} GB")
    print(f"sum latency (one fused forward) = {sum(r['latency_s'] for r in ok)*1000:.0f} ms "
          f"vs best single (vista3d) = {[r['latency_s'] for r in ok if r['name']=='vista3d']}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
