"""Linear-probe CT-RATE 18-way abnormality classification across frozen encoders.

The classification counterpart to the segmentation benchmark: same "swap the
encoder, keep everything else identical" premise. For each frozen backbone we
pool its NATIVE deepest feature map (GAP), fit a single linear layer (BCE), and
report per-class AUROC/AP on the CT-RATE validation set.

Fairness knobs come from each model's own config (like the seg track):
  * orientation (model.preprocessing.axcodes) — RAS cache reoriented on the fly
  * intensity window (model.preprocessing.intensity) — applied on the fly
The HU cache (unified.data.ctrate) is model-independent, so ALL backbones share
one cache.

  python -m scripts.ctrate_linprobe --cache-root /home/lukas/data/CTRATE \
      --models ctfm vista3d voco_b suprem_unet --roi 128 128 128
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

from unified.utils import load_config
from unified.data.transforms import _resolved_preprocessing
from unified.data.ctrate import (
    _CSV_PATHS, load_label_table, CTRateCachedDataset, PATHOLOGIES,
)
import unified.models.backbones  # noqa: F401 (registers adapters)
from unified.models import build_backbone


# --- pooling & metrics --------------------------------------------------------

@torch.no_grad()
def native_embedding(backbone, x: torch.Tensor) -> torch.Tensor:
    """GAP the encoder's deepest NATIVE feature map -> (B, C).

    Uses encoder_forward (pre-adapter native features) when available, picking
    the tensor with the most channels (the deepest semantic map; this skips any
    raw-input tensor some adapters pass through, e.g. ct-clip / dino3d). Falls
    back to the contract bottleneck (forward_features[-1]) for adapters whose
    encoder_forward exposes no real feature map (e.g. dino3d vit_adapter).
    """
    deepest = None
    if hasattr(backbone, "encoder_forward"):
        native = backbone.encoder_forward(x)
        cand = [t for t in native if torch.is_tensor(t) and t.dim() == 5]
        if cand:
            deepest = max(cand, key=lambda t: t.shape[1])
    if deepest is None or deepest.shape[1] <= 1:
        deepest = backbone.forward_features(x)[-1]
    return deepest.mean(dim=(2, 3, 4))


def auroc(y_true, y_score):
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score)); ranks[order] = np.arange(1, len(y_score) + 1)
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(y_true, y_score):
    if y_true.sum() == 0:
        return np.nan
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    recall = tp / y.sum()
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return ap


# --- feature extraction -------------------------------------------------------

def extract_split(name, backbone, cfg, split, cache_root, roi, dev, emb_dir, bs):
    emb_path = emb_dir / f"{name}_{split}_{roi[0]}.npz"
    if emb_path.exists():
        d = np.load(emb_path)
        print(f"  [{split}] embeddings cached: X={d['X'].shape}", flush=True)
        return d["X"], d["Y"]

    axcodes, intensity = _resolved_preprocessing(cfg)
    labels = load_label_table(hf_hub_download(
        "ibrahimhamamci/CT-RATE", repo_type="dataset",
        filename=_CSV_PATHS[split]["labels"]))
    ds = CTRateCachedDataset(sorted(labels), Path(cache_root) / split, labels,
                             roi=tuple(roi), intensity=intensity, axcodes=axcodes)
    if len(ds) == 0:
        print(f"  [{split}] no cached volumes found", flush=True)
        return None, None
    dl = DataLoader(ds, batch_size=bs, num_workers=2, shuffle=False)
    Xs, Ys = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (vol, lab) in enumerate(dl):
            emb = native_embedding(backbone, vol.to(dev))
            Xs.append(emb.float().cpu().numpy())
            Ys.append(lab.numpy())
            if (i + 1) % 50 == 0:
                print(f"  [{split}] {(i+1)*bs}/{len(ds)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.savez(emb_path, X=X, Y=Y)
    print(f"  [{split}] extracted X={X.shape} -> {emb_path.name}", flush=True)
    return X, Y


def fit_probe(Xtr, Ytr, Xva, Yva, dev, epochs, lr, wd, seed):
    torch.manual_seed(seed)
    xt = torch.tensor(Xtr, device=dev, dtype=torch.float32)
    yt = torch.tensor(Ytr, device=dev, dtype=torch.float32)
    mu, sd = xt.mean(0, keepdim=True), xt.std(0, keepdim=True) + 1e-6
    xt = (xt - mu) / sd
    lin = torch.nn.Linear(Xtr.shape[1], 18).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(lin(xt), yt).backward()
        opt.step()
    with torch.no_grad():
        xv = (torch.tensor(Xva, device=dev, dtype=torch.float32) - mu) / sd
        scores = torch.sigmoid(lin(xv)).cpu().numpy()
    aucs = np.array([auroc(Yva[:, k], scores[:, k]) for k in range(18)])
    aps = np.array([average_precision(Yva[:, k], scores[:, k]) for k in range(18)])
    return aucs, aps


# --- driver -------------------------------------------------------------------

def run_model(name, args, dev):
    print(f"\n=== {name} ===", flush=True)
    cfg = load_config(REPO / "configs" / "models" / f"{name}.yaml")
    mcfg = cfg["model"]
    backbone = build_backbone(mcfg["name"], weights=mcfg.get("weights"),
                              **mcfg.get("kwargs", {})).to(dev).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    emb_dir = Path(args.emb_dir)
    try:
        Xtr, Ytr = extract_split(name, backbone, cfg, "train", args.cache_root,
                                 args.roi, dev, emb_dir, args.batch_size)
        Xva, Yva = extract_split(name, backbone, cfg, "valid", args.cache_root,
                                 args.roi, dev, emb_dir, args.batch_size)
    finally:
        del backbone
        if dev == "cuda":
            torch.cuda.empty_cache()
    if Xtr is None or Xva is None:
        return None
    aucs, aps = fit_probe(Xtr, Ytr, Xva, Yva, dev, args.probe_epochs,
                          args.probe_lr, args.probe_wd, args.seed)
    macro_auc = float(np.nanmean(aucs))
    macro_ap = float(np.nanmean(aps))
    print(f"  macro-AUROC={macro_auc:.4f}  macro-AP={macro_ap:.4f}  "
          f"(embed dim={Xtr.shape[1]}, n_train={len(Xtr)}, n_val={len(Xva)})",
          flush=True)
    return {"model": name, "macro_auroc": macro_auc, "macro_ap": macro_ap,
            "embed_dim": int(Xtr.shape[1]), "n_train": int(len(Xtr)),
            "n_val": int(len(Xva)),
            "per_class_auroc": {PATHOLOGIES[k]: (None if np.isnan(aucs[k])
                                                 else float(aucs[k]))
                                for k in range(18)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="/home/lukas/data/CTRATE")
    ap.add_argument("--models", nargs="+",
                    default=["ctfm", "vista3d", "voco_b", "suprem_unet"])
    ap.add_argument("--roi", nargs=3, type=int, default=[128, 128, 128])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--emb-dir", default="/home/lukas/data/CTRATE/_emb")
    ap.add_argument("--probe-epochs", type=int, default=500)
    ap.add_argument("--probe-lr", type=float, default=1e-2)
    ap.add_argument("--probe-wd", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/home/lukas/data/CTRATE/_emb/linprobe_results.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  roi={tuple(args.roi)}  models={args.models}", flush=True)
    results = []
    for name in args.models:
        try:
            r = run_model(name, args, dev)
            if r:
                results.append(r)
        except Exception as e:  # noqa: BLE001 - one bad backbone shouldn't kill the sweep
            import traceback
            print(f"  {name} FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    results.sort(key=lambda r: r["macro_auroc"], reverse=True)
    print("\n==== CT-RATE linear-probe (macro over 18 abnormalities) ====")
    print(f"{'model':16s} {'macro-AUROC':>12s} {'macro-AP':>10s} {'dim':>6s}")
    for r in results:
        print(f"{r['model']:16s} {r['macro_auroc']:12.4f} {r['macro_ap']:10.4f} "
              f"{r['embed_dim']:6d}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
