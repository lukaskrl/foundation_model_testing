"""Full-volume evaluation of the interface heads, per case (docs/UPSAMPLER_PLAN.md,
"Inside a trained head").

scripts/evaluate.py reports only the per-class mean over cases. The paper needs paired
confidence intervals between heads, so this records, for every case of the split,
per-class Dice and normalized surface Dice at 1 and 2 voxels (1.5 / 3 mm), with the same
model build, sliding window and checkpoint (best.pt) as scripts/evaluate.py. One JSON per
run, rewritten after every case, so a rerun resumes. Summaries and bootstrap intervals:
``python -m scripts.upsampler_heads --test``.

Usage:
    python -m scripts.upsampler_test_eval --run dino3d_i1g_frz_pt_f100 [--split test]
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
import torch.nn.functional as F  # noqa: E402

from unified.data import TotalSegmentatorDataset, build_val_transforms, load_classes  # noqa: E402
from unified.evaluation import Evaluator  # noqa: E402
from unified.models import SegModel, build_backbone, build_head  # noqa: E402
from unified.utils import load_checkpoint, load_config, setup_logging  # noqa: E402

OUT_DIR = REPO / "results/upsampler/test_heads"
NSD_TOL = (1, 2)            # voxels at 1.5 mm


def build_model(cfg, ckpt, device):
    """Exactly as scripts/evaluate.py: frozen runs save only adapter + head, so the
    encoder comes from the run's weights file."""
    m = cfg["model"]
    weights = m.get("weights") if m.get("pretrained", True) else None
    backbone = build_backbone(m["name"], weights=weights, **m.get("kwargs", {}))
    head_kwargs = {k: v for k, v in cfg["head"].items() if k != "name"}
    if hasattr(backbone, "head_kwargs"):
        head_kwargs.update(backbone.head_kwargs())
    head = build_head(cfg["head"].get("name", "unified_seg_head"), **head_kwargs)
    model = SegModel(backbone, head, freeze_backbone=bool(m.get("freeze_backbone", False)))
    model.to(device)
    load_checkpoint(ckpt, model=model, map_location=device, strict=True)
    return model.eval()


@torch.no_grad()
def nsd_per_class(pred, label, classes_idx, tols=NSD_TOL, margin=3):
    """Normalized surface Dice per class on the bounding box of gt ∪ pred.

    Boundary = mask minus its 3x3x3 erosion; "within tol" = inside a (2 tol + 1)^3
    dilation of the other boundary (Chebyshev distance), as in
    unified/upsampler/probe.py:surface_counts. A class missing from the prediction
    scores 0."""
    out = {t: {} for t in tols}
    shape = pred.shape
    for c in classes_idx:
        P, G = pred == c, label == c
        U = P | G
        nz = [torch.nonzero(U.any(dim=tuple(j for j in range(3) if j != a))).flatten()
              for a in range(3)]
        sl = tuple(slice(max(int(n[0]) - margin, 0), min(int(n[-1]) + margin + 1, s))
                   for n, s in zip(nz, shape))
        Pc, Gc = P[sl][None, None].half(), G[sl][None, None].half()
        dP = Pc - (-F.max_pool3d(-Pc, 3, 1, 1))
        dG = Gc - (-F.max_pool3d(-Gc, 3, 1, 1))
        f32 = dict(dtype=torch.float32)   # fp16 sums overflow (inf) past 65504 voxels
        sP, sG = float(dP.sum(**f32)), float(dG.sum(**f32))
        for t in tols:
            if sP == 0:
                out[t][c] = 0.0
                continue
            nG = F.max_pool3d(dG, 2 * t + 1, 1, t)
            nP = F.max_pool3d(dP, 2 * t + 1, 1, t)
            out[t][c] = (float((dP * nG).sum(**f32)) + float((dG * nP).sum(**f32))) / max(sP + sG, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under runs/interface/")
    ap.add_argument("--config", default=None, help="default configs/interface/<run>.yaml")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    cfg = load_config(a.config or str(REPO / f"configs/interface/{a.run}.yaml"))
    setup_logging(None)
    classes = load_classes()
    ids = [l.strip() for l in (REPO / f"unified/data/splits/{a.split}.txt").read_text().splitlines()
           if l.strip()][: a.limit]

    out = OUT_DIR / f"{a.run}_{a.split}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    res.setdefault("_meta", {"run": a.run, "ckpt": a.ckpt, "split": a.split,
                             "nsd_tol_vox": list(NSD_TOL), "spacing_mm": 1.5})
    cases = res.setdefault("cases", {})
    todo = [s for s in ids if s not in cases]
    print(f"{a.run}: {len(ids) - len(todo)} cases done, {len(todo)} to go", flush=True)
    if not todo:
        return

    ds = TotalSegmentatorDataset(cfg["data"]["dataset_root"], todo, classes,
                                 use_source_affine=bool(cfg["data"].get("use_source_affine", False)))
    tf = build_val_transforms(cfg)

    class Composed(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            return tf(ds[i])

    from monai.data import DataLoader
    loader = DataLoader(Composed(), batch_size=1, num_workers=a.workers)

    dev = torch.device("cuda")
    model = build_model(cfg, REPO / "runs/interface" / a.run / a.ckpt, dev)
    ev = Evaluator(cfg, classes, metrics=["dice"])
    t0 = time.time()
    for i, batch in enumerate(loader):
        sid = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
        image = batch["image"].to(dev, non_blocking=True)
        label = batch["label"].to(dev, non_blocking=True)
        with torch.no_grad():            # as Evaluator.evaluate: fp32, no autocast
            pred = ev._infer(model, image)
        dice, gt_present, _ = ev._dice_from_confusion(ev._confusion(pred, label))
        present = [j + 1 for j in torch.nonzero(gt_present).flatten().tolist()]
        plain = lambda t: t.as_subclass(torch.Tensor)  # noqa: E731  (drop MetaTensor)
        nsd = nsd_per_class(plain(pred)[0, 0], plain(label)[0, 0].to(pred.dtype), present)
        cases[sid] = {
            "dice": {classes[c - 1]: float(dice[c - 1]) for c in present},
            **{f"nsd{t}": {classes[c - 1]: v for c, v in nsd[t].items()} for t in NSD_TOL},
        }
        tmp = out.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(res))
        os.replace(tmp, out)
        md = sum(cases[sid]["dice"].values()) / max(len(present), 1)
        print(f"  [{i + 1}/{len(todo)}] {sid}: {len(present)} classes, case mean dice {md:.4f}  "
              f"({(time.time() - t0) / (i + 1):.1f}s/case)", flush=True)
        del image, label, pred


if __name__ == "__main__":
    main()
