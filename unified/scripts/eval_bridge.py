"""Score ANY checkpoint — ours or CT-FM's original repo — under BOTH protocols.

Arm A (CT-FM's own lighter pipeline) and this repo report numbers that are not
comparable as they stand. This script removes every difference except the model.

What was verified before writing this (2026-09-22), and what it buys:

* **Identical label space.** CT-FM's `get_ts_datalist` reads `<subject>/label.nii.gz`
  — the *merged* label file this repo writes, whose values are ALPHABETICAL class
  indices (`unified/data/classes.txt`), not TotalSegmentator's official numbering.
  Verified voxel-exact against `segmentations/*.nii.gz` on s0000. Their config's
  `LabelFilterd`/`MapLabelValued` are the identity over 0..117, so both models
  predict the same 118 values with the same meaning.
  CONSEQUENCE: arm A's Macro_Dice is unaffected (a permutation of class identity
  does not change a mean over classes), but its per-class names in W&B are WRONG —
  `get_ts_class_labels` names channel k by the official map. Per-class numbers from
  arm A must be renamed through `classes.txt`, which is what this script does.
* **Identical geometry and intensity.** The dataset is already 1.5 mm isotropic, so
  the `Spacingd` this repo applies and CT-FM omits is a no-op. Both use SPL
  orientation, `CropForegroundd(margin=10)` and HU [-1024, 2048] -> [0, 1].

So the protocols differ in exactly three knobs, each toggled here:

  1. spatial   full foreground-cropped volume   vs  a 192x240x240 crop of it
  2. window    roi (96,96,96)  overlap 0.50     vs  roi (96,160,160) overlap 0.625
  3. reduction macro-by-CLASS                   vs  macro-by-CASE

Knob 3 is free — both reductions come from the same per-case confusion matrices,
so every pass reports both. Knobs 1 and 2 change inference and cost a pass each.

`by_class` (this repo): per-class mean over the cases where the class is in the gt,
then the mean over classes that appeared at least once. Every class counts equally.

`by_case` (CT-FM): per-case mean over the classes present in that case's gt, then
the mean over cases. This is `monai.metrics.DiceHelper(include_background=False,
reduction="mean", ignore_empty=True)` aggregated over cases, i.e. what
`project.metrics.monai.DiceScore` computes. Every CASE counts equally, so a class
present in 3 of 57 subjects barely registers.

CT-FM validates on `RandSpatialCropd`. This script uses a CENTER crop instead: the
point is a comparable number, not a reproduction of their sampling noise.

Usage:
    # arm A's checkpoint under both protocols, on the untouched test split
    python -m scripts.eval_bridge --model ctfm_original \\
        --checkpoint ../CT-FM/evaluation/runs/totalseg/checkpoints/ct_fm_headtohead_v1/best.ckpt \\
        --split test --out results/bridge/armA_test.json

    # one of ours, same protocols
    python -m scripts.eval_bridge --model ours --config configs/models/ctfm.yaml \\
        --checkpoint runs/lowshot/ls_ctfm_ft_pt_f100/best.pt \\
        --split test --out results/bridge/ours_ctfm_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from unified.utils import load_config, load_checkpoint, setup_logging  # noqa: E402
from unified.data import TotalSegmentatorDataset, load_classes  # noqa: E402

# The two protocols, as (spatial crop, sliding-window roi, overlap).
PROTOCOLS = {
    "ours_fullvol": {
        "crop": None,
        "roi": (96, 96, 96),
        "overlap": 0.5,
        "what": "this repo: full foreground-cropped volume, roi 96^3, overlap 0.50",
    },
    "ctfm_valcrop": {
        "crop": (192, 240, 240),
        "roi": (96, 160, 160),
        "overlap": 0.625,
        "what": "CT-FM evaluation/totalseg.yaml val path: 192x240x240 crop, roi 96x160x160, overlap 0.625",
    },
}


def build_model(args, device):
    """Return an eval-mode nn.Module mapping (B,1,D,H,W) -> (B,118,D,H,W)."""
    if args.model == "ours":
        from unified.models import build_backbone, build_head, SegModel
        cfg = load_config(args.config)
        mcfg = cfg["model"]
        # A FROZEN run's checkpoint stores only the trainable adapter+head; the encoder
        # weights are not in it. Building with weights=None would leave a RANDOM encoder
        # under a head trained on the pretrained one. load_checkpoint's omitted-tensor
        # hash catches that, so pass the same weights the run was built with.
        weights = mcfg.get("weights") if mcfg.get("pretrained", True) else None
        backbone = build_backbone(mcfg["name"], weights=weights, **mcfg.get("kwargs", {}))
        head_kwargs = {k: v for k, v in cfg["head"].items() if k != "name"}
        if hasattr(backbone, "head_kwargs"):
            head_kwargs.update(backbone.head_kwargs())
        head = build_head(cfg["head"].get("name", "unified_seg_head"), **head_kwargs)
        model = SegModel(backbone, head,
                         freeze_backbone=bool(mcfg.get("freeze_backbone", False)))
        model.to(device)
        load_checkpoint(args.checkpoint, model=model, map_location=device, strict=True)
        return model.eval(), cfg

    # CT-FM's original: monai SegResNetDS, rebuilt HERE rather than through lighter,
    # so both models run in one process against one data pipeline. Their System
    # stores it as `model.trunk.*` (TrunkHeadWrapper with head=None).
    from monai.networks.nets import SegResNetDS
    net = SegResNetDS(spatial_dims=3, in_channels=1, out_channels=args.num_classes,
                      init_filters=32, blocks_down=[1, 2, 2, 4, 4], dsdepth=4)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    prefix = "model.trunk."
    stripped = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not stripped:
        raise SystemExit(f"no keys under {prefix!r} in {args.checkpoint}")
    net.load_state_dict(stripped, strict=True)
    print(f"[bridge] loaded {len(stripped)} tensors into SegResNetDS "
          f"(epoch {ckpt.get('epoch')}, global_step {ckpt.get('global_step')})")

    class _Wrap(torch.nn.Module):
        """SegResNetDS returns a deep-supervision LIST while training; a single
        tensor in eval. Guard anyway so the inferer always sees a tensor."""
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            out = self.net(x)
            return out[0] if isinstance(out, (list, tuple)) else out

    return _Wrap(net).to(device).eval(), None


def build_loader(cfg, split, splits_dir, classes, limit, num_workers):
    """One data pipeline for both models: our val transforms, which were verified
    to match CT-FM's val preprocessing step for step (see module docstring)."""
    from monai.data import DataLoader
    from unified.data import build_val_transforms

    ids = [l.strip() for l in (Path(splits_dir) / f"{split}.txt").read_text().splitlines()
           if l.strip()]
    if limit:
        ids = ids[:limit]
    ds = TotalSegmentatorDataset(cfg["data"]["dataset_root"], ids, classes,
                                use_source_affine=bool(cfg["data"].get("use_source_affine", False)))
    tf = build_val_transforms(cfg)

    class _Composed(torch.utils.data.Dataset):
        def __init__(self, base, t):
            self.base, self.t = base, t

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            return self.t(self.base[i])

    return DataLoader(_Composed(ds, tf), batch_size=1, num_workers=num_workers), ids


def center_crop_pad(image, label, size):
    """CT-FM's val spatial op, made deterministic: center crop then constant pad."""
    from monai.transforms import CenterSpatialCrop, SpatialPad
    crop, pad = CenterSpatialCrop(roi_size=size), SpatialPad(spatial_size=size, mode="constant")
    return (pad(crop(image[0]))[None], pad(crop(label[0]))[None])


@torch.no_grad()
def run_protocol(model, loader, device, proto, num_classes, classes):
    """One inference pass; returns BOTH reductions plus per-class detail."""
    from monai.inferers import sliding_window_inference

    nc = num_classes - 1
    d_run, d_cnt = torch.zeros(nc, dtype=torch.float64), torch.zeros(nc)
    per_case = []          # per-case mean over gt-present classes -> the by_case metric
    t0 = time.time()

    for n, batch in enumerate(loader, 1):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        if proto["crop"] is not None:
            image, label = center_crop_pad(image, label, proto["crop"])

        logits = sliding_window_inference(
            inputs=image, roi_size=proto["roi"], sw_batch_size=2,
            predictor=model, overlap=proto["overlap"], mode="gaussian",
        )
        pred = logits.argmax(dim=1, keepdim=True)
        del logits

        cm = torch.bincount(label.reshape(-1).to(torch.int64) * num_classes
                            + pred.reshape(-1).to(torch.int64),
                            minlength=num_classes * num_classes
                            ).reshape(num_classes, num_classes).double()
        tp, gt, pr = torch.diag(cm), cm.sum(1), cm.sum(0)
        dice = (2.0 * tp / (gt + pr).clamp(min=1.0))[1:].cpu()   # drop background
        present = (gt[1:] > 0).cpu()

        d_run[present] += dice[present]
        d_cnt[present] += 1
        # ignore_empty=True: a case contributes the mean over ITS gt-present classes.
        per_case.append(float(dice[present].mean()) if present.any() else float("nan"))

        del image, label, pred, cm
        if n % 10 == 0:
            print(f"    {n} cases, by_case running mean "
                  f"{sum(v for v in per_case if v == v) / max(1, len(per_case)):.4f}", flush=True)

    seen = d_cnt > 0
    by_class_per = (d_run / d_cnt.clamp(min=1))
    valid = [float(v) for v in per_case if v == v]
    return {
        "protocol": proto["what"],
        "roi": list(proto["roi"]), "overlap": proto["overlap"],
        "crop": list(proto["crop"]) if proto["crop"] else None,
        "n_cases": len(per_case),
        "macro_dice_by_class": float(by_class_per[seen].mean()),
        "macro_dice_by_case": (sum(valid) / len(valid)) if valid else float("nan"),
        "n_classes_seen": int(seen.sum()),
        "seconds": round(time.time() - t0, 1),
        "per_class": {name: (float(v) if s else None)
                      for name, v, s in zip(classes, by_class_per.tolist(), seen.tolist())},
        "per_case": per_case,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["ours", "ctfm_original"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None,
                    help="required for --model ours; for ctfm_original only supplies the data block")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--protocols", default="ours_fullvol,ctfm_valcrop")
    ap.add_argument("--splits-dir", default=str(REPO / "unified" / "data" / "splits"))
    ap.add_argument("--num-classes", type=int, default=118)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source-affine", choices=["auto", "true", "false"], default="auto",
                    help="override data.use_source_affine, i.e. the RESOLUTION the model is\n                         fed. 'true' = native 1.5 mm, 'false' = 2.25 mm (Spacingd\n                         re-resamples). This is not a metric knob: a checkpoint must be\n                         scored at the resolution it was TRAINED at, and CT-FM's repo\n                         trains at 1.5 mm while this repo's corpus is 2.25 mm.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    setup_logging(None)
    classes = load_classes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For the CT-FM model there is no unified config; borrow the ctfm one purely for
    # its data/eval block, which is exactly the preprocessing their config specifies.
    data_cfg = load_config(args.config or str(REPO / "configs" / "models" / "ctfm.yaml"))
    if args.source_affine != "auto":
        data_cfg["data"]["use_source_affine"] = (args.source_affine == "true")
    src = bool(data_cfg["data"].get("use_source_affine", False))
    print(f"[bridge] voxel geometry: use_source_affine={src} -> "
          f"{'1.5mm (native)' if src else '2.25mm (Spacingd re-resamples)'}", flush=True)
    model, _ = build_model(args, device)

    out = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "label_space": "alphabetical classes.txt indices 1..117 (verified voxel-exact vs segmentations/)",
        "use_source_affine": src,
        "effective_spacing_mm": 1.5 if src else 2.25,
        "results": {},
    }
    for key in [p.strip() for p in args.protocols.split(",") if p.strip()]:
        if key not in PROTOCOLS:
            raise SystemExit(f"unknown protocol {key!r}; have {sorted(PROTOCOLS)}")
        loader, ids = build_loader(data_cfg, args.split, args.splits_dir, classes,
                                   args.limit, args.num_workers)
        print(f"[bridge] {args.model} | {key} | {len(ids)} subjects", flush=True)
        out["results"][key] = run_protocol(model, loader, device, PROTOCOLS[key],
                                           args.num_classes, classes)
        r = out["results"][key]
        print(f"[bridge] {key}: by_class {r['macro_dice_by_class']:.4f}  "
              f"by_case {r['macro_dice_by_case']:.4f}  ({r['seconds']}s)", flush=True)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=2))

    summary = {k: {"by_class": round(v["macro_dice_by_class"], 4),
                   "by_case": round(v["macro_dice_by_case"], 4)}
               for k, v in out["results"].items()}
    print(json.dumps(summary, indent=2))
    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
