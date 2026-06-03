"""Sanity-check the inference pipeline end-to-end and visualize it.

Runs the full eval-time path for a handful of subjects:

    raw subject -> val transforms -> model (sliding-window) -> argmax pred

and, for each subject, prints exactly what is passed to the model and what comes
back (shapes, dtypes, intensity / class statistics), then dumps a PNG montage
that puts the input CT, the ground-truth label, and the prediction side by side
on a few axial slices.

This is a debugging / visualization tool. It now also reports per-sample Dice and
HD95 against the ground truth (same metric definitions as the evaluator: foreground
-only, euclidean 95th-percentile Hausdorff, macro = mean over present foreground
classes), but ``scripts/evaluate.py`` remains the canonical evaluator over a full
split. A checkpoint is optional: omit ``--checkpoint`` to sanity check that an
untrained (randomly-initialised head + pretrained backbone) pipeline runs and
produces correctly-shaped outputs.

Example:
    python scripts/infer.py \
        --config configs/models/vista3d.yaml \
        --checkpoint runs/vista3d/best.pt \
        --split val --num-subjects 3 \
        --out-dir runs/infer_sanity
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from unified.utils import load_config, load_checkpoint, setup_logging, get_logger  # noqa: E402
from unified.data import (  # noqa: E402
    TotalSegmentatorDataset, load_classes, build_val_transforms,
)
from unified.models import build_backbone, build_head, SegModel  # noqa: E402


# ----------------------------------------------------------------------------- #
# Model / data construction (mirrors scripts/evaluate.py)
# ----------------------------------------------------------------------------- #
def build_model(cfg, device):
    """Build backbone + head + SegModel exactly as training/eval does."""
    mcfg = cfg["model"]
    backbone = build_backbone(mcfg["name"], weights=mcfg.get("weights"),
                              **mcfg.get("kwargs", {}))
    head = build_head(
        cfg["head"].get("name", "unified_seg_head"),
        num_classes=cfg["head"]["num_classes"],
        feature_channels=cfg["head"]["feature_channels"],
        feature_strides=cfg["head"]["feature_strides"],
        decoder_channels=cfg["head"]["decoder_channels"],
        norm=cfg["head"]["norm"],
        deep_supervision=cfg["head"].get("deep_supervision", False),
    )
    model = SegModel(backbone, head,
                     freeze_backbone=bool(mcfg.get("freeze_backbone", False)))
    model.to(device)
    model.eval()
    return model


def _read_split(path: Path):
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


class _Composed(torch.utils.data.Dataset):
    """Apply a transform pipeline lazily on top of a raw dataset."""
    def __init__(self, base, t):
        self.base, self.t = base, t

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.t(self.base[i])


# ----------------------------------------------------------------------------- #
# Visualization (PIL only -- matplotlib is not installed in this env)
# ----------------------------------------------------------------------------- #
# A fixed, perceptually-spread palette so the same class index gets the same
# colour in the GT and prediction panels. Index 0 (background) stays black.
def _label_palette(num_classes: int) -> np.ndarray:
    rng = np.random.RandomState(0)  # deterministic colours across runs
    palette = rng.randint(40, 256, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    return palette


def _to_uint8_gray(slice2d: np.ndarray) -> np.ndarray:
    """Normalise a 2D float slice to 0..255 grayscale for display."""
    lo, hi = float(slice2d.min()), float(slice2d.max())
    if hi - lo < 1e-6:
        return np.zeros_like(slice2d, dtype=np.uint8)
    return (((slice2d - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def _colorize_label(slice2d: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map an integer-label 2D slice to an RGB image via the palette."""
    idx = np.clip(slice2d.astype(np.int64), 0, len(palette) - 1)
    return palette[idx]


def _overlay(gray_rgb: np.ndarray, label_rgb: np.ndarray, alpha=0.5) -> np.ndarray:
    """Blend a colorized label over a grayscale background where label > 0."""
    out = gray_rgb.copy().astype(np.float32)
    mask = label_rgb.any(axis=-1)
    out[mask] = (1 - alpha) * out[mask] + alpha * label_rgb[mask].astype(np.float32)
    return out.astype(np.uint8)


def _orient_plane(sl2d: np.ndarray) -> np.ndarray:
    """Rotate an in-plane slice so the anatomy displays upright.

    After RAS orientation the volume axes are (R, A, S). For every view the
    in-plane slice has the head/anterior axis as its second dim, so transposing
    then flipping vertically puts superior/anterior at the top of the image.
    """
    return np.flipud(sl2d.T)


# Volume axes after RAS orientation are (R, A, S) -> plane = axis sliced along.
VIEWS = {"axial": 2, "coronal": 1, "sagittal": 0}


def save_montage(image, label, pred, palette, out_path: Path, n_slices=5,
                 axis=0):
    """Write a PNG: rows = [CT | GT overlay | Pred overlay], cols = slices.

    image: (D,H,W) float;  label/pred: (D,H,W) int (label may be None).
    Slices are picked along ``axis`` (default 0 = axial-ish), preferring slices
    where the prediction has the most foreground so empty slices don't dominate.
    """
    from PIL import Image

    image = np.asarray(image)
    pred = np.asarray(pred)
    has_label = label is not None
    if has_label:
        label = np.asarray(label)

    # Move the chosen slicing axis to front.
    image = np.moveaxis(image, axis, 0)
    pred = np.moveaxis(pred, axis, 0)
    if has_label:
        label = np.moveaxis(label, axis, 0)

    depth = image.shape[0]
    # Spread the slices over the central 10-90% of the foreground *mass* (GT if
    # present, else pred). Trimming to the mass percentiles (rather than the
    # first/last non-empty slice) keeps the strip on the dense middle region and
    # drops the near-blank end slices.
    ref = label if has_label else pred
    fg_per_slice = (ref > 0).reshape(depth, -1).sum(axis=1).astype(np.float64)
    total = fg_per_slice.sum()
    if total == 0:
        chosen = np.linspace(0, depth - 1, n_slices).astype(int)
    else:
        cdf = np.cumsum(fg_per_slice) / total
        lo = int(np.searchsorted(cdf, 0.10))
        hi = int(np.searchsorted(cdf, 0.90))
        if hi <= lo:  # degenerate (mass in a single slice) -> full extent
            fg_idx = np.flatnonzero(fg_per_slice > 0)
            lo, hi = int(fg_idx[0]), int(fg_idx[-1])
        chosen = np.unique(np.linspace(lo, hi, n_slices).astype(int))

    tiles = []
    for s in chosen:
        gray = _to_uint8_gray(_orient_plane(image[s]))
        gray_rgb = np.stack([gray] * 3, axis=-1)
        col = [gray_rgb]
        if has_label:
            col.append(_overlay(gray_rgb,
                                _colorize_label(_orient_plane(label[s]), palette)))
        col.append(_overlay(gray_rgb,
                            _colorize_label(_orient_plane(pred[s]), palette)))
        tiles.append(np.concatenate(col, axis=0))  # stack rows vertically

    grid = np.concatenate(tiles, axis=1)  # lay slices left-to-right
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)
    return out_path, chosen.tolist()


# ----------------------------------------------------------------------------- #
# Diagnostics
# ----------------------------------------------------------------------------- #
def _describe(name, t: torch.Tensor, log):
    arr = t.detach().float()
    log.info(
        f"  {name:12s} shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={arr.min().item():.3f} max={arr.max().item():.3f} "
        f"mean={arr.mean().item():.3f}"
    )


def compute_sample_metrics(pred, label, num_classes, classes, want_hd95):
    """Per-sample Dice (+ optional HD95) for one volume, mirroring the evaluator.

    ``pred`` and ``label`` are ``(1, 1, D, H, W)`` integer tensors. one-hot
    expansion + metrics run on CPU because a 117-class ``(1, C, D, H, W)`` tensor
    is several GiB and OOMs the GPU for the larger backbones (same reason the
    evaluator offloads). Uses the identical metric definitions as
    ``unified.evaluation.evaluator`` so these numbers match validation:
    foreground-only Dice (macro over classes present in the ground truth; a
    missed organ scores 0) and euclidean 95th-percentile Hausdorff (macro over
    classes present in BOTH gt and prediction, since a distance needs two
    non-empty surfaces).
    """
    import gc
    import warnings
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    from monai.networks.utils import one_hot

    # MONAI warns "the ground truth/prediction of class N is all 0 ..." for every
    # class absent on either side; we handle those explicitly, so silence them.
    empty_class_warn = r"the (ground truth|prediction) of class"

    pred_oh = one_hot(pred.detach().cpu(), num_classes=num_classes)
    label_oh = one_hot(label.detach().cpu(), num_classes=num_classes)
    sp = [0] + list(range(2, label_oh.ndim))            # batch + spatial dims
    gt_present = label_oh.sum(dim=sp)[1:] > 0           # foreground classes in GT

    dm = DiceMetric(include_background=False, reduction="mean_batch",
                    get_not_nans=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=empty_class_warn,
                                category=UserWarning)
        dm(y_pred=pred_oh, y=label_oh)
    dice_pc = dm.aggregate().cpu()                      # (num_classes - 1,)
    # Average over GT-present classes only. reduction="mean_batch" collapses
    # MONAI's empty-gt NaN to 0.0, so ``~isnan`` would NOT exclude gt-absent
    # classes (it would average over all 117) — derive presence from the label
    # instead, matching CT-FM's ignore_empty=True. A gt-present but missed organ
    # keeps its 0 Dice.
    res = {
        "macro_dice": dice_pc[gt_present].mean().item() if bool(gt_present.any()) else float("nan"),
        "dice_per_class": {classes[i]: float(dice_pc[i])
                           for i in range(len(classes)) if bool(gt_present[i])},
    }
    if want_hd95:
        hm = HausdorffDistanceMetric(include_background=False,
                                     distance_metric="euclidean", percentile=95,
                                     reduction="mean_batch", get_not_nans=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=empty_class_warn,
                                    category=UserWarning)
            hm(y_pred=pred_oh, y=label_oh)
        hd_pc = hm.aggregate().cpu()
        # HD95 only for classes present in BOTH gt and prediction (a distance
        # needs two non-empty surfaces). MONAI otherwise returns a spurious 0.0
        # when the prediction is empty and NaN when gt is empty; excluding
        # either-missing classes keeps this consistent with the evaluator and
        # avoids a completely-missed organ logging a fake 0 mm.
        pred_present = pred_oh.sum(dim=sp)[1:] > 0
        hmask = gt_present & pred_present & torch.isfinite(hd_pc)
        res["macro_hd95"] = hd_pc[hmask].mean().item() if bool(hmask.any()) else float("nan")
        res["hd95_per_class"] = {classes[i]: float(hd_pc[i])
                                 for i in range(len(classes)) if bool(hmask[i])}
    del pred_oh, label_oh
    gc.collect()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Model config YAML")
    ap.add_argument("--checkpoint", default=None,
                    help="Trained checkpoint .pt (optional; omit for an untrained "
                         "shape/sanity check)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--splits-dir",
                    default=str(REPO / "unified" / "data" / "splits"))
    ap.add_argument("--subject-ids", default=None,
                    help="Comma-separated subject IDs; overrides --split")
    ap.add_argument("--num-subjects", type=int, default=2,
                    help="How many subjects from the split to run")
    ap.add_argument("--out-dir", default=str(REPO / "runs" / "infer_sanity"),
                    help="Directory for PNG montages")
    ap.add_argument("--n-slices", type=int, default=5,
                    help="Axial slices per montage")
    ap.add_argument("--save-nifti", action="store_true",
                    help="Also write the prediction volume as a NIfTI per subject")
    ap.add_argument("--skip-metrics", action="store_true",
                    help="Skip the Dice/HD95 computation (montage / shape sanity only)")
    args = ap.parse_args()

    setup_logging(None)
    log = get_logger("infer")
    cfg = load_config(args.config)
    classes = load_classes()
    num_classes = cfg["data"]["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device}  model={cfg['model']['name']}  num_classes={num_classes}")

    # --- subjects ---------------------------------------------------------- #
    if args.subject_ids:
        ids = [s.strip() for s in args.subject_ids.split(",") if s.strip()]
    else:
        ids = _read_split(Path(args.splits_dir) / f"{args.split}.txt")
    ids = ids[: args.num_subjects]
    log.info(f"subjects: {ids}")

    # --- data -------------------------------------------------------------- #
    raw = TotalSegmentatorDataset(cfg["data"]["dataset_root"], ids, classes)
    tf = build_val_transforms(cfg)
    ds = _Composed(raw, tf)

    # --- model ------------------------------------------------------------- #
    model = build_model(cfg, device)
    if args.checkpoint:
        epoch, extra = load_checkpoint(args.checkpoint, model=model,
                                       map_location=device, strict=True)
        log.info(f"loaded checkpoint {args.checkpoint} (epoch={epoch}, extra={extra})")
    else:
        log.warning("no checkpoint -- running with UNTRAINED head; "
                    "predictions are meaningless, this only checks shapes/flow")

    # --- sliding-window inferer (same settings as the evaluator) ----------- #
    from monai.inferers import sliding_window_inference
    sw = cfg["eval"]["sliding_window"]
    roi = tuple(sw["roi_size"])
    log.info(f"sliding window: roi={roi} sw_batch={sw['sw_batch_size']} "
             f"overlap={sw['overlap']} mode={sw['mode']}")

    # Match the evaluator: compute HD95 only when the eval config asks for it.
    want_hd95 = "hd95" in cfg["eval"].get("metrics", [])
    do_metrics = not args.skip_metrics

    palette = _label_palette(num_classes)
    out_dir = Path(args.out_dir)
    macro_dice_all: list[float] = []
    macro_hd95_all: list[float] = []

    for i in range(len(ds)):
        sample = ds[i]
        sid = ids[i]
        image = sample["image"].unsqueeze(0).to(device)   # (1,1,D,H,W)
        label = sample.get("label")
        if label is not None:
            label = label.unsqueeze(0)                     # (1,1,D,H,W) on CPU

        log.info(f"[{sid}] INPUT to model:")
        _describe("image", image, log)
        if label is not None:
            present = torch.unique(label).tolist()
            _describe("label", label, log)
            log.info(f"  label classes present: {len(present)} "
                     f"(e.g. {present[:10]}{'...' if len(present) > 10 else ''})")

        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=image, roi_size=roi,
                sw_batch_size=sw["sw_batch_size"], predictor=model,
                overlap=sw["overlap"], mode=sw["mode"],
            )
        pred = logits.argmax(dim=1, keepdim=True)          # (1,1,D,H,W)

        log.info(f"[{sid}] OUTPUT from model:")
        _describe("logits", logits, log)
        _describe("pred", pred, log)
        pred_classes = torch.unique(pred).tolist()
        log.info(f"  pred classes present: {len(pred_classes)} "
                 f"(e.g. {pred_classes[:10]}{'...' if len(pred_classes) > 10 else ''})")

        # --- per-sample metrics vs ground truth ---------------------------- #
        if label is not None and do_metrics:
            m = compute_sample_metrics(pred, label, num_classes, classes, want_hd95)
            n_d = len(m["dice_per_class"])
            log.info(f"[{sid}] METRICS vs GT (full-volume, foreground-only):")
            log.info(f"  macro Dice = {m['macro_dice']:.4f}  "
                     f"(mean over {n_d} foreground classes present in GT)")
            if "macro_hd95" in m:
                n_h = len(m["hd95_per_class"])
                log.info(f"  macro HD95 = {m['macro_hd95']:.2f} voxels  "
                         f"(mean over {n_h} classes)")
            # Per-class breakdown, worst Dice first (most useful for debugging).
            for name, dv in sorted(m["dice_per_class"].items(), key=lambda kv: kv[1]):
                hv = m.get("hd95_per_class", {}).get(name)
                hs = f"  hd95={hv:7.2f}" if hv is not None else ""
                log.info(f"    dice={dv:.4f}{hs}  {name}")
            mj = out_dir / f"{sid}_metrics.json"
            mj.parent.mkdir(parents=True, exist_ok=True)
            import json as _json
            mj.write_text(_json.dumps({"id": sid, **m}, indent=2))
            log.info(f"[{sid}] metrics -> {mj}")
            if m["macro_dice"] == m["macro_dice"]:  # not NaN
                macro_dice_all.append(m["macro_dice"])
            if m.get("macro_hd95", float("nan")) == m.get("macro_hd95", float("nan")):
                macro_hd95_all.append(m["macro_hd95"])

        # --- visualize ----------------------------------------------------- #
        img_np = image[0, 0].cpu().numpy()
        pred_np = pred[0, 0].cpu().numpy()
        label_np = label[0, 0].cpu().numpy() if label is not None else None
        rows = "CT | GT | Pred" if label is not None else "CT | Pred"
        for view, axis in VIEWS.items():
            png, slices = save_montage(
                img_np, label_np, pred_np, palette,
                out_dir / f"{sid}_{view}.png", n_slices=args.n_slices, axis=axis,
            )
            log.info(f"[{sid}] {view:8s} montage -> {png}  "
                     f"(rows: {rows}; slices {slices})")

        if args.save_nifti:
            import nibabel as nib
            nii = out_dir / f"{sid}_pred.nii.gz"
            nib.save(nib.Nifti1Image(pred_np.astype(np.int16), affine=np.eye(4)), nii)
            log.info(f"[{sid}] prediction volume -> {nii}")

        del logits, pred, image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if macro_dice_all:
        import statistics
        msg = (f"summary over {len(macro_dice_all)} subject(s): "
               f"mean macro Dice = {statistics.mean(macro_dice_all):.4f}")
        if macro_hd95_all:
            msg += f"  |  mean macro HD95 = {statistics.mean(macro_hd95_all):.2f} voxels"
        log.info(msg)

    log.info(f"done -- montages in {out_dir}")


if __name__ == "__main__":
    main()
