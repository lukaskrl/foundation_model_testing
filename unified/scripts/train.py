"""Train a single model on TotalSegmentator with the shared config."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from unified.utils import load_config, setup_logging, get_logger  # noqa: E402
from unified.data import (
    TotalSegmentatorDataset, load_classes,
    build_train_transforms, build_val_transforms,
    build_cache_det_transforms,
    build_train_post_transforms, build_val_post_transforms,
    CachedDataset, preprocessing_fingerprint,
)  # noqa: E402
from unified.models import build_backbone, build_head, SegModel  # noqa: E402
from unified.training import Trainer  # noqa: E402
from unified.evaluation import Evaluator  # noqa: E402


def _read_split(path: Path):
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def _build_dataset(cfg, raw, *, training, cache_dir):
    """Wrap the raw NIfTI dataset in the preprocessing pipeline.

    If ``cache_dir`` is given, the model-independent prefix (EnsureTyped →
    Spacingd → CropForegroundd) is cached to disk per subject. Orientation,
    intensity windowing, and augmentations run on the fly each epoch so the
    cache is shared across every model.
    """
    import torch

    if cache_dir is not None:
        det = build_cache_det_transforms(cfg)
        post = (build_train_post_transforms(cfg) if training
                else build_val_post_transforms(cfg))
        return CachedDataset(raw, det_transforms=det,
                             cache_dir=cache_dir, post_transforms=post)

    transforms = build_train_transforms(cfg) if training else build_val_transforms(cfg)

    class Composed(torch.utils.data.Dataset):
        def __init__(self, base, t):
            self.base = base
            self.t = t

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            return self.t(self.base[i])

    return Composed(raw, transforms)


def _build_loader(cfg, ds, batch_size, shuffle):
    from monai.data import DataLoader
    nw = cfg["data"]["num_workers"]
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=(nw > 0),
        prefetch_factor=(4 if nw > 0 else None),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="model config YAML")
    ap.add_argument("--output", required=True, help="output dir for run artifacts")
    ap.add_argument("--splits-dir", default=str(REPO / "unified" / "data" / "splits"))
    ap.add_argument("--epochs", type=int, default=None,
                    help="override cfg.train.epochs (for sanity/benchmark runs)")
    ap.add_argument("--resume", nargs="?", const=True, default=None, metavar="CKPT",
                    help="resume from a checkpoint; omit path to auto-detect the "
                         "latest checkpoint inside --output")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    setup_logging(args.output)
    log = get_logger("train")

    import torch
    if torch.cuda.is_available():
        dev_desc = f"cuda ({torch.cuda.get_device_name(0)})"
    else:
        dev_desc = "cpu"
    log.info("=" * 72)
    log.info("model=%s config=%s", cfg["model"]["name"], args.config)
    log.info("output=%s device=%s", args.output, dev_desc)
    effective_bs = cfg["model"].get("batch_size", cfg["train"]["batch_size"])
    effective_amp = cfg["model"].get("amp", cfg["train"].get("amp", True))
    effective_accum = max(1, int(cfg["model"].get(
        "grad_accum_steps", cfg["train"].get("grad_accum_steps", 1))))
    log.info(
        "epochs=%d batch_size=%d grad_accum=%d eff_batch=%d lr=%g amp=%s",
        cfg["train"]["epochs"], effective_bs, effective_accum,
        effective_bs * effective_accum,
        cfg["train"].get("optimizer", {}).get("lr", float("nan")),
        effective_amp,
    )
    log.info("=" * 72)

    classes = load_classes()
    splits_dir = Path(args.splits_dir)
    train_ids = _read_split(splits_dir / "train.txt")
    val_ids = _read_split(splits_dir / "val.txt")
    log.info("splits: train=%d val=%d classes=%d",
             len(train_ids), len(val_ids), len(classes))

    train_raw = TotalSegmentatorDataset(cfg["data"]["dataset_root"], train_ids, classes)
    val_raw = TotalSegmentatorDataset(cfg["data"]["dataset_root"], val_ids, classes)

    # Disk cache: ``<cache.dir>/<fingerprint>/<subject>.pt``. Fingerprint
    # encodes axcodes/spacing/intensity/margin so per-model preprocessing
    # variants never collide.
    cache_cfg = (cfg["data"].get("cache") or {})
    cache_root = None
    if cache_cfg.get("enabled", False):
        fp = preprocessing_fingerprint(cfg)
        cache_root = Path(cache_cfg["dir"]) / fp
        log.info("disk cache: %s (fingerprint=%s)", cache_root, fp)

    train_ds = _build_dataset(cfg, train_raw, training=True, cache_dir=cache_root)
    val_ds = _build_dataset(cfg, val_raw, training=False, cache_dir=cache_root)

    # Per-model batch_size override (filed under model: so the base.yaml
    # comparison guard stays in place). Falls back to train.batch_size.
    bs = cfg["model"].get("batch_size", cfg["train"]["batch_size"])
    train_loader = _build_loader(cfg, train_ds, batch_size=bs, shuffle=True)
    val_loader = _build_loader(cfg, val_ds, batch_size=1, shuffle=False)

    # Build model.
    mcfg = cfg["model"]
    backbone = build_backbone(
        mcfg["name"],
        weights=mcfg.get("weights"),
        **mcfg.get("kwargs", {}),
    )
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
    log.info("model %s, trainable params: %d / %d total (freeze_backbone=%s)",
             mcfg["name"], model.num_trainable_params(),
             model.num_total_params(), model.freeze_backbone)

    evaluator = Evaluator(cfg, classes)
    trainer = Trainer(cfg, model, train_loader, val_loader, evaluator, args.output,
                      resume=args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
