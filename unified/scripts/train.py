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
)  # noqa: E402
from unified.models import build_backbone, UnifiedSegHead, SegModel  # noqa: E402
from unified.training import Trainer  # noqa: E402
from unified.evaluation import Evaluator  # noqa: E402


def _read_split(path: Path):
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def _build_loader(cfg, ds, transforms, batch_size, shuffle):
    import torch
    from monai.data import Dataset as MonaiDataset, DataLoader

    # Wrap MONAI transforms over the raw dataset.
    class Composed(torch.utils.data.Dataset):
        def __init__(self, base, t):
            self.base = base
            self.t = t

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            return self.t(self.base[i])

    composed = Composed(ds, transforms)
    return DataLoader(
        composed,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=shuffle,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="model config YAML")
    ap.add_argument("--output", required=True, help="output dir for run artifacts")
    ap.add_argument("--splits-dir", default=str(REPO / "unified" / "data" / "splits"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(args.output)
    log = get_logger("train")
    log.info("config %s, output %s", args.config, args.output)

    classes = load_classes()
    splits_dir = Path(args.splits_dir)
    train_ids = _read_split(splits_dir / "train.txt")
    val_ids = _read_split(splits_dir / "val.txt")

    train_ds = TotalSegmentatorDataset(cfg["data"]["dataset_root"], train_ids, classes)
    val_ds = TotalSegmentatorDataset(cfg["data"]["dataset_root"], val_ids, classes)

    train_tf = build_train_transforms(cfg)
    val_tf = build_val_transforms(cfg)

    train_loader = _build_loader(cfg, train_ds, train_tf,
                                 batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = _build_loader(cfg, val_ds, val_tf, batch_size=1, shuffle=False)

    # Build model.
    mcfg = cfg["model"]
    backbone = build_backbone(
        mcfg["name"],
        weights=mcfg.get("weights"),
        **mcfg.get("kwargs", {}),
    )
    head = UnifiedSegHead(
        num_classes=cfg["head"]["num_classes"],
        feature_channels=cfg["head"]["feature_channels"],
        feature_strides=cfg["head"]["feature_strides"],
        decoder_channels=cfg["head"]["decoder_channels"],
        norm=cfg["head"]["norm"],
    )
    model = SegModel(backbone, head)
    log.info("model %s, trainable params: %d",
             mcfg["name"], model.num_trainable_params())

    evaluator = Evaluator(cfg, classes)
    trainer = Trainer(cfg, model, train_loader, val_loader, evaluator, args.output)
    trainer.run()


if __name__ == "__main__":
    main()
