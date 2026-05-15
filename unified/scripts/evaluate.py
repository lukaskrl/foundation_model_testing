"""Evaluate a trained model on the test split."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from unified.utils import load_config, load_checkpoint, setup_logging  # noqa: E402
from unified.data import (
    TotalSegmentatorDataset, load_classes, build_val_transforms,
)  # noqa: E402
from unified.models import build_backbone, UnifiedSegHead, SegModel  # noqa: E402
from unified.evaluation import Evaluator  # noqa: E402


def _read_split(path: Path):
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--splits-dir", default=str(REPO / "unified" / "data" / "splits"))
    ap.add_argument("--out", default=None,
                    help="JSON file to dump metrics into")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(None)
    classes = load_classes()

    ids = _read_split(Path(args.splits_dir) / f"{args.split}.txt")
    ds = TotalSegmentatorDataset(cfg["data"]["dataset_root"], ids, classes)
    tf = build_val_transforms(cfg)

    class Composed(torch.utils.data.Dataset):
        def __init__(self, base, t):
            self.base, self.t = base, t

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            return self.t(self.base[i])

    from monai.data import DataLoader
    loader = DataLoader(Composed(ds, tf), batch_size=1, num_workers=cfg["data"]["num_workers"])

    mcfg = cfg["model"]
    backbone = build_backbone(mcfg["name"], weights=None, **mcfg.get("kwargs", {}))
    head = UnifiedSegHead(
        num_classes=cfg["head"]["num_classes"],
        feature_channels=cfg["head"]["feature_channels"],
        feature_strides=cfg["head"]["feature_strides"],
        decoder_channels=cfg["head"]["decoder_channels"],
        norm=cfg["head"]["norm"],
    )
    model = SegModel(backbone, head)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    load_checkpoint(args.checkpoint, model=model, map_location=device, strict=True)

    evaluator = Evaluator(cfg, classes)
    metrics = evaluator.evaluate(model, loader, device)
    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
