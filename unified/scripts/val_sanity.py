"""Per-model validation sanity check: 1 val subject, sliding-window inference,
Dice metric. Verifies the full validation pipeline (backbone + head + sliding
window + DiceMetric) wires up end-to-end. Random-init model by default — we're
not measuring quality, just exercising the path.

    python -m scripts.val_sanity --config configs/models/voco_b.yaml

For models that need a larger ROI (e.g. ct-clip needs >=160):

    python -m scripts.val_sanity --config configs/models/ctclip.yaml --roi-override 160
"""
from __future__ import annotations
import argparse
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from unified.utils import load_config, setup_logging, get_logger  # noqa: E402
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
    ap.add_argument("--subjects", type=int, default=1,
                    help="how many val subjects to evaluate")
    ap.add_argument("--roi-override", type=int, default=None,
                    help="override sliding-window roi_size (e.g. 160 for ct-clip)")
    ap.add_argument("--load-weights", action="store_true",
                    help="load pretrained backbone weights (default: random init)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.roi_override is not None:
        cfg["eval"]["sliding_window"]["roi_size"] = [args.roi_override] * 3
    setup_logging(None)
    log = get_logger("val_sanity")

    name = cfg["model"]["name"]
    if torch.cuda.is_available():
        dev_desc = f"cuda ({torch.cuda.get_device_name(0)})"
    else:
        dev_desc = "cpu"
    log.info("=" * 72)
    log.info("val sanity model=%s device=%s", name, dev_desc)
    log.info("roi=%s sw_batch=%d overlap=%.2f subjects=%d",
             cfg["eval"]["sliding_window"]["roi_size"],
             cfg["eval"]["sliding_window"]["sw_batch_size"],
             cfg["eval"]["sliding_window"]["overlap"],
             args.subjects)
    log.info("=" * 72)

    classes = load_classes()
    ids = _read_split(REPO / "unified" / "data" / "splits" / "val.txt")[: args.subjects]
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
    loader = DataLoader(Composed(ds, tf), batch_size=1, num_workers=2, pin_memory=False)

    mcfg = cfg["model"]
    weights = mcfg.get("weights") if args.load_weights else None
    backbone = build_backbone(mcfg["name"], weights=weights, **mcfg.get("kwargs", {}))
    head = UnifiedSegHead(
        num_classes=cfg["head"]["num_classes"],
        feature_channels=cfg["head"]["feature_channels"],
        feature_strides=cfg["head"]["feature_strides"],
        decoder_channels=cfg["head"]["decoder_channels"],
        norm=cfg["head"]["norm"],
    )
    model = SegModel(backbone, head)
    log.info("trainable params: %d", model.num_trainable_params())

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    log.info("device: %s", device)

    evaluator = Evaluator(cfg, classes)

    t0 = time.time()
    try:
        metrics = evaluator.evaluate(model, loader, device)
    except torch.cuda.OutOfMemoryError:
        log.warning("CUDA OOM during sliding-window inference; falling back to CPU")
        torch.cuda.empty_cache()
        model.to("cpu")
        metrics = evaluator.evaluate(model, loader, torch.device("cpu"))
    dt = time.time() - t0

    mean_dice = metrics["mean_dice"]
    log.info("VAL OK: %s — mean_dice=%.4f over %d subj, dt=%.1fs",
             name, mean_dice, args.subjects, dt)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
