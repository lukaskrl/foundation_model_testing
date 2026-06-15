"""Smoke-test checkpoint loading + eval for ONE model, in a fresh process.

Builds the backbone WITH its pretrained weights (the load path the VRAM probe
skipped), wires the head + SegModel exactly as training does, and runs the real
Evaluator on a single subject. The head is random (no fine-tuned checkpoint), so
Dice is meaningless — this only checks that weights load and the full eval path
(sliding window + confusion-matrix Dice + present-class HD95) runs without
crashing.

Prints one machine-parseable line:
    RESULT {"model": .., "status": "ok", "mean_dice": .., "mean_hd95": .., "dt_s": ..}
    RESULT {"model": .., "status": "oom"}                       (exit 3)
    RESULT {"model": .., "status": "error", "msg": ".."}        (exit 2)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from unified.utils import load_config  # noqa: E402
from unified.data import (  # noqa: E402
    TotalSegmentatorDataset, load_classes, build_val_transforms,
)
from unified.models import build_backbone, build_head, SegModel  # noqa: E402
from unified.evaluation import Evaluator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subject", default="s0000")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["model"]["name"]

    def emit(d, code):
        print("RESULT " + json.dumps({"model": name, **d}), flush=True)
        sys.exit(code)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        classes = load_classes()
        raw = TotalSegmentatorDataset(cfg["data"]["dataset_root"], [args.subject], classes)
        tf = build_val_transforms(cfg)

        class C(torch.utils.data.Dataset):
            def __init__(s, b, t): s.b, s.t = b, t
            def __len__(s): return len(s.b)
            def __getitem__(s, i): return s.t(s.b[i])

        from monai.data import DataLoader
        loader = DataLoader(C(raw, tf), batch_size=1, num_workers=0)

        mcfg = cfg["model"]
        t_load = time.time()
        # weights=mcfg["weights"] exercises the adapter's checkpoint loading.
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
        load_s = time.time() - t_load

        evaluator = Evaluator(cfg, classes)
        t0 = time.time()
        metrics = evaluator.evaluate(model, loader, device)
        dt = time.time() - t0

        emit({"status": "ok",
              "mean_dice": round(metrics["mean_dice"], 4),
              "mean_hd95": (round(metrics["mean_hd95"], 2)
                            if "mean_hd95" in metrics else None),
              "load_s": round(load_s, 1), "dt_s": round(dt, 1),
              "params_M": round(model.num_total_params() / 1e6, 1)}, 0)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        emit({"status": "oom"}, 3)
    except Exception as e:  # noqa: BLE001
        if "out of memory" in str(e).lower():
            emit({"status": "oom"}, 3)
        traceback.print_exc()
        emit({"status": "error", "msg": f"{type(e).__name__}: {e}"[:240]}, 2)


if __name__ == "__main__":
    main()
