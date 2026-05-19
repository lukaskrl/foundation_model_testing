"""Sliding-window evaluation with per-class Dice and (optionally) HD95.

Same evaluator across all models. Operates at the spacing/orientation produced
by the val transforms.
"""
from __future__ import annotations
from typing import Dict, List

import torch
from tqdm.auto import tqdm

from ..utils import get_logger


class Evaluator:
    def __init__(self, cfg, classes: List[str]):
        self.cfg = cfg
        self.classes = classes  # length 117 (does not include background)
        self.num_classes = cfg["data"]["num_classes"]
        self.log = get_logger("eval")
        sw = cfg["eval"]["sliding_window"]
        self.roi = tuple(sw["roi_size"])
        self.sw_batch = sw["sw_batch_size"]
        self.overlap = sw["overlap"]
        self.mode = sw["mode"]
        self.want_hd95 = "hd95" in cfg["eval"]["metrics"]

    @torch.no_grad()
    def evaluate(self, model, loader, device) -> Dict:
        from monai.inferers import sliding_window_inference
        from monai.metrics import DiceMetric

        model.eval()
        dice = DiceMetric(include_background=False, reduction="mean_batch",
                          get_not_nans=False)
        per_class_running = torch.zeros(self.num_classes - 1)
        per_class_count = torch.zeros(self.num_classes - 1)

        try:
            total = len(loader)
        except TypeError:
            total = None
        pbar = tqdm(
            loader,
            total=total,
            desc=f"[val] roi={tuple(self.roi)}",
            dynamic_ncols=True,
            leave=False,
        )
        for batch in pbar:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            logits = sliding_window_inference(
                inputs=image,
                roi_size=self.roi,
                sw_batch_size=self.sw_batch,
                predictor=model,
                overlap=self.overlap,
                mode=self.mode,
            )
            pred = logits.argmax(dim=1, keepdim=True)

            # Convert to one-hot for DiceMetric
            from monai.networks.utils import one_hot
            pred_oh = one_hot(pred, num_classes=self.num_classes)
            label_oh = one_hot(label, num_classes=self.num_classes)

            dice(y_pred=pred_oh, y=label_oh)
            # accumulate per-class
            per = dice.aggregate().cpu()  # length num_classes-1 if include_background=False
            dice.reset()
            mask = ~torch.isnan(per)
            per_class_running[mask] += per[mask]
            per_class_count[mask] += 1

            running_mean = (
                per_class_running / per_class_count.clamp(min=1)
            )
            valid_running = running_mean[per_class_count > 0]
            postfix = {
                "dice": f"{valid_running.mean().item():.4f}" if valid_running.numel() else "n/a",
                "classes": f"{int((per_class_count > 0).sum().item())}/{self.num_classes - 1}",
            }
            if torch.cuda.is_available():
                postfix["gpu_GB"] = f"{torch.cuda.max_memory_allocated() / 1e9:.1f}"
            pbar.set_postfix(postfix, refresh=False)
        pbar.close()

        per_class_mean = (per_class_running / per_class_count.clamp(min=1)).tolist()
        valid = [x for x, c in zip(per_class_mean, per_class_count.tolist()) if c > 0]
        mean_dice = sum(valid) / max(1, len(valid))

        out = {"mean_dice": mean_dice}
        for name, d in zip(self.classes, per_class_mean):
            out[f"dice/{name}"] = d
        return out
