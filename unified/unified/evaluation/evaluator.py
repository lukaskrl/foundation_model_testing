"""Sliding-window evaluation with per-class Dice and (optionally) HD95.

Same evaluator across all models. Operates at the spacing/orientation produced
by the val transforms.
"""
from __future__ import annotations
import ctypes
import gc
from typing import Dict, List

import torch
from tqdm.auto import tqdm

from ..utils import get_logger

try:
    _libc_malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
    _libc_malloc_trim.argtypes = [ctypes.c_size_t]
except OSError:
    _libc_malloc_trim = None


def _release_heap():
    """Force glibc to return freed memory to the OS.

    The 117-class one-hot tensors used per val iteration churn the heap badly
    and would otherwise leave the process with steadily growing RSS, OOM-killing
    long sweeps. malloc_trim(0) trims the heap each iteration.
    """
    gc.collect()
    if _libc_malloc_trim is not None:
        _libc_malloc_trim(0)


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
        from monai.metrics import DiceMetric, HausdorffDistanceMetric

        model.eval()
        dice = DiceMetric(include_background=False, reduction="mean_batch",
                          get_not_nans=False)
        hd95 = (
            HausdorffDistanceMetric(
                include_background=False,
                distance_metric="euclidean",
                percentile=95,
                reduction="mean_batch",
                get_not_nans=False,
            )
            if self.want_hd95
            else None
        )
        dice_running = torch.zeros(self.num_classes - 1)
        dice_count = torch.zeros(self.num_classes - 1)
        hd95_running = torch.zeros(self.num_classes - 1)
        hd95_count = torch.zeros(self.num_classes - 1)

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
            # Move argmax + label to CPU before the (1, C, D, H, W) one-hot
            # expansion. For 117-class CT volumes that is ~6 GiB per tensor;
            # keeping it on GPU OOMs the larger backbones (e.g. voco_h).
            pred_cpu = logits.argmax(dim=1, keepdim=True).cpu()
            label_cpu = label.detach().cpu()
            del logits, image, label
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            from monai.networks.utils import one_hot
            pred_oh = one_hot(pred_cpu, num_classes=self.num_classes)
            label_oh = one_hot(label_cpu, num_classes=self.num_classes)
            dice(y_pred=pred_oh, y=label_oh)
            per = dice.aggregate().cpu()
            dice.reset()
            mask = ~torch.isnan(per)
            dice_running[mask] += per[mask]
            dice_count[mask] += 1

            if hd95 is not None:
                hd95(y_pred=pred_oh, y=label_oh)
                per_hd = hd95.aggregate().cpu()
                hd95.reset()
                hd_mask = torch.isfinite(per_hd)
                hd95_running[hd_mask] += per_hd[hd_mask]
                hd95_count[hd_mask] += 1

            running_mean = dice_running / dice_count.clamp(min=1)
            valid_running = running_mean[dice_count > 0]
            postfix = {
                "dice": f"{valid_running.mean().item():.4f}" if valid_running.numel() else "n/a",
                "classes": f"{int((dice_count > 0).sum().item())}/{self.num_classes - 1}",
            }
            if hd95 is not None:
                hd_running_mean = hd95_running / hd95_count.clamp(min=1)
                valid_hd = hd_running_mean[hd95_count > 0]
                postfix["hd95"] = (
                    f"{valid_hd.mean().item():.2f}" if valid_hd.numel() else "n/a"
                )
            if torch.cuda.is_available():
                postfix["gpu_GB"] = f"{torch.cuda.max_memory_allocated() / 1e9:.1f}"
            pbar.set_postfix(postfix, refresh=False)

            del pred_oh, label_oh, pred_cpu, label_cpu
            _release_heap()
        pbar.close()

        dice_per_class = (dice_running / dice_count.clamp(min=1)).tolist()
        valid_dice = [x for x, c in zip(dice_per_class, dice_count.tolist()) if c > 0]
        mean_dice = sum(valid_dice) / max(1, len(valid_dice))

        out = {"mean_dice": mean_dice}
        for name, d in zip(self.classes, dice_per_class):
            out[f"dice/{name}"] = d

        if hd95 is not None:
            hd_per_class = (hd95_running / hd95_count.clamp(min=1)).tolist()
            valid_hd = [x for x, c in zip(hd_per_class, hd95_count.tolist()) if c > 0]
            mean_hd95 = sum(valid_hd) / max(1, len(valid_hd))
            out["mean_hd95"] = mean_hd95
            for name, h, c in zip(self.classes, hd_per_class, hd95_count.tolist()):
                out[f"hd95/{name}"] = h if c > 0 else float("nan")
        return out
