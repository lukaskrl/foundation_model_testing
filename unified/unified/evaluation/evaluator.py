"""Sliding-window evaluation with per-class Dice and (optionally) HD95.

Same evaluator across all models. Operates at the spacing/orientation produced
by the val transforms.
"""
from __future__ import annotations
import ctypes
import gc
import warnings
from typing import Dict, List

import torch
from tqdm.auto import tqdm

from ..utils import get_logger

try:
    _libc_malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
    _libc_malloc_trim.argtypes = [ctypes.c_size_t]
except OSError:
    _libc_malloc_trim = None

# MONAI's Dice/Hausdorff internals warn "the ground truth/prediction of class N
# is all 0, this may result in nan/inf distance" for every class absent on
# either side. We handle those explicitly (gt-absent excluded from Dice;
# either-absent excluded from HD95), so the warnings are pure noise. Matched at
# the start of the message (warnings filters are start-anchored regexes).
_EMPTY_CLASS_WARN = r"the (ground truth|prediction) of class"


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
        # CT-FM-style patch validation: IN ADDITION to the full-volume metrics
        # above, score one random crop per case (RandSpatialCrop -> SpatialPad)
        # with the same sliding-window inferer, and report the per-crop macro
        # Dice/HD95 over present foreground classes, averaged across cases —
        # mirroring CT-FM's Macro_Dice so that headline is directly comparable.
        # crop_size is in the model's fed orientation (SPL for ct-fm), matching
        # CT-FM's val_max_patch_size [192, 240, 240]. The crop is re-randomised
        # every val round (matching CT-FM's RandSpatialCropd), so the patch
        # curve carries crop-placement noise by design.
        pv = cfg["eval"].get("patch_validation", {}) or {}
        self.patch_enabled = bool(pv.get("enabled", False))
        self.patch_crop_size = tuple(pv.get("crop_size", (192, 240, 240)))
        self._patch_crop = None
        if self.patch_enabled:
            from monai.transforms import Compose, RandSpatialCropd, SpatialPadd
            self._patch_crop = Compose([
                RandSpatialCropd(keys=("image", "label"),
                                 roi_size=self.patch_crop_size, random_size=False),
                SpatialPadd(keys=("image", "label"),
                            spatial_size=self.patch_crop_size, mode="constant"),
            ])

    def _new_metrics(self):
        """Fresh (DiceMetric, HausdorffDistanceMetric|None) pair, reduction per-class."""
        from monai.metrics import DiceMetric, HausdorffDistanceMetric
        dice = DiceMetric(include_background=False, reduction="mean_batch",
                          get_not_nans=False)
        hd95 = (
            HausdorffDistanceMetric(
                include_background=False, distance_metric="euclidean",
                percentile=95, reduction="mean_batch", get_not_nans=False,
            )
            if self.want_hd95 else None
        )
        return dice, hd95

    def _infer_one_hot(self, model, image, label):
        """Sliding-window inference -> (pred_one_hot, label_one_hot) on CPU.

        argmax + one-hot are done on CPU: for 117-class CT volumes each
        (1, C, D, H, W) one-hot is ~6 GiB, and keeping it on GPU OOMs the larger
        backbones. Inputs are left intact for the caller (the patch pass reuses
        the loaded volume).
        """
        from monai.inferers import sliding_window_inference
        from monai.networks.utils import one_hot
        logits = sliding_window_inference(
            inputs=image, roi_size=self.roi, sw_batch_size=self.sw_batch,
            predictor=model, overlap=self.overlap, mode=self.mode,
        )
        pred_cpu = logits.argmax(dim=1, keepdim=True).cpu()
        label_cpu = label.detach().cpu()
        del logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pred_oh = one_hot(pred_cpu, num_classes=self.num_classes)
        label_oh = one_hot(label_cpu, num_classes=self.num_classes)
        del pred_cpu, label_cpu
        return pred_oh, label_oh

    @staticmethod
    def _update(pred_oh, label_oh, dice, hd95,
                dice_running, dice_count, hd95_running, hd95_count):
        """Fold one case's per-class Dice/HD95 into the running accumulators.

        Returns this case's macro Dice and macro HD95. Macro Dice is the mean
        over foreground classes present in the ground truth (NaN-excluded) —
        exactly CT-FM's Macro_Dice reduction (DiceHelper reduction='mean',
        ignore_empty=True) on a single sample; a class predicted but absent in
        gt is excluded, a class in gt but missed scores 0. Macro HD95 is the
        mean over foreground classes present in BOTH gt and prediction (a
        distance needs two non-empty surfaces).
        """
        # Foreground-class presence, taken directly from the one-hot maps.
        # We must NOT rely on MONAI's NaN-for-empty-gt convention here: with
        # reduction="mean_batch" the aggregate collapses an all-NaN class to
        # 0.0, so ``~isnan(per)`` would wrongly keep gt-absent classes as a 0
        # Dice and deflate the macro (averaging over all 117 classes instead of
        # the handful present per case). Deriving presence from the label is
        # exact and matches CT-FM's ignore_empty=True (gt-absent excluded).
        sp = [0] + list(range(2, label_oh.ndim))          # batch + spatial dims
        gt_present = label_oh.sum(dim=sp)[1:] > 0          # (nc,) foreground in gt

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_EMPTY_CLASS_WARN,
                                    category=UserWarning)
            dice(y_pred=pred_oh, y=label_oh)
        per = dice.aggregate().cpu()
        dice.reset()
        # Score every gt-present class — a gt-present but missed organ keeps its
        # legitimate 0 Dice; gt-absent classes are excluded from the macro.
        dice_running[gt_present] += per[gt_present]
        dice_count[gt_present] += 1
        case_dice = per[gt_present].mean().item() if bool(gt_present.any()) else float("nan")

        case_hd = float("nan")
        if hd95 is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=_EMPTY_CLASS_WARN,
                                        category=UserWarning)
                hd95(y_pred=pred_oh, y=label_oh)
            per_hd = hd95.aggregate().cpu()
            hd95.reset()
            # HD95 needs two non-empty surfaces, so it is only defined for
            # classes present in BOTH gt and prediction. MONAI otherwise returns
            # a misleading 0.0 (pred empty) or NaN (gt empty); excluding
            # either-missing classes avoids a missed organ logging a fake 0 mm.
            pred_present = pred_oh.sum(dim=sp)[1:] > 0
            hd_mask = gt_present & pred_present & torch.isfinite(per_hd)
            hd95_running[hd_mask] += per_hd[hd_mask]
            hd95_count[hd_mask] += 1
            case_hd = per_hd[hd_mask].mean().item() if bool(hd_mask.any()) else float("nan")
        return case_dice, case_hd

    def _reduce_per_class(self, running, count, prefix, out):
        """Write per-class series (mean over cases where the class was present)
        and return the macro mean over classes that appeared at least once."""
        per_class = (running / count.clamp(min=1)).tolist()
        counts = count.tolist()
        for name, v, c in zip(self.classes, per_class, counts):
            # A class with count 0 never qualified (Dice: never in any gt;
            # HD95: never in both gt and pred). Report NaN so the trainer drops
            # it rather than logging a fake 0 that would skew per-organ panels.
            out[f"{prefix}/{name}"] = v if c > 0 else float("nan")
        valid = [v for v, c in zip(per_class, counts) if c > 0]
        return sum(valid) / max(1, len(valid))

    @torch.no_grad()
    def evaluate(self, model, loader, device) -> Dict:
        model.eval()
        nc = self.num_classes - 1

        # Full-volume accumulators.
        f_dice, f_hd95 = self._new_metrics()
        f_d_run, f_d_cnt = torch.zeros(nc), torch.zeros(nc)
        f_h_run, f_h_cnt = torch.zeros(nc), torch.zeros(nc)
        # CT-FM-style patch accumulators (+ per-crop macro lists for the headline).
        p_dice, p_hd95 = self._new_metrics()
        p_d_run, p_d_cnt = torch.zeros(nc), torch.zeros(nc)
        p_h_run, p_h_cnt = torch.zeros(nc), torch.zeros(nc)
        p_case_dice: List[float] = []
        p_case_hd: List[float] = []

        try:
            total = len(loader)
        except TypeError:
            total = None
        desc = f"[val] roi={tuple(self.roi)}"
        if self.patch_enabled:
            desc += f" +patch{tuple(self.patch_crop_size)}"
        pbar = tqdm(loader, total=total, desc=desc, dynamic_ncols=True, leave=False)

        for batch in pbar:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            # --- full volume ---
            pred_oh, label_oh = self._infer_one_hot(model, image, label)
            self._update(pred_oh, label_oh, f_dice, f_hd95,
                         f_d_run, f_d_cnt, f_h_run, f_h_cnt)
            del pred_oh, label_oh
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _release_heap()

            # --- CT-FM-style single random crop (re-randomised each case) ---
            if self.patch_enabled:
                cropped = self._patch_crop({"image": image[0], "label": label[0]})
                crop_img = cropped["image"].unsqueeze(0)
                crop_lbl = cropped["label"].unsqueeze(0)
                pred_oh, label_oh = self._infer_one_hot(model, crop_img, crop_lbl)
                cd, ch = self._update(pred_oh, label_oh, p_dice, p_hd95,
                                      p_d_run, p_d_cnt, p_h_run, p_h_cnt)
                p_case_dice.append(cd)
                p_case_hd.append(ch)
                del pred_oh, label_oh, crop_img, crop_lbl, cropped
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _release_heap()

            del image, label

            fv = (f_d_run / f_d_cnt.clamp(min=1))[f_d_cnt > 0]
            postfix = {
                "dice": f"{fv.mean().item():.4f}" if fv.numel() else "n/a",
                "classes": f"{int((f_d_cnt > 0).sum().item())}/{nc}",
            }
            if self.patch_enabled:
                pv = [x for x in p_case_dice if x == x]  # drop NaN
                postfix["patch_dice"] = f"{sum(pv) / len(pv):.4f}" if pv else "n/a"
            if torch.cuda.is_available():
                postfix["gpu_GB"] = f"{torch.cuda.max_memory_allocated() / 1e9:.1f}"
            pbar.set_postfix(postfix, refresh=False)
        pbar.close()

        # --- full-volume summary: per-class macro over classes that appeared ---
        out: Dict = {}
        out["mean_dice"] = self._reduce_per_class(f_d_run, f_d_cnt, "dice", out)
        if f_hd95 is not None:
            out["mean_hd95"] = self._reduce_per_class(f_h_run, f_h_cnt, "hd95", out)

        # --- patch summary: headline = mean over crops of the per-crop macro
        #     (CT-FM Macro_Dice); plus the per-organ series on the crops. ---
        if self.patch_enabled:
            dvals = [x for x in p_case_dice if x == x]
            out["patch_mean_dice"] = sum(dvals) / max(1, len(dvals))
            self._reduce_per_class(p_d_run, p_d_cnt, "patch_dice", out)
            if p_hd95 is not None:
                hvals = [x for x in p_case_hd if x == x]
                out["patch_mean_hd95"] = sum(hvals) / max(1, len(hvals))
                self._reduce_per_class(p_h_run, p_h_cnt, "patch_hd95", out)
        return out
