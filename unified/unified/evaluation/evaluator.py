"""Sliding-window evaluation with per-class Dice and (optionally) HD95.

Same evaluator across all models. Operates at the spacing/orientation produced
by the val transforms.

Dice is computed on the GPU from a per-case confusion matrix (no dense one-hot).
This is mathematically identical to MONAI's foreground ``DiceMetric`` but ~100x
cheaper: the previous path expanded two 118-channel one-hot volumes on the CPU
(~2 GB each) per case and ran Dice/HD95 over them, which dominated validation
wall-clock and left the GPU idle (the sliding-window forward is ~0.4 s/case on
GPU; the CPU one-hot + Dice + HD95 was 1.4-14 s/case). The confusion matrix is a
118x118 tensor built with a single ``bincount``, so the reduction stays on the
GPU and the loader can prefetch the next volume during the forward pass.

HD95 has no GPU path in MONAI (scipy surface distances), so it stays on the CPU.
But a Hausdorff distance is only defined for classes present in BOTH the ground
truth and the prediction, so we build a thin one-hot over just those classes
rather than all 117 — the both-present set is bounded by the handful of organs
actually in each scan, which is where the bulk of the old HD95 cost went.
"""
from __future__ import annotations
import math
import warnings
from typing import Dict, List

import torch
from tqdm.auto import tqdm

from ..utils import get_logger

# MONAI's Dice/Hausdorff internals warn "the ground truth/prediction of class N
# is all 0, this may result in nan/inf distance" for every class absent on
# either side. We only feed HD95 the both-present classes, but keep the filter
# as a guard. Matched at the start of the message (filters are start-anchored).
_EMPTY_CLASS_WARN = r"the (ground truth|prediction) of class"


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

    def _infer(self, model, image):
        """Sliding-window inference -> integer label map ``(1, 1, D, H, W)``.

        Stays on the input device (GPU). The full ``(1, C, D, H, W)`` logits
        volume is freed as soon as the argmax is taken, so only the small integer
        prediction is carried forward into the metric reductions.
        """
        from monai.inferers import sliding_window_inference
        logits = sliding_window_inference(
            inputs=image, roi_size=self.roi, sw_batch_size=self.sw_batch,
            predictor=model, overlap=self.overlap, mode=self.mode,
        )
        pred = logits.argmax(dim=1, keepdim=True)
        del logits
        return pred

    def _confusion(self, pred, label):
        """Per-case ``(C, C)`` confusion matrix on the prediction's device.

        Row = ground-truth class, column = predicted class. Built with a single
        ``bincount`` over the flattened ``label * C + pred`` — no dense one-hot.
        """
        C = self.num_classes
        lab = label.reshape(-1).to(torch.int64)
        prd = pred.reshape(-1).to(torch.int64)
        cm = torch.bincount(lab * C + prd, minlength=C * C)
        return cm.reshape(C, C)

    @staticmethod
    def _dice_from_confusion(cm):
        """Foreground per-class Dice plus gt-present / pred-present masks.

        ``Dice_c = 2*TP_c / (|gt_c| + |pred_c|)`` where ``TP_c`` is the matrix
        diagonal, ``|gt_c|`` the row sum and ``|pred_c|`` the column sum. This is
        exactly MONAI's foreground ``DiceMetric`` (``include_background=False``):
        a gt-present-but-missed class keeps its legitimate 0; a class absent from
        the gt is excluded by the returned ``gt_present`` mask, matching CT-FM's
        ``ignore_empty=True``. Background (index 0) is dropped.
        """
        cm = cm.double()
        tp = torch.diag(cm)
        gt = cm.sum(dim=1)                       # |gt_c|
        pr = cm.sum(dim=0)                        # |pred_c|
        dice = (2.0 * tp) / (gt + pr).clamp(min=1.0)
        return dice[1:], (gt[1:] > 0), (pr[1:] > 0)

    def _hd95_present(self, pred, label, both_fg):
        """Per-class HD95 over classes present in BOTH gt and prediction.

        ``both_fg`` is the foreground both-present mask (length ``num_classes-1``,
        0-based). Builds a thin one-hot on the CPU with one channel per such class
        and runs MONAI's ``HausdorffDistanceMetric`` over it. Returns
        ``{foreground_index: hd95}`` for finite distances only. HD95 needs two
        non-empty surfaces, so restricting to both-present classes is both correct
        (MONAI would otherwise return a spurious 0/NaN) and cheap — scipy never
        runs a distance transform for a class we would mask away.
        """
        from monai.metrics import HausdorffDistanceMetric
        present = torch.nonzero(both_fg, as_tuple=False).flatten().tolist()
        if not present:
            return {}
        global_ids = [j + 1 for j in present]    # 0-based fg index -> class id
        p = pred[0, 0].cpu()
        l = label[0, 0].cpu()
        pred_oh = torch.stack([(p == c) for c in global_ids]).unsqueeze(0).float()
        label_oh = torch.stack([(l == c) for c in global_ids]).unsqueeze(0).float()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_EMPTY_CLASS_WARN,
                                    category=UserWarning)
            hm = HausdorffDistanceMetric(
                include_background=True, distance_metric="euclidean",
                percentile=95, reduction="mean_batch", get_not_nans=False,
            )
            hm(y_pred=pred_oh, y=label_oh)
            per = hm.aggregate().cpu()
        return {j: v for j, v in zip(present, per.tolist()) if math.isfinite(v)}

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

        d_run, d_cnt = torch.zeros(nc), torch.zeros(nc)
        h_run, h_cnt = torch.zeros(nc), torch.zeros(nc)

        try:
            total = len(loader)
        except TypeError:
            total = None
        pbar = tqdm(loader, total=total, desc=f"[val] roi={tuple(self.roi)}",
                    dynamic_ncols=True, leave=False)

        for batch in pbar:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            pred = self._infer(model, image)
            cm = self._confusion(pred, label)
            dice_fg, gt_present, pred_present = self._dice_from_confusion(cm)

            dpc = dice_fg.cpu()
            gpm = gt_present.cpu()
            d_run[gpm] += dpc[gpm]
            d_cnt[gpm] += 1

            if self.want_hd95:
                both = (gt_present & pred_present).cpu()
                for j, v in self._hd95_present(pred, label, both).items():
                    h_run[j] += v
                    h_cnt[j] += 1

            del image, label, pred, cm

            fv = (d_run / d_cnt.clamp(min=1))[d_cnt > 0]
            postfix = {
                "dice": f"{fv.mean().item():.4f}" if fv.numel() else "n/a",
                "classes": f"{int((d_cnt > 0).sum().item())}/{nc}",
            }
            if torch.cuda.is_available():
                postfix["gpu_GB"] = f"{torch.cuda.max_memory_allocated() / 1e9:.1f}"
            pbar.set_postfix(postfix, refresh=False)
        pbar.close()

        out: Dict = {}
        out["mean_dice"] = self._reduce_per_class(d_run, d_cnt, "dice", out)
        if self.want_hd95:
            out["mean_hd95"] = self._reduce_per_class(h_run, h_cnt, "hd95", out)
        return out
