"""Measure peak training VRAM for ONE (model config, batch_size) in a fresh
process, replicating the real train step exactly.

Mirrors unified.training.Trainer: deep-supervision forward (training mode),
bf16 autocast (or fp32 when model.amp=false), DiceCE(+DS) loss, AdamW step with
grad-clip. Inputs are synthetic patches of the real per-step shape
``(batch_size * num_samples_per_volume, 1, P, P, P)`` — peak VRAM is determined
by tensor shapes, not values, so this is accurate and skips the data pipeline.

Pretrained weights are NOT loaded (``weights=None``): parameter *count* — and
thus memory — is identical to the pretrained model, and loading is slow.

Prints one machine-parseable line to stdout:
    RESULT {"model": ..., "bs": B, "eff": E, "peak_gb": .., "reserved_gb": .., "step_s": ..}
on OOM:    RESULT {"model": ..., "bs": B, "status": "oom"}      (exit 3)
on error:  RESULT {"model": ..., "bs": B, "status": "error", "msg": ".."} (exit 2)
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
from unified.models import build_backbone, build_head, SegModel  # noqa: E402
from unified.training import build_loss, build_optimizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--patch", type=int, default=None,
                    help="override patch size (cube); default = cfg data.patch_size")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["model"]["name"]
    bs = args.batch_size
    P = args.patch if args.patch is not None else int(cfg["data"]["patch_size"][0])
    num_samples = int(cfg["data"].get("num_samples_per_volume", 1))
    eff = bs * num_samples
    nc = int(cfg["data"]["num_classes"])

    def emit(d, code):
        print("RESULT " + json.dumps({"model": name, "bs": bs, "eff": eff,
                                      "patch": P, **d}), flush=True)
        sys.exit(code)

    if not torch.cuda.is_available():
        emit({"status": "error", "msg": "no cuda"}, 2)

    device = torch.device("cuda")
    try:
        mcfg = cfg["model"]
        backbone = build_backbone(mcfg["name"], weights=None, **mcfg.get("kwargs", {}))
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
        model.train()

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = build_optimizer(cfg, trainable)
        loss_fn = build_loss(cfg)

        # AMP exactly as the trainer resolves it.
        model_amp = cfg["model"].get("amp")
        amp = bool(model_amp if model_amp is not None else cfg["train"].get("amp", True))
        amp_dtype, scaler = None, None
        if amp:
            use_bf16 = torch.cuda.is_bf16_supported()
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
            if amp_dtype is torch.float16:
                scaler = torch.cuda.amp.GradScaler()
        grad_clip = cfg["train"].get("grad_clip", 0.0)

        image = torch.randn(eff, 1, P, P, P, device=device)
        label = torch.randint(0, nc, (eff, 1, P, P, P), device=device, dtype=torch.long)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            if amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(image)
                    loss = loss_fn(logits, label)
            else:
                logits = model(image)
                loss = loss_fn(logits, label)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        torch.cuda.synchronize()
        step_s = (time.time() - t0) / max(1, args.steps)

        peak = torch.cuda.max_memory_allocated() / 1e9
        resv = torch.cuda.max_memory_reserved() / 1e9
        amp_desc = {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(amp_dtype, "fp32")
        emit({"status": "ok", "peak_gb": round(peak, 2), "reserved_gb": round(resv, 2),
              "step_s": round(step_s, 3), "amp": amp_desc,
              "trainable_M": round(model.num_trainable_params() / 1e6, 1),
              "total_M": round(model.num_total_params() / 1e6, 1),
              "frozen": bool(model.freeze_backbone),
              "loss": round(float(loss.item()), 4)}, 0)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        emit({"status": "oom"}, 3)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        # An OOM sometimes surfaces as a generic RuntimeError.
        if "out of memory" in str(e).lower():
            emit({"status": "oom"}, 3)
        traceback.print_exc()
        emit({"status": "error", "msg": msg[:200]}, 2)


if __name__ == "__main__":
    main()
