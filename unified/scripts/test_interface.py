"""Phase 5 of the I1/I2 interface — assert the properties the comparison rests on.

I1/I2 only mean something if the decoder every encoder gets is *the same object*.
Under the 5-level contract it never was: a ViT's adapter manufactured four levels a
CNN got for free, so a between-encoder difference mixed representation quality with
adapter capacity. These checks are the guard rail:

  1. The DECODER is byte-identical across encoders — same parameter count regardless
     of how many native grids an encoder exposes or how many doublings it needs.
     This is what the weight-shared up block and shared output conv buy.
  2. ``I2 - I1`` costs the SAME constant for every encoder (one 28,640-param
     ConvStem plus its 8,704-param projection = 37,344), so the arm difference is
     one number per encoder and not a per-encoder design choice.
  3. Output geometry matches the input, and deep supervision emits one map per
     decoder stage, finest first.
  4. Under ``freeze_backbone`` nothing outside the adapter and head trains.

Adapter parameter counts DO differ between encoders — they are a function of each
encoder's own tap widths (ctfm 32..512, dino3d 4x1024) — and that is correct: it is
the encoder's geometry, not a choice we made for it.

Usage:  python -m scripts.test_interface            # ctfm + dino3d, CPU
        python -m scripts.test_interface --models ctfm,vista3d,voco_b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import yaml  # noqa: E402

from unified.models import build_backbone, build_head, SegModel  # noqa: E402

STEM_COST = 28_640 + 8_704      # ConvStem(out_ch=32) + ChannelNeck(32 -> 256)


def count(mod) -> int:
    return sum(p.numel() for p in mod.parameters())


def build(stem_cfg: str, interface: str, patch, width: int, num_classes: int):
    mcfg = yaml.safe_load((REPO / "configs" / "models" / f"{stem_cfg}.yaml").read_text())["model"]
    kwargs = dict(mcfg.get("kwargs", {}))
    kwargs.update(interface=interface, interface_width=width, interface_patch_size=tuple(patch))
    backbone = build_backbone(mcfg["name"], weights=None, **kwargs)
    head = build_head("interface_seg_head", num_classes=num_classes,
                      deep_supervision=True, **backbone.head_kwargs())
    return SegModel(backbone, head, freeze_backbone=True), backbone, head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ctfm,dino3d_layerwise")
    ap.add_argument("--patch", default="96,96,96")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--num-classes", type=int, default=118)
    args = ap.parse_args()

    patch = tuple(int(v) for v in args.patch.split(","))
    stems = [s.strip() for s in args.models.split(",") if s.strip()]
    torch.set_grad_enabled(False)

    rows, decoders, failures = [], {}, []
    for stem in stems:
        per_arm = {}
        for arm in ("i1", "i2"):
            model, backbone, head = build(stem, arm, patch, args.width, args.num_classes)
            x = torch.zeros(1, 1, *patch)

            model.eval()
            out = model(x)
            if out.shape != (1, args.num_classes, *patch):
                failures.append(f"{stem}/{arm}: eval output {tuple(out.shape)} "
                                f"!= {(1, args.num_classes, *patch)}")

            model.train()
            ds = model(x)
            k = backbone.k_upblocks()
            if not isinstance(ds, list) or len(ds) != k:
                failures.append(f"{stem}/{arm}: deep supervision gave "
                                f"{len(ds) if isinstance(ds, list) else type(ds).__name__}, expected {k}")
            elif tuple(ds[0].shape[2:]) != patch:
                failures.append(f"{stem}/{arm}: finest DS map {tuple(ds[0].shape[2:])} != {patch}")

            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            leaked = {n for n in trainable
                      if not (n.startswith("head.") or n.startswith("backbone.adapter."))}
            if leaked:
                failures.append(f"{stem}/{arm}: {len(leaked)} param(s) outside adapter/head "
                                f"train while frozen, e.g. {sorted(leaked)[:2]}")

            per_arm[arm] = {
                "adapter": count(backbone.adapter),
                "decoder": count(head),
                "trainable": model.num_trainable_params(),
                "strides": backbone.output_strides(),
                "k": k,
            }
            decoders.setdefault(count(head), []).append(f"{stem}/{arm}")

        delta = per_arm["i2"]["adapter"] - per_arm["i1"]["adapter"]
        if delta != STEM_COST:
            failures.append(f"{stem}: I2-I1 adapter delta {delta:,} != {STEM_COST:,}")
        rows.append((stem, per_arm, delta))

    print(f"\npatch {patch}   common width W={args.width}\n")
    print(f"{'encoder':<20} {'arm':<4} {'native strides':<22} {'k':>2} "
          f"{'adapter':>10} {'decoder':>10} {'trainable':>11}")
    print("-" * 86)
    for stem, per_arm, delta in rows:
        for arm in ("i1", "i2"):
            a = per_arm[arm]
            print(f"{stem:<20} {arm.upper():<4} {str(a['strides']):<22} {a['k']:>2} "
                  f"{a['adapter']:>10,} {a['decoder']:>10,} {a['trainable']:>11,}")
        print(f"{'':<20} {'I2-I1':<4} {'':<22} {'':>2} {delta:>10,}")

    print("\nchecks")
    if len(decoders) == 1:
        n, who = next(iter(decoders.items()))
        print(f"  [ok] decoder identical across all {len(who)} configs: {n:,} params")
    else:
        failures.append(f"decoder differs across encoders: "
                        + "; ".join(f"{n:,} -> {w}" for n, w in decoders.items()))
    if not failures:
        print("  [ok] I2-I1 is a constant +{:,} for every encoder".format(STEM_COST))
        print("  [ok] output geometry and deep supervision correct")
        print("  [ok] frozen encoder: only adapter + head train")
        print("\nPASS")
        return
    print()
    for f in failures:
        print(f"  [FAIL] {f}")
    print("\nFAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
