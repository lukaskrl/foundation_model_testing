"""Generate the frozen / low-shot / pretrained-vs-scratch benchmark matrix.

The decisive experiment for the "recipe dominates representation" story: hold
the decoder + recipe FIXED and vary only

    encoder state : {frozen, finetune}
    init          : {pretrained, scratch}
    data fraction : {1, 5, 10, 25, 100}%
    backbone      : the 11 CT foundation encoders

Plus the best-effort arms in ``EXTRA_ARMS`` (currently dino3d + 3D ViT-Adapter),
which share the recipe and batch but are flagged ``probe_comparable=False`` in
MANIFEST.csv because their frozen condition trains more than a thin neck. They
belong in a separate table, never merged into the headline probe column.

Each emitted config reuses the backbone's own ``model:`` block from
``configs/models/<backbone>.yaml`` (so the ViT probe-honest ``pyramid_mode:
upsample`` necks carry through unchanged) and adds only the four experimental
knobs. Recipe (train:) stays locked in base.yaml; head/eval untouched. Batch is
made UNIFORM across the whole matrix (bs=2, accum=2 -> optimizer batch 8) so
frozen-vs-finetune and pretrained-vs-scratch carry no batch-size confound — the
existing headline sweep (each model at its own batch) stays a SEPARATE table.

Usage:
    python -m scripts.gen_lowshot_configs               # write all configs + MANIFEST.csv
    python -m scripts.gen_lowshot_configs --dry-run     # just print the plan

Outputs: configs/lowshot/<run_name>.yaml  +  configs/lowshot/MANIFEST.csv
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "configs" / "models"
OUT_DIR = REPO / "configs" / "lowshot"

# The 11 encoders from the headline sweep (config stem -> short tag for names).
#
# dino3d deliberately reads `dino3d_upsample.yaml`, NOT `dino3d.yaml`: the
# latter's default is `pyramid_mode: vit_adapter`, whose Injector/Extractor
# interaction blocks live inside the frozen encoder and leave ~25.9 M params
# trainable under freeze_backbone (vs 0.3-1.2 M of pure neck for every other
# encoder). That is a fine best-effort finetune arm but it is not a probe, and
# dropping it into this matrix would make the frz_pt column measure a different
# quantity for dino3d than for everyone else. Keep the probe-honest `upsample`
# neck here; the vit_adapter arm is generated separately via EXTRA_ARMS below.
BACKBONES = {
    "ctfm": "ctfm",
    "vista3d": "vista3d",
    "biomedparse": "biomedparse",
    "suprem_unet": "supUnet",
    "suprem_segresnet": "supSegres",
    "suprem_swinunetr": "supSwin",
    "voco_b": "vocoB",
    "voco_h": "vocoH",
    "ctclip": "ctclip",
    "sam_med3d": "samMed3d",
    "dino3d_upsample": "dino3d",
}

# Best-effort arms that share the matrix's recipe and batch but are NOT
# cross-backbone probe-comparable, so they live under their own run tags and are
# flagged `probe_comparable=False` in MANIFEST.csv. They form a self-contained
# parallel table: all four conditions inside an arm use one adapter, so
# frz-vs-ft and pretrained-vs-scratch remain meaningful *within* the arm.
#
# `dino3d_vitadapter` is the faithful plain-ViT dense-prediction recipe (3DINO's
# 3D ViT-Adapter: SpatialPriorModule bidirectionally fused with the ViT through
# deformable Injector/Extractor blocks), so the pretrained representation feeds
# every contract scale instead of only stride-16. Report it beside the dino3d
# `upsample` row, never merged into the headline probe column.
EXTRA_ARMS = {
    "dino3d_vitadapter": "dino3dVA",
}

# ARM S — the matched-stem control.
#
# Every entry here is the SAME encoder as its BACKBONES twin with one field
# changed: `stem_fusion: true`, which wraps it in StemFusionBackbone so a shared
# SpatialPriorModule3D is summed into contract levels 0-3. Because the wrapper is
# one generic code path, every Arm S model gains a bit-identical 349,584
# parameters — that constant is what makes the arm a controlled comparison rather
# than a per-backbone tuning exercise.
#
# Arm S is NOT probe-comparable against Arm N: it adds a raw-input branch, so a
# frozen Arm S run measures more than a thin neck. It IS comparable *within* the
# arm, which is the whole point. The headline quantity is the per-model delta
#
#     compensable(m) = gap_N(m) - gap_S(m)
#
# so both arms must exist for the same (condition, fraction) cell to subtract.
# Read Arm S rows via the `arm` column, never merged into the Arm N column.
#
# Pilot scope: ctfm (hierarchical CNN, 1:1 native pyramid) and dino3d (columnar
# ViT, nothing below stride 16) — the maximal-contrast pair, so the delta should
# be largest here. Extend to the other nine once the pilot reads sensibly.
ARM_S = {
    "ctfm_stem": "ctfmS",
    "dino3d_stem": "dino3dS",
}

# A frozen run is a clean representation probe only if every trainable parameter
# sits in a thin neck. These pyramid modes break that: `vit_adapter` trains the
# interaction blocks, `spm` adds a fresh conv branch on the raw input volume.
NON_PROBE_MODES = {"vit_adapter", "spm"}

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.00]
# (freeze_backbone, pretrained, tag)
CONDITIONS = [
    (True, True, "frz_pt"),    # frozen probe of the pretrained representation (headline)
    (True, False, "frz_sc"),   # frozen random-feature floor (control)
    (False, True, "ft_pt"),    # full finetune from pretrained
    (False, False, "ft_sc"),   # full finetune from scratch (the convergence control)
]

# Uniform batch across the matrix to remove the batch-size confound.
UNIFORM_BATCH_SIZE = 2
UNIFORM_GRAD_ACCUM = 2


def _frac_tag(f: float) -> str:
    return f"f{int(round(f * 100)):03d}"


def _load_model_block(stem: str) -> dict:
    cfg = yaml.safe_load((MODELS_DIR / f"{stem}.yaml").read_text())
    if "model" not in cfg:
        raise SystemExit(f"{stem}.yaml has no model: block")
    return cfg["model"]


def build(dry_run: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    arm_of = {
        **{s: "N" for s in BACKBONES},
        **{s: "S" for s in ARM_S},
        **{s: "B" for s in EXTRA_ARMS},
    }
    for stem, tag in {**BACKBONES, **ARM_S, **EXTRA_ARMS}.items():
        base_model = _load_model_block(stem)
        kw = base_model.get("kwargs", {})
        # CNN encoders have a native pyramid and no pyramid_mode knob at all.
        adapter = kw.get("pyramid_mode", "native")
        # Arm S keeps its inner pyramid mode and adds the shared stem on top, so
        # name both — "upsample+stem" is a different adapter from "upsample", and
        # collapsing them would hide the arm inside the adapter column.
        stem_fused = bool(kw.get("stem_fusion", False))
        if stem_fused:
            adapter = f"{adapter}+stem"
        for freeze, pretrained, cond in CONDITIONS:
            for frac in FRACTIONS:
                run = f"ls_{tag}_{cond}_{_frac_tag(frac)}"
                model = dict(base_model)
                model["batch_size"] = UNIFORM_BATCH_SIZE
                model["grad_accum_steps"] = UNIFORM_GRAD_ACCUM
                model["freeze_backbone"] = freeze
                model["pretrained"] = pretrained
                doc = {"model": model, "data": {"train_fraction": float(frac)}}
                rows.append({
                    "run_name": run,
                    "config": f"configs/lowshot/{run}.yaml",
                    "backbone": stem,
                    "freeze_backbone": freeze,
                    "pretrained": pretrained,
                    "train_fraction": frac,
                    "condition": cond,
                    # New columns are APPENDED — run_lowshot_matrix.sh selects
                    # positionally ($3 backbone, $6 fraction, $7 condition).
                    "adapter": adapter,
                    # A fused stem is a raw-input conv branch, so an Arm S frozen
                    # run trains more than a thin neck — same disqualification the
                    # `spm` / `vit_adapter` modes carry.
                    "probe_comparable": (adapter not in NON_PROBE_MODES
                                         and not stem_fused),
                    "arm": arm_of[stem],
                })
                if not dry_run:
                    header = ("# AUTO-GENERATED by scripts/gen_lowshot_configs.py — "
                              "do not edit by hand.\n")
                    (OUT_DIR / f"{run}.yaml").write_text(
                        header + yaml.safe_dump(doc, sort_keys=False))

    if not dry_run:
        with (OUT_DIR / "MANIFEST.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    n = len(rows)
    per_arm = len(CONDITIONS) * len(FRACTIONS)
    print(f"{'PLANNED' if dry_run else 'WROTE'} {n} configs "
          f"({len(BACKBONES)} backbones + {len(ARM_S)} arm-S + {len(EXTRA_ARMS)} "
          f"extra arms, x {len(CONDITIONS)} conditions x {len(FRACTIONS)} fractions)")
    print(f"  conditions : {[c for *_, c in CONDITIONS]}")
    print(f"  fractions  : {[f'{int(f*100)}%' for f in FRACTIONS]}")
    print(f"  arm N      : {len(BACKBONES) * per_arm} probe-comparable configs")
    for stem, tag in ARM_S.items():
        print(f"  arm S      : {per_arm} configs  ls_{tag}_*  <- {stem} "
              f"(matched stem; subtract against its arm-N twin)")
    for stem, tag in EXTRA_ARMS.items():
        print(f"  arm B      : {per_arm} configs  ls_{tag}_*  <- {stem} "
              f"(probe_comparable=False; report separately)")
    if not dry_run:
        print(f"  -> {OUT_DIR}  (+ MANIFEST.csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build(dry_run=args.dry_run)
