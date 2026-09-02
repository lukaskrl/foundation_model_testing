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

Arms, all tagged in the ``arm`` column of MANIFEST.csv:

    N  the encoder as delivered (+ the ``ABLATIONS`` neck variants)
    S  ``ARM_S``      — N plus a bit-identical fused conv stem  (matched CAPACITY)
    B  ``EXTRA_ARMS`` — best-effort dense recipe (3D ViT-Adapter)
    W  ``ARM_W``      — N on a bit-identical forced HU window   (matched INPUT)

Arm W exists because the per-encoder intensity window is the one remaining
uncontrolled input difference, and it is not a small one: the SuPreM / VoCo
soft-tissue window clips 27.6 % of foreground voxels on this task and flattens
the lungs entirely. See the ``ARM_W_WINDOWS`` comment for the measurement, the
clamp-vs-affine argument for why the prediction differs between the frozen and
finetuned conditions, and the cross-over that identifies which mechanism is at
work. Only Arm N rows carry ``probe_comparable=True``; S, B and W each answer a
different question and must never be merged into the headline column.

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
import copy
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
    "dino3d_layerwise": "dino3d",
    # Second report-supervised encoder, and the only hierarchical one. With
    # ctclip alone, "report-supervised" was confounded with "columnar ViT", so no
    # claim about the pretraining PARADIGM was possible from n=1.
    "merlin": "merlin",
}

# Within-arm ablations: same arm letter and same probe-honesty as their BACKBONES
# twin, isolating ONE neck choice. Not extra encoders — they carry their own run
# tag so the headline column stays one row per model.
#
# `dino3d_upsample` is the single-block neck the matrix used to run: all five
# contract levels were projections of block 23, with blocks 5/11/17 computed and
# discarded. Arm N now reads all four (dino3d_layerwise), so this row measures
# what the 1-of-24 choice cost. Parameter-matched to its twin (1,017,792 adapter
# params either way) and the coarsest level reads the same block-23 tensor in
# both, so the pair differs only in which tensor each level reads.
#
# Only a couple of cells need running (frz_pt at f010 and f100 is enough to size
# the effect) — the full 20 are generated for consistency, not because they should
# all be queued.
# `merlin_isotropic` is the same tower and the same 722,784-param adapter as the
# Arm N `merlin` row, differing only in WHERE the factor-2 depth reduction happens
# (input before the encoder vs each level after it). Merlin was pretrained at
# 1.5 x 1.5 x 3.0 mm while this framework locks 1.5 mm isotropic, so this row
# measures what respecting its pretrained voxel geometry is worth. Pooling has no
# parameters, so the pair is exactly parameter-matched. Like dino3dU, only a couple
# of cells need running (frz_pt at f010 and f100 sizes the effect); the full 20 are
# generated for consistency, not because they should all be queued.
ABLATIONS = {
    "dino3d_upsample": "dino3dU",
    "merlin_isotropic": "merlinISO",
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
# Scope: EVERY Arm N encoder. It began as a two-model pilot (ctfm + dino3d, the
# maximal-contrast pair) but a pilot cannot support the claim the arm is for.
# `compensable` is meant to separate "missing fine detail, a stem fixes it" from
# "missing pretrained mid-level features" AS A PROPERTY OF AN ARCHITECTURE FAMILY,
# and with n=1 per family there is no way to tell "columnar ViTs need a stem" from
# "dino3d needs a stem". Covering all twelve gives every family at least two
# members:
#
#     hierarchical CNN     ctfm, vista3d, suprem_unet, suprem_segresnet
#     hierarchical Swin    voco_b, voco_h, suprem_swinunetr
#     inflated-2D CNN      merlin
#     2D per-slice         biomedparse
#     columnar bottleneck  ctclip, sam_med3d
#     columnar ViT         dino3d
#
# It also closes the gap that mattered most: ctclip and sam_med3d run `upsample`,
# so their stride-1 skip is a trilinear resample of a 10x24x24 (resp. 8^3) grid and
# the shared head has no raw-input path of its own (`enc0` was deliberately
# removed). Their Arm N score is bounded by interpolation rather than by
# representation quality — exactly the handicap this arm exists to quantify — yet
# they were the two encoders it did not cover.
#
# Generated from the Arm N config + {stem_fusion, stem_inplanes} rather than from
# hand-written `<name>_stem.yaml` files, for the same reason the wrapper is one
# generic module: every Arm S row must differ from its Arm N twin by EXACTLY the
# shared stem and nothing else. Twelve hand-maintained twins would be twelve
# chances to drift. (`configs/models/ctfm_stem.yaml` and `dino3d_stem.yaml` are
# kept for standalone one-off runs; they are no longer the matrix's source, and
# the generated configs reproduce them byte-for-byte.)
ARM_S_INPLANES = 16

# What each encoder's Arm N FINE levels already are, before Arm S adds its stem.
# This is the interpretation key for `compensable`, and without it the delta is
# read wrong for five of the twelve encoders.
#
#   pretrained            strides 1-8 are genuine pretrained features. Arm S is a
#                         fresh stem ALONGSIDE a pretrained one, so the honest
#                         contrast is pretrained-stem vs +fresh-stem.
#   fresh_stem_s0(s1)     the encoder cannot reach stride 1 (Swin patch-embed,
#                         inflated-2D conv1) or strides 1-2 (per-slice FocalNet),
#                         so Arm N ALREADY contains a from-scratch conv stem on
#                         raw voxels. Arm S is then its SECOND fresh stem, and
#                         `compensable ~ 0` means "detail was already routed",
#                         NOT "this encoder routes detail well".
#   resampled_*           no raw-input path at all: every fine level is a
#                         trilinear resample of one coarse tensor. Arm S supplies
#                         a wire that did not exist. This is the headline case,
#                         and the largest delta should live here.
#   spm_fused             Arm B only — the SPM is already bidirectionally fused
#                         into every level, which is why the wrapper refuses it.
#
# Verified against the code, not asserted: see the stem-parameter audit in the
# Arm S smoke test (voco/suprem_swinunetr 28,640 · merlin 28,640 · biomedparse
# 84,992 · every `pretrained` and `resampled_*` encoder 0).
ARM_N_FINE_SOURCE = {
    "ctfm": "pretrained",
    "vista3d": "pretrained",
    "suprem_unet": "pretrained",
    "suprem_segresnet": "pretrained",
    "voco_b": "fresh_stem_s0",
    "voco_h": "fresh_stem_s0",
    "suprem_swinunetr": "fresh_stem_s0",
    "merlin": "fresh_stem_s0",
    "merlin_isotropic": "fresh_stem_s0",
    "biomedparse": "fresh_stem_s0s1",
    "ctclip": "resampled_bottleneck",
    "sam_med3d": "resampled_bottleneck",
    "dino3d_layerwise": "resampled_tokens",
    "dino3d_upsample": "resampled_tokens",
    "dino3d_vitadapter": "spm_fused",
}

# ARM W — the matched-INPUT control.
#
# Every other arm feeds each encoder the intensity normalization it was
# pretrained with (`model.preprocessing.intensity`), on the principle that the
# window is part of the encoder's input interface rather than a tunable knob.
# That principle is right, but it is not free, and the cost is not uniform:
#
#   measured on 12 TotalSeg subjects (raw HU, post-Spacing/CropForeground),
#   fraction of each class's voxels CLIPPED by the encoder's own window
#
#     window                    classes >90% clipped   >50%    fg voxels clipped
#     suprem/voco  [-175, 250]        6 / 116            39           27.6%
#     merlin/bmp   [-1000, 1000]      0 / 116             0            0.7%
#     ctfm         [-1024, 2048]      0 / 116             0            0.1%
#
# So five of the thirteen Arm N encoders (voco_b, voco_h, suprem_unet,
# suprem_segresnet, suprem_swinunetr) see a whole-body task through a
# soft-tissue window that flattens the lungs entirely (97-99% of lung/trachea
# voxels) and a third of the skeleton. No representation can recover what was
# clipped before the first conv, so a cross-encoder ranking read off Arm N alone
# cannot separate "worse representation" from "narrower input window".
#
# Arm W separates them. A linear window is `clamp -> affine`; the affine half is
# undone by any layer that can learn, the clamp half is permanent. That makes the
# prediction ARM-DEPENDENT, which is exactly why this is worth running:
#
#   ft_*   the encoder adapts, so the affine shift costs ~nothing and any gap
#          that remains is the clipped information. The cleanest cross-encoder
#          number available.
#   frz_*  the frozen first layers CANNOT absorb the affine shift, so a gap here
#          mixes lost information with distribution mismatch.
#
# Two forced windows give the full 2x2 on every encoder rather than a one-way
# test, and the cross-over is what identifies the mechanism:
#
#   if narrowing costs ctfm about what broadening gains voco  -> INFORMATION,
#      and the Arm N ranking is confounded; report both columns.
#   if broadening HURTS voco while narrowing hurts ctfm       -> input fidelity
#      dominates, the native window really is the right interface, Arm N stands.
#
# Like Arm S, this is one shared code path applied uniformly rather than a
# per-backbone edit — that is what makes every Arm W row see a BIT-IDENTICAL
# window, and the arm a controlled comparison. `axcodes` is deliberately NOT
# touched: orientation is a separate part of the input interface, and merlin /
# ctclip are geometrically wrong under anything but SRA (see their adapters).
#
# Cost is config-only. The window runs POST-cache (`build_train_post_transforms`)
# and `preprocessing_fingerprint` hashes only spacing / crop margin / num_classes,
# so every window variant shares the one existing disk cache — no repopulation.
ARM_W_WINDOWS = {
    # tag -> the intensity block forced onto every encoder. `None` means "drop the
    # per-encoder override", which lets base.yaml's own `data.intensity` apply —
    # so `wshared` equals the base window BY CONSTRUCTION rather than by a
    # duplicated literal that could drift out of sync with base.yaml.
    "wshared": None,
    # The SuPreM / VoCo soft-tissue window, forced onto encoders that never saw
    # it. This is the other half of the cross-over: it asks what the narrow
    # window costs an encoder whose representation is NOT adapted to it.
    "wnarrow": {"mode": "range", "a_min": -175.0, "a_max": 250.0,
                "b_min": 0.0, "b_max": 1.0, "clip": True},
}

# Run-tag suffix per window variant (mirrors ARM_S's `ctfm` -> `ctfmS`).
ARM_W_SUFFIX = {"wshared": "WS", "wnarrow": "WN"}


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

# Uniform batch across the matrix to remove the batch-size confound. Both values
# apply to EVERY encoder: with train.loss.batch_dice the Dice denominator pools
# over the micro-batch, so a per-encoder batch size would hand each encoder a
# different loss as well as a different optimizer recipe.
#
# Sized by scripts/vram_sweep.py against 80 GB cards (runs/vram_sweep/): bs=3 is
# the largest value every encoder fits, bounded by voco_h at 65.4 GB reserved;
# biomedparse / sam_med3d / voco_h all OOM at bs=4. num_samples_per_volume=2, so
# this is 6 patches per forward and 12 per optimizer update.
UNIFORM_BATCH_SIZE = 3
UNIFORM_GRAD_ACCUM = 2


def _frac_tag(f: float) -> str:
    return f"f{int(round(f * 100)):03d}"


def _load_model_block(stem: str) -> dict:
    cfg = yaml.safe_load((MODELS_DIR / f"{stem}.yaml").read_text())
    if "model" not in cfg:
        raise SystemExit(f"{stem}.yaml has no model: block")
    return cfg["model"]


def _base_intensity() -> dict:
    """base.yaml's own ``data.intensity`` — the window `wshared` resolves to."""
    base = yaml.safe_load((REPO / "configs" / "base.yaml").read_text())
    return dict(base["data"]["intensity"])


def _resolve_intensity(base_intensity: dict, preprocessing) -> dict:
    """The window an encoder actually receives, mirroring transforms.py.

    ``unified/data/transforms.py:_resolved_preprocessing`` merges the model's
    override ON TOP of ``data.intensity`` rather than replacing it, so a config
    that restates only ``a_min``/``a_max`` still inherits the base ``clip`` and
    ``b_*``. Reproduced here so the no-op detection below compares what the
    encoder really sees, not what its config happens to spell out.

    Normalized for BEHAVIOURAL comparison, not syntactic: an absent ``mode``
    means ``range`` (``transforms.py:_orient_intensity_list`` reads it with that
    default), and ints are coerced to float. Without this, the SuPreM / VoCo
    configs — which spell the soft-tissue window without a ``mode:`` key — would
    not be recognised as already being ``wnarrow``, and the generator would emit
    five byte-equivalent duplicate arms.
    """
    merged = {**base_intensity, **((preprocessing or {}).get("intensity") or {})}
    merged.setdefault("mode", "range")
    return {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
            for k, v in merged.items()}


def _apply_window(model_block: dict, window) -> dict:
    """Deep-copy ``model:`` with its intensity override forced to ``window``.

    ``window is None`` removes the override entirely so base.yaml's window
    applies. ``axcodes`` is preserved untouched — orientation is a separate part
    of the encoder's input interface, and merlin / ctclip are geometrically
    WRONG under anything but ``SRA`` (their adapters consume dim 2 as the axial
    axis). Arm W varies intensity and nothing else.
    """
    model = copy.deepcopy(model_block)
    pre = dict(model.get("preprocessing") or {})
    if window is None:
        pre.pop("intensity", None)
    else:
        pre["intensity"] = dict(window)
    if pre:
        model["preprocessing"] = pre
    else:
        model.pop("preprocessing", None)
    return model


def _apply_stem_fusion(model_block: dict, inplanes: int = ARM_S_INPLANES) -> dict:
    """Deep-copy ``model:`` with the shared Arm S stem switched on.

    ``registry.build_backbone`` pops ``stem_fusion`` / ``stem_inplanes`` and wraps
    the constructed backbone in ``StemFusionBackbone``, so no backbone module
    needs to know the arm exists and every encoder gets a BIT-IDENTICAL stem
    (349,584 parameters at ``inplanes=16``). That constant is the whole basis for
    reading the arm as a controlled comparison.
    """
    model = copy.deepcopy(model_block)
    kw = dict(model.get("kwargs") or {})
    kw["stem_fusion"] = True
    kw["stem_inplanes"] = inplanes
    model["kwargs"] = kw
    return model


def _plan_jobs():
    """Expand every arm into ``(stem, tag, arm, window_tag, model_block)`` jobs.

    Arm W rows are emitted only where the forced window actually DIFFERS from the
    encoder's own — `wshared` is a no-op for ctfm (its native window already is
    base.yaml's) and `wnarrow` is a no-op for the five soft-tissue encoders. For
    those cells the Arm N row IS the Arm W cell, so `window_cost` is 0 by
    construction; emitting a byte-identical duplicate would only burn a second
    GPU-week producing the same number. Skips are reported in the summary.
    """
    arm_of = {
        **{s: "N" for s in BACKBONES},
        **{s: "N" for s in ABLATIONS},   # same arm, one neck choice varied
        **{s: "B" for s in EXTRA_ARMS},
    }
    jobs, skipped = [], []
    for stem, tag in {**BACKBONES, **ABLATIONS, **EXTRA_ARMS}.items():
        jobs.append((stem, tag, arm_of[stem], "native", _load_model_block(stem)))

    # ARM S — the same encoder plus the shared stem, one generic code path.
    # Orthogonal to Arm W by design: crossing stem x window would quadruple the
    # matrix while identifying nothing the two separate arms do not already.
    for stem, tag in BACKBONES.items():
        jobs.append((stem, f"{tag}S", "S", "native",
                     _apply_stem_fusion(_load_model_block(stem))))

    base_intensity = _base_intensity()
    # Arm W is scoped to BACKBONES: it crosses the window factor with the
    # ENCODER factor. Crossing it with the neck ablations / Arm S / Arm B as well
    # would multiply the matrix for no extra identification — those arms already
    # hold the encoder fixed and vary something else.
    for wtag, window in ARM_W_WINDOWS.items():
        for stem, tag in BACKBONES.items():
            base_model = _load_model_block(stem)
            native = _resolve_intensity(base_intensity, base_model.get("preprocessing"))
            forced_block = _apply_window(base_model, window)
            forced = _resolve_intensity(base_intensity, forced_block.get("preprocessing"))
            if forced == native:
                skipped.append((stem, wtag))
                continue
            jobs.append((stem, f"{tag}{ARM_W_SUFFIX[wtag]}", "W", wtag, forced_block))
    return jobs, skipped


def build(dry_run: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    jobs, skipped = _plan_jobs()

    for stem, tag, arm, window, base_model in jobs:
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
                model = copy.deepcopy(base_model)
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
                    # positionally ($3 backbone, $6 fraction, $7 condition,
                    # $8 adapter, $9 probe_comparable, $10 arm, $11 window).
                    "adapter": adapter,
                    # "belongs in the headline Arm N probe column". Three ways to
                    # fall out: a pyramid mode that trains more than a thin neck,
                    # a fused stem (a raw-input conv branch — the adapter-string
                    # test alone would not catch "native+stem"), or a forced
                    # window, since an Arm W row is a clean thin-neck probe but of
                    # a DIFFERENT INPUT and merging it into the Arm N column would
                    # compare encoders across two normalizations at once.
                    "probe_comparable": (adapter not in NON_PROBE_MODES
                                         and not stem_fused
                                         and window == "native"),
                    "arm": arm,
                    "window": window,
                    # What Arm N's fine levels already were — the key that says
                    # how to read `compensable` for this encoder. See
                    # ARM_N_FINE_SOURCE.
                    "arm_n_fine_source": ARM_N_FINE_SOURCE.get(stem, "unknown"),
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
    n_w = sum(1 for j in jobs if j[2] == "W")
    n_s = sum(1 for j in jobs if j[2] == "S")
    print(f"{'PLANNED' if dry_run else 'WROTE'} {n} configs "
          f"({len(BACKBONES)} backbones + {len(ABLATIONS)} ablation + {n_s} "
          f"arm-S + {len(EXTRA_ARMS)} extra arms + {n_w} arm-W, x {len(CONDITIONS)} "
          f"conditions x {len(FRACTIONS)} fractions)")
    print(f"  conditions : {[c for *_, c in CONDITIONS]}")
    print(f"  fractions  : {[f'{int(f*100)}%' for f in FRACTIONS]}")
    print(f"  arm N      : {len(BACKBONES) * per_arm} probe-comparable configs")
    for stem, tag in ABLATIONS.items():
        print(f"  arm N abl  : {per_arm} configs  ls_{tag}_*  <- {stem} "
              f"(neck ablation; only a couple of cells need running)")
    print(f"  arm S      : {n_s * per_arm} configs across all {n_s} encoders — the "
          f"shared 349,584-param stem, subtract against the arm-N twin:")
    print(f"                 compensable(m) = gap_N(m) - gap_S(m)   "
          f"irreducible(m) = gap_S(m)")
    by_src = {}
    for stem in BACKBONES:
        by_src.setdefault(ARM_N_FINE_SOURCE.get(stem, "unknown"), []).append(stem)
    for srcname, members in sorted(by_src.items()):
        print(f"                 arm-N fine levels {srcname:22s} {', '.join(members)}")
    for stem, tag in EXTRA_ARMS.items():
        print(f"  arm B      : {per_arm} configs  ls_{tag}_*  <- {stem} "
              f"(probe_comparable=False; report separately)")
    print(f"  arm W      : {n_w * per_arm} configs across {len(ARM_W_WINDOWS)} forced "
          f"windows {sorted(ARM_W_WINDOWS)} — subtract against the arm-N twin:")
    print(f"                 window_cost(m, w) = score(m, w) - score(m, native)")
    for stem, wtag in skipped:
        print(f"                 SKIP {stem} x {wtag}: identical to its native "
              f"window — the arm-N row IS that cell (window_cost = 0)")
    print(f"  minimal set to size the effect: ARM=W COND='frz_pt|ft_pt' FRAC=1.0 "
          f"({n_w * 2} runs, + their {n_w * 2} arm-N twins if not already done)")
    if not dry_run:
        print(f"  -> {OUT_DIR}  (+ MANIFEST.csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build(dry_run=args.dry_run)
