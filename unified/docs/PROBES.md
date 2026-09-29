# Representation probes: cheap measurements over the frozen encoders

Training a cell of the low-shot matrix costs 5–30 GPU-hours. Some questions about
the *pretrained representations* do not need a training run to answer, and this
document records the ones that were measured directly instead — what they found,
and, importantly, where the probe turned out **not** to be a valid substitute.

Scripts: `scripts/_probe_capacity.py`, `_probe_features.py`, `_probe_redundancy.py`,
`_probe_union.py`, `_probe_window.py`. Result JSONs: `results/probes/`.
Feature caches are ~5 GB and regenerable, so they are not committed.

> **Corrections, 2026-09-15.** Several claims in the first version of this document did
> not survive re-verification. They are corrected in place:
>
> * §1: the CKA alignment figures did not match the committed result JSON.
> * §2: the correlation table used seed 0 only; it now uses both seeds.
> * §4: the representation-level union result is **withdrawn**. The probe's optimizer
>   makes weight decay 100 destructive rather than regularising. The class-level
>   benchmark evidence against complementarity stands.
> * §5: the probe's air-structure reading is **reversed** at benchmark level, and every
>   Arm W delta is confounded with a change in augmentation strength.
> * §5.3 (new): the fine-tuned noise floor, pretrained vs scratch, and the frozen random
>   twin. It replaces §5.2's statement that the fine-tuned comparison had no error bar.

---

## 1. The extraction contract

Cross-encoder voxel correspondence is the whole point, so the extractor
(`_probe_features.py`) is deliberate about it:

* One shared canonical frame — the model-independent det prefix (`Spacingd` →
  `CropForegroundd`) followed by `Orientationd(RAS)`.
* Intensity normalization is applied to the **whole canonical volume** before
  patching. It is orientation-invariant, and the `percentile` / `znorm` modes are
  volume-level statistics that would be wrong computed per-patch.
* Each encoder's `axcodes` permute/flip is applied to the patch, and the
  **inverse** is applied to its feature maps, putting every encoder back in RAS.
  Round-trip is exact for all three axcodes in use (`RAS`, `SPL`, `SRA`).
* All native levels are resampled to a common stride-4 grid (24³ at patch 96) and
  concatenated — one hypercolumn per voxel per encoder.
* Native encoder features are used, **not** the contract pyramid: the contract
  adapters are randomly initialised until trained, and would inject noise.

Alignment check: linear CKA between the `ctfm` and `vista3d` hypercolumns is **0.238** on
matched voxels (standardized features, all 51,200 voxels; `results/probes/redundancy_s1.json`),
or 0.203 with centering only. Shuffling one encoder's voxel order drops it to **0.001**.
The correspondence is real. (The first version quoted 0.42 vs 0.039; neither figure can be
reproduced from the committed outputs.)

Default sample: 32 val subjects × 4 patches × 400 voxels = 51,200 voxels,
116 classes present, subject-wise splits throughout (voxels within a patch are
heavily correlated — the unit of replication is the subject).

---

## 2. Where the probe is NOT valid: cross-encoder ranking

A voxel linear probe does **not** substitute for a training run. Spearman ρ
between probe balanced accuracy (mean of the two seeds of §4) and benchmark mean Dice
(legacy-framework runs, `results/legacy_lowshot/`):

| benchmark cell | n | ρ (all) | ρ (arch-matched) | Pearson |
|---|---|---|---|---|
| `frz_pt` @ 10 % | 11 | 0.691 | 0.933 | 0.408 |
| `frz_pt` @ 25 % | 11 | 0.645 | 0.800 | 0.387 |
| `ft_pt` @ 10 % | 11 | 0.436 | 0.533 | 0.072 |
| `ft_pt` @ 25 % | 11 | 0.445 | 0.533 | 0.070 |

(`arch-matched` drops `ctclip` and `dino3d`, where the benchmark used `upsample` /
`vit_adapter` and the probe used the `_layerwise` variants.)

The first version of this table used seed 0 only (ρ 0.609 / 0.545 / 0.373 / 0.382).
Single-seed values move by up to 0.13 in the arch-matched column. The probe scores also
come from the §4 fit, in which `ctclip_layerwise` and `dino3d_layerwise` selected the
destructive wd = 100, so only the arch-matched column is free of that artifact.

The probe carries some signal about **frozen** performance and essentially none
about **fine-tuned** performance — which is what you would expect, and is the
honest scope limit. At n ≈ 9–11 even the frozen correlation has a confidence
interval far too wide to rank encoders with. **Do not use the probe to order
encoders.**

A **paired** comparison — same encoder, same voxels, same readout, one input factor
changed — was expected to be better conditioned, since it removes the
between-encoder variance driving the noise above. **It is not** (§5.1): the paired
window deltas were confirmed against training runs and the probe overestimated them
by 2–15× and mis-ranked which encoder was affected most. The probe is not a
quantitative instrument in any form. Use it to generate hypotheses, never to
measure.

---

## 3. Capacity: can K encoders coexist?

`_probe_capacity.py`, all 12 with real checkpoints, 96³ batch 2, bf16, one A100:

| quantity | value |
|---|---|
| total resident weights | 5.23 GB |
| peak, sequential fused forward | 9.40 GB |
| latency, all 12 | 574 ms |
| latency, vista3d alone | 31.6 ms (**18× cheaper**) |

A fusion backbone or a 12-teacher distillation step is not compute-limited on this
hardware. Compute was never the obstacle.

---

## 4. Does the union of encoders beat the best single one? The benchmark says no; the probe cannot say

**The class-level evidence stands.** Taking the best encoder's benchmark Dice for every
class separately gains only **+0.002** over the best single encoder (legacy-framework
runs at 10 % and 25 %), and `vista3d` wins 86–100 of 115 classes in every cell. There is
little complementarity to exploit on dense anatomy.

**The representation-level result is withdrawn.** `_probe_union.py` reported that all
twelve encoders concatenated scored 0.087 below `vista3d` concatenated with itself, and
that no pairing beat that duplicate control. Those numbers are not a valid measurement.
The probe fits with AdamW at lr 3e-2 under a 400-step cosine schedule, and PyTorch applies
decoupled weight decay as `param.mul_(1 - lr * weight_decay)`. At wd = 100 that factor is
−2 on the first step, and it passes through ≈ 0 about 61 % of the way through the
schedule before becoming an ordinary shrinkage. Everything learned before that point is
wiped, so "wd = 100" means "fit with the last ~40 % of steps at lr ≤ 0.01", not strong L2.

Validation picked wd = 100 for the full union and the duplicate control in both seeds,
and for 19 of 22 pairings, but wd ≤ 10 for 17 of 24 single-encoder fits. The conditions
were therefore compared under different effective training budgets. The earlier check
that every condition collapses to chance at wd ≥ 300 used the same fit and shows the same
artifact: at wd = 300 the zero crossing moves to 78 % of the schedule.

The uniqueness-vs-usefulness correlation (ρ = −0.43) is withdrawn with it, because its
usefulness axis is the same probe score. The raw numbers stay in
`results/probes/union.json` for the record. A valid rerun needs a fit in which weight
decay cannot flip or erase the weights: a closed-form ridge, or a grid with lr × wd ≪ 1.

---

## 5. The HU window

`_probe_window.py`. All 12 encoders extracted under three windows — native,
narrow/soft-tissue `[-175, 250]`, broad `[-1024, 2048]` — and probed under each.

**Validity check:** where a forced window coincides with an encoder's native one,
the delta is **exactly 0.0000** (the five natively-soft encoders under narrow;
`ctfm` under broad). The override path is correct and the pipeline deterministic.

| encoder | native | → narrow | → broad |
|---|---|---|---|
| vista3d | 0.5981 | **−0.116** | +0.013 |
| ctfm | 0.3254 | −0.012 | 0.0000 |
| sam_med3d | 0.3036 | −0.065 | −0.064 |
| dino3d (layerwise) | 0.4230 | −0.017 | +0.002 |
| suprem_unet * | 0.4517 | 0.0000 | **−0.031** |
| voco_b * | 0.4245 | 0.0000 | **−0.065** |
| suprem_segresnet * | 0.2749 | 0.0000 | **−0.064** |
| voco_h * | 0.3494 | 0.0000 | **−0.053** |
| suprem_swinunetr * | 0.3195 | 0.0000 | **−0.054** |

\* natively soft-windowed.

Every encoder scores best in the window it was pretrained on; the largest gain from
any non-native window is +0.015, inside noise.

Neither these probe deltas nor their per-anatomy breakdown should be used as a
measurement, for two reasons:

* **The fit.** `_probe_window.py` uses the same AdamW fit and wd grid as §4. Some
  forced-window conditions selected the destructive wd = 100 while their native baseline
  selected wd = 10 (seed 0 of `vista3d → broad` and `suprem_unet → broad` among them), so
  those deltas partly compare different effective training budgets.
* **The per-anatomy reading reversed at benchmark level.** The probe showed the three
  SuPreM encoders losing most on air-filled structures when given the broad window
  (`suprem_unet` air-structure recall 0.705 → 0.570). The benchmark shows the opposite
  (§5.1): `suprem_unet` improves on all six air structures and loses on bone and soft
  tissue instead.

The conclusion the first version drew from that reading is withdrawn: that there is no
neutral window, and that Arm W measures interface mismatch rather than information
availability.

### 5.1 Pre-registered confirmation — RESOLVED

Five runs, frozen, 25 % data, **all five under the current framework** — the Arm N
twins were re-run rather than taken from `results/legacy_lowshot/`, because those
are July-17 runs at effective batch 4 / fp32 / 135 steps per epoch against the
current 6 / bf16 / 90. That drift is worth about as much Dice as the window effect
itself, so the legacy baselines are not a valid control here. This is what
`wandb.run_tag: v2` in `base.yaml` is warning about.

| encoder | window | Arm N | Arm W | **actual Δ** | probe predicted | over-estimate |
|---|---|---|---|---|---|---|
| vista3d | narrow | 0.8330 | 0.8253 | **−0.0077** | −0.116 | **15×** |
| vista3d | broad | 0.8330 | 0.8313 | −0.0017 | +0.013 | sign wrong |
| suprem_unet | broad | 0.8030 | 0.7884 | **−0.0146** | −0.031 | 2.1× |

**Two conclusions, and they point opposite ways.**

*The probe failed.* It overestimated every delta by 2–15×, got one sign wrong, and
mis-ranked the effects — it called vista3d-under-narrow the largest effect
(−0.116) when suprem_unet-under-broad is (−0.0146). §2's scope limit therefore
extends to paired comparisons too.

*The measured effect is small.* The change at benchmark level is **−0.002 to −0.015
Dice** — against a 0.030 gap between these two encoders in the same cell, and a 0.229
spread across all eleven encoders frozen at 25 % (legacy-framework runs). **Roughly 16×
smaller than the between-encoder spread.** Forcing vista3d into the soft-tissue window
that clips 27.6 % of foreground voxels and flattens 97–99 % of lung voxels changes its
Dice by −0.008 — but none of these deltas is the window alone (next paragraph).

**Confound: these deltas are not the window alone.** The additive intensity
augmentations are scaled to the normalized range, not to HU (`HEAD_DESIGN.md` §7), so a
0.10 shift or noise std means 0.10 × the window width in HU. Every Arm W run therefore
also changed augmentation strength:

| run | window | shift / noise std in HU | vs native |
|---|---|---|---|
| `vista3d` native | `[-964, 1054]` | 202 | — |
| `vista3d` → narrow | `[-175, 250]` | 42.5 | 4.7× weaker |
| `vista3d` → broad | `[-1024, 2048]` | 307 | 1.5× stronger |
| `suprem_unet` native | `[-175, 250]` | 42.5 | — |
| `suprem_unet` → broad | `[-1024, 2048]` | 307 | 7.2× stronger |

The measured 0 to 0.015 Dice is a bound on window **plus** augmentation change. It is
still small against the between-encoder spread, but it cannot be attributed to the
window. The README's caveat stands until the augmentations are defined in HU.

**Where the change lands.** Best-epoch Dice averaged per anatomy group (air = lung lobes
and trachea, 6 classes; bone, 63; soft tissue, 46):

| run | air | bone | soft tissue | overall |
|---|---|---|---|---|
| `vista3d` → narrow | **−0.032** | −0.007 | −0.006 | −0.0077 |
| `vista3d` → broad | −0.002 | −0.002 | −0.002 | −0.0017 |
| `suprem_unet` → broad | **+0.015** | −0.014 | −0.020 | −0.0146 |

Clipping the lungs hurts the lungs. Narrowing `vista3d`'s window costs most on the air
structures, and giving `suprem_unet` back the HU range its window clipped improves all
six of them, which is the information loss the clipping statistic predicts.
`suprem_unet`'s net loss comes from bone and soft tissue, and it coincides with a 7.2×
stronger augmentation in HU. The first version's claim that "the five soft-windowed
encoders are not handicapped by their window" is withdrawn. This is one run per arm, but
the air column is above per-class noise: across the four `vista3d` frozen replicates of
§5.2, per-class sd is 0.0042 median (0.0123 at p90) and only 0.0022 for the six air
classes, so −0.032 and +0.015 are roughly 14× and 7× that spread.

**Limits.** Two encoders, one condition (frozen @ 25 %). Nothing here is tested at
other fractions, fine-tuned, or on the other nine encoders. The single-seed limit is
resolved in §5.2; the augmentation confound is not.

---

### 5.2 The noise floor, measured

Everything above compares single runs. Until now this benchmark had never measured
its own run-to-run variance, so no delta in it was falsifiable.

**These are true replicates.** `train.seed: 42` in `configs/base.yaml` is never
consumed — there is no `torch.manual_seed` anywhere in the training path
(documented at `unified/utils/checkpoint.py:127`). Adapter initialisation and the
augmentation stream are drawn fresh per process, so repeating one config yields
independent replicates rather than cudnn-nondeterminism twins.

Four runs of `ls_vista3d_frz_pt_f025`, identical config, current framework:

| run | best dice | @epoch |
|---|---|---|
| `runs/lowshot/ls_vista3d_frz_pt_f025` | 0.83303 | 450 |
| `runs/noise/vista3d_frz_f025_r2` | 0.83235 | 450 |
| `runs/noise/vista3d_frz_f025_r3` | 0.83313 | 500 |
| `runs/noise/vista3d_frz_f025_r4` | 0.83386 | 450 |

**σ = 0.00062** (n = 4, mean 0.83309, range 0.00151, 95 % CI on σ
[0.00035, 0.00230]). A difference of two single runs therefore carries
SE = σ√2 = 0.00087.

Applying that to §5.1 — `z` at the point estimate, and at the pessimistic end of
the σ CI:

| effect | Δ | z | z at σ_hi | verdict |
|---|---|---|---|---|
| suprem_unet, broad | −0.0146 | 16.7 | 4.5 | **real** |
| vista3d, narrow | −0.0077 | 8.8 | 2.4 | **real** |
| vista3d, broad | −0.0017 | 1.9 | 0.5 | **not distinguishable from noise** |

So the window bound of §5.1 survives, and survives even if σ is three times what
we measured. The −0.0017 row does not, and should be quoted as "no measurable
effect", not as a small one. The bound is unchanged: window-plus-augmentation cost is
**0 to −0.015 Dice** against a 0.229 spread across eleven encoders (legacy framework).

**Scope limit, and it matters.** σ was measured in *one* cell, frozen at 25 %,
where only the 9.2 M adapter+head parameters move. Fine-tuned cells train the
whole encoder and are expected to be noisier. **This σ must not be carried to the
fine-tuned comparisons.** §5.3 measures the fine-tuned floor separately, and it is about
three times larger.

Secondary note: the benchmark's statistic is best-over-evals, a max, which can
compress variance relative to a single evaluation. Here it does not matter — the
sd of the final-epoch dice is 0.00059, essentially identical.

### 5.3 Fine-tuned noise floor, pretrained vs scratch, and the frozen random twin

These are benchmark results rather than probe results, but they extend §5.2's replicate
logic to fine-tuned cells and to what the pretrained weights are worth. Numbers:
`results/probes/pretrain_vs_scratch.json`.

**Fine-tuned noise floor.** Four runs of `ls_ctfm_ft_sc_f100` (scratch, 100 % data) score
0.8702, 0.8665, 0.8702 and 0.8689. **σ = 0.00173** (95 % CI [0.00098, 0.00646]), about
2.8× the frozen floor. A two-run difference carries SE ≈ 0.0024.

**Pretrained minus scratch, fine-tuned at 100 % data:**

| encoder | pretrained | scratch | Δ | z |
|---|---|---|---|---|
| `ctfm` | 0.8679 | 0.8690 (mean of 4) | −0.0010 | −0.43 |
| `vista3d` | 0.8720 | 0.8699 | +0.0021 | +0.85 |
| `biomedparse` | 0.8639 | 0.8573 | +0.0066 | +2.72 |
| `suprem_swinunetr` | 0.8450 | 0.8453 | −0.0003 | −0.12 |
| `voco_b` | 0.8553 | 0.8607 | −0.0054 | −2.21 |

Every z uses the `ctfm` σ. At full data with fine-tuning, no encoder gains more than
0.007 Dice from its pretrained weights, and the sign is not consistent: `biomedparse` is
+2.7 SE, `voco_b` is −2.2 SE, and the other three sit inside ±1 SE. The mean over the
five is **+0.0004**. The earlier reading "`ctfm` scratch ≥ pretrained" (0.8702 vs 0.8679)
rested on one scratch run; against four replicates the difference is −0.001, inside
noise.

**Frozen pretrained minus frozen random twin, 100 % data.** Architecture, adapter,
decoder and recipe are identical; only the encoder weights differ.

| encoder | frozen pretrained | frozen random | Δ | random retains |
|---|---|---|---|---|
| `voco_b` | 0.8436 | 0.8083 | +0.035 | 95.8 % |
| `biomedparse` | 0.8429 | 0.8248 | +0.018 | 97.9 % |
| `suprem_swinunetr` | 0.8342 | 0.8211 | +0.013 | 98.4 % |

These gaps are 15–40× the frozen two-run SE of §5.2 (measured at 25 %), so they are real.
All three encoders now have all four cells, and in each the advantage shrinks once the
encoder is fine-tuned — most sharply for `voco_b`, which has the largest frozen gap and a
slightly negative fine-tuned one:

| encoder | frozen | fine-tuned |
|---|---|---|
| `voco_b` | +0.035 | −0.005 |
| `biomedparse` | +0.018 | +0.007 |
| `suprem_swinunetr` | +0.013 | −0.000 |

Whatever the frozen probe is measuring, fine-tuning at full data removes nearly all of
it.

**Limits.**

* One pretrained run per encoder, and σ borrowed from `ctfm` scratch.
* **Two of these encoders were pretrained on this dataset.** VISTA3D's own fold list
  (`VISTA/vista3d/data/jsons/TotalSegmentatorV2_5_folds.json`) puts 47 of our 57
  validation and 73 of our 89 test subjects in its training folds, so `vista3d`'s
  pretrained side is scored largely on data it trained on with labels — and still gains
  +0.002. VoComni's list (`Large-Scale-Medical/VoComni/VoComni.json`) is renumbered and
  carries no TotalSegmentator ids, so `voco_b`'s overlap is unknown.
* One recipe: a single AdamW learning rate for encoder and decoder, which can distort
  pretrained features (Kumar et al., 2022). An encoder-LR control is needed before
  claiming pretrained ≈ scratch.
* All three random twins are transformers whose stride-1 level (strides 1–2 for
  `biomedparse`) comes from a trainable raw-input stem rather than the encoder
  (`arm_n_fine_source`), which is where the encoder matters least. The retention figures
  may not carry over to native-stride-1 CNNs. BatchNorm encoders also need their norm
  statistics recalibrated before a random twin is a fair null.
* One random draw per twin.
* Validation split only (57 subjects), which also selects the checkpoint; the test split
  is unused.
* The `vista3d` and `biomedparse` scratch runs were killed on 2026-09-04 and resumed from
  epoch checkpoints, and `voco_b` pretrained was resumed from epoch 20. Resuming restores
  optimizer, scheduler and scaler state.

---

## 6. Caveats that apply to everything above

* Linear readout on frozen native features. This bounds *linear* extractability;
  a nonlinear fusion adapter could find more.
* 2 seeds, 32 of 57 val subjects. Per-seed spread on vista3d solo is 0.573–0.618,
  so effects under ≈ ±0.02 are noise.
* Every benchmark number is best-epoch Dice on the 57-subject validation split, which also
  selects the checkpoint; the test split has not been used. The training path sets no
  seeds.
* One task (TotalSegmentator dense anatomy), in-domain. Says nothing about the
  CT-RATE classification track or out-of-distribution behaviour.
* Encoders with no fine-scale native output (`ctclip`, `sam_med3d`) have
  structurally impoverished hypercolumns. That handicaps the representation-level test
  of §4, which is withdrawn in any case.

## 7. Reproducing

```bash
# feature caches (~5 GB per window), then the analyses
python -m scripts._probe_capacity  --out results/probes/capacity.json
python -m scripts._probe_features  --n-subj 32 --n-patch 4 --n-vox 400 --out /tmp/feats
python -m scripts._probe_features  --n-subj 32 --n-patch 4 --n-vox 400 \
    --window -175 250   --out /tmp/feats_soft
python -m scripts._probe_features  --n-subj 32 --n-patch 4 --n-vox 400 \
    --window -1024 2048 --out /tmp/feats_broad
python -m scripts._probe_union     --feats /tmp/feats --seeds 2 --out results/probes/union.json
python -m scripts._probe_window    --roots native=/tmp/feats soft=/tmp/feats_soft \
    broad=/tmp/feats_broad --seeds 2 --out results/probes/window.json
```
