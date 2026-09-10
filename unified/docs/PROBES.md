# Representation probes: cheap measurements over the frozen encoders

Training a cell of the low-shot matrix costs 5–30 GPU-hours. Some questions about
the *pretrained representations* do not need a training run to answer, and this
document records the ones that were measured directly instead — what they found,
and, importantly, where the probe turned out **not** to be a valid substitute.

Scripts: `scripts/_probe_capacity.py`, `_probe_features.py`, `_probe_redundancy.py`,
`_probe_union.py`, `_probe_window.py`. Result JSONs: `results/probes/`.
Feature caches are ~5 GB and regenerable, so they are not committed.

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

Alignment check: `CKA(ctfm, vista3d) = 0.42` on matched voxels vs `0.039` on
shuffled voxels. The correspondence is real.

Default sample: 32 val subjects × 4 patches × 400 voxels = 51,200 voxels,
116 classes present, subject-wise splits throughout (voxels within a patch are
heavily correlated — the unit of replication is the subject).

---

## 2. Where the probe is NOT valid: cross-encoder ranking

A voxel linear probe does **not** substitute for a training run. Spearman ρ
between probe balanced accuracy and benchmark mean Dice:

| benchmark cell | n | ρ (all) | ρ (arch-matched) | Pearson |
|---|---|---|---|---|
| `frz_pt` @ 10 % | 11 | 0.609 | 0.817 | 0.415 |
| `frz_pt` @ 25 % | 11 | 0.545 | 0.683 | 0.392 |
| `ft_pt` @ 10 % | 11 | 0.373 | 0.450 | 0.097 |
| `ft_pt` @ 25 % | 11 | 0.382 | 0.450 | 0.094 |

(`arch-matched` drops `ctclip` and `dino3d`, where the benchmark used `upsample` /
`vit_adapter` and the probe used the `_layerwise` variants.)

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

## 4. Does the union of encoders beat the best single one? No.

`_probe_union.py`. Subject-wise train/val/test, weight decay tuned per condition on
val, balanced accuracy on test, 2 seeds. The **duplicate control** — the best
encoder concatenated with itself — adds parameters and exactly zero information,
and is the null for any "more dimensions helped" explanation.

| condition | dim | test | vs solo | vs duplicate control |
|---|---|---|---|---|
| vista3d solo | 1,488 | 0.5957 | — | −0.014 |
| **vista3d ⊕ vista3d** (no new info) | 2,976 | **0.6094** | +0.014 | — |
| **all 12 encoders** | 18,240 | **0.5225** | −0.073 | **−0.087** |
| vista3d + voco_b (best pair) | 2,208 | 0.6009 | +0.005 | −0.009 |
| vista3d + ctclip (worst pair) | 4,048 | 0.5022 | −0.094 | −0.107 |

Duplicating an encoder **helps** (+0.014); adding eleven real foundation models
**costs** 0.087 against that control. No pairing beats the duplicate control.
Regularisation is not the explanation: wd = 100 is an interior optimum for all
three conditions, and every condition collapses to chance (0.0094) at wd ≥ 300.

Mechanism: **uniqueness and usefulness are anti-correlated**, ρ = −0.43 across the
twelve. `ctclip` has the most variance not linearly predictable from the other
eleven (0.658) and the worst probe score (0.057). The encoders carrying something
nobody else has carry something nobody needs.

Together with the per-class oracle over the existing runs (**+0.002** Dice over the
best single encoder — see the low-shot results), two independent measurements say
there is no exploitable complementarity between these encoders on dense anatomy.

---

## 5. The HU window (paired, so the probe is on firmer ground)

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

**This contradicts the reading in the README.** The clipping statistic there is
correct — the soft-tissue window does clip 27.6 % of foreground voxels and 97–99 %
of lung-lobe/trachea voxels. But the inference that the five soft-windowed encoders
are therefore *handicapped* does not follow. Hand them the information back and
they get **worse**, and worst on exactly the air-filled structures the clipping was
supposed to be destroying:

| encoder | air-structure recall, native → broad |
|---|---|
| suprem_unet | 0.705 → 0.570 (**−0.135**) |
| suprem_swinunetr | 0.573 → 0.449 (−0.124) |
| suprem_segresnet | 0.546 → 0.437 (−0.109) |

If this holds at benchmark level, the consequence for the arm structure is:
**there is no neutral window.** Forcing a shared window does not remove a confound,
it adds a per-encoder distribution-mismatch penalty of differing size (vista3d
−0.116 to narrow; voco_b −0.065 to broad). Arm N — each encoder in its own
pretraining window — is then the *least biased* comparison available, and Arm W
measures interface mismatch rather than information availability.

### 5.1 Pre-registered confirmation — RESOLVED

Three Arm W runs, frozen, 25 % data. Arm N baselines already exist, so no extra
runs are needed for the control side.

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

*The finding survives, smaller and more useful than predicted.* Window cost at
benchmark level is **−0.002 to −0.015 Dice** — against a 0.030 gap between these
two encoders in the same cell, and a 0.229 spread across all eleven encoders
frozen at 25 %. **Roughly 16× smaller than the between-encoder spread.** Forcing
vista3d into the soft-tissue window that clips 27.6 % of foreground voxels and
flattens 97–99 % of lung voxels costs 0.008 Dice. The trainable adapter absorbs
almost all of it.

So the README's caveat — that an Arm N ranking "cannot separate worse
representation from narrower input window" — is too strong. It can: the window
term is bounded at ~0.015 while the ranking spans 0.229. The clipping statistic is
right; the confound it was thought to create is small enough to quote a bound for
rather than control away with 360 runs.

The counterintuitive direction does hold: handing `suprem_unet` back the HU range
its pretraining window clipped makes it **worse** (−0.0146), the largest of the
three effects. The five soft-windowed encoders are not handicapped by their window.

**Limits.** Two encoders, one condition (frozen @ 25 %). Nothing here is tested at
other fractions, fine-tuned, or on the other nine encoders. The single-seed limit is
resolved in §5.2.

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
effect", not as a small one. The bound is unchanged: window cost is **0 to −0.015
Dice** against a 0.229 spread across eleven encoders.

**Scope limit, and it matters.** σ was measured in *one* cell, frozen at 25 %,
where only the 9.2 M adapter+head parameters move. Fine-tuned cells train the
whole encoder and are expected to be noisier. **This σ must not be carried to the
fine-tuned comparisons** — in particular the scratch-vs-pretrained result
(`ctfm` 0.8702 scratch vs 0.8679 pretrained, Δ = 0.0023 fine-tuned at 100 % data)
has *no* error bar and does not acquire one from this measurement. That cell needs
its own replicates before "pretraining is nearly free" can be claimed.

Secondary note: the benchmark's statistic is best-over-evals, a max, which can
compress variance relative to a single evaluation. Here it does not matter — the
sd of the final-epoch dice is 0.00059, essentially identical.

---

## 6. Caveats that apply to everything above

* Linear readout on frozen native features. This bounds *linear* extractability;
  a nonlinear fusion adapter could find more.
* 2 seeds, 32 of 57 val subjects. Per-seed spread on vista3d solo is 0.573–0.618,
  so effects under ≈ ±0.02 are noise.
* One task (TotalSegmentator dense anatomy), in-domain. Says nothing about the
  CT-RATE classification track or out-of-distribution behaviour.
* Encoders with no fine-scale native output (`ctclip`, `sam_med3d`) have
  structurally impoverished hypercolumns. Note this handicaps the *hypothesis* in
  §4, which lost anyway.

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
