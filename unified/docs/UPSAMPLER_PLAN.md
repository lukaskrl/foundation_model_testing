# Plan: an encoder-agnostic 3D feature upsampler for frozen CT foundation models

**Status:** drafted 2026-09-28. Steps A–E implemented the same day (see "Implementation status" at
the end); Stage 1 running.
**Target:** ISBI 2027 four-page paper, due **Mon 26 Oct 2026**, 11:59pm EDT (biomedicalimaging.org/2027).

## The claim

> A 3D, image-guided feature upsampler, trained once without labels, lifts the coarse features of *any*
> frozen 3D foundation model to fine resolution. It clearly improves thin-structure segmentation,
> including on encoders it never saw.

Every experiment below serves that sentence.

### Why this problem

- Most public 3D ViT foundation models use 16³ patches (3DINO, SAM-Med3D, CT-CLIP, SPECTRE, VoxelFM). At
  1.5 mm spacing one token covers 24 mm, so these models are good at semantics and bad at fine structures.
- Measured in this repo (`docs/HEAD_DESIGN.md` §7, runs/interface): frozen dino3d trails frozen ctfm by
  0.079 Dice under the same I1 head. The gap tracks structure thinness (Spearman −0.76), and a 37k-parameter
  raw-CT stem closes 79% of it. So most of a frozen ViT's dense deficit here is a resolution-interface problem.
- 2D encoder-agnostic upsamplers exist: AnyUp (arXiv 2510.12764, ICLR 2026 oral), JAFAR, LoftUp, FeatUp,
  DiveUp (arXiv 2603.13571). No 3D or medical version was found in arXiv checks on 2026-09-28.
- Re-check 2026-10-02 found closer work, none of which upsamples native 3D encoders encoder-agnostically:
  - VoxelFeat (MIDL 2025): FeatUp extended to CT volumes with 3D position encodings, for a 2D encoder
    (SAM2), label-free, trained per encoder, used for interactive segmentation. Must be cited; our
    novelty claim narrows to "encoder-agnostic, for native 3D encoders, transfers unchanged".
  - VoxCor (arXiv 2605.13798): training-free triplanar 2D-ViT features plus a closed-form projection.
  - Upsample Anything (CVPR 2026): per-image test-time-optimised Gaussian kernels, encoder-agnostic,
    training-free. A 3D port is the strongest training-free baseline after our bilateral.
  - RaysUp (arXiv 2606.22749), Weighted Reverse Convolution (arXiv 2605.17472): more 2D upsamplers.
  - "Big, Bright, or Invisible" (arXiv 2608.05960): frozen 3D CT encoders miss small, low-contrast
    findings. Supports the motivation.

### Why it is a method, not a benchmark

- It is one module with one objective and one property: it works on any encoder without retraining.
- The CT decides only the attention weights, and the output is a weighted mix of the encoder's own
  features. It can place semantics precisely but cannot invent them. That separates resolution (from the
  image) from knowledge (from the encoder), so the "the head is doing the work" critique does not apply.
- It improves the features, not one task's output, so the same module serves segmentation, detection or
  dense probing.
- It survives either outcome. If it generalizes to unseen encoders, that is the headline. If it needs
  per-encoder training, it is still the first 3D guided upsampler, with measured thin-structure gains.

---

## 1. What to implement

### A. Feature extraction (about half a day, mostly reuse)

- Take each ViT's native tokens from its existing `encoder_forward`:

  | encoder | channels | native grid (96³ patch) |
  |---|---|---|
  | dino3d | 1024 | 6³, stride 16 |
  | sam_med3d | 384 | 8³ on a 128³ canvas, stride 12 in patch space |
  | ctclip | 512 | 10×24×24, stride 9.6×4×4 (anisotropic) |

- Map features back to RAS. (Correction: `scripts/_probe_features.py` does NOT get this right for SPL
  and SRA encoders. Its round trip is consistent, but the encoder sees a mis-oriented patch. The
  upsampler uses `unified/upsampler/geometry.py:apply_ornt`, which is tested against MONAI.)
- Carry each grid's physical coordinates, so the upsampler never needs a regular lattice.
- Work at 1.5 mm (`data.use_source_affine: true`), not the 2.25 mm corpus.

### B. The upsampler module (2–3 days, the core)

- **Guidance encoder:** a small 3D CNN (2–3 layers, 32–64 channels) over the CT at the target resolution.
  It produces one query per fine voxel. Pooled onto the coarse grid, it produces one key per coarse cell.
- **Attention:** each fine voxel attends to the ~27 nearest coarse cells in physical space. The score is
  query · key plus a learned bias on the relative position. The output is a weighted sum of the **raw
  encoder features**, with no projection. That is what makes it dimension-agnostic and encoder-agnostic,
  and what stops it from inventing semantics.
- **Output:** stride 2 by default, then trilinear to stride 1. Ablate stride 4 and stride 1.
- **Memory:** do not gather 27 copies of the features. That would be about 6 GB for a 48³ output with
  1024 channels in bf16. Loop over the 27 offsets instead and accumulate `weight × shifted(features)`,
  which needs about 0.25 GB per offset.
- **Version 2, only if needed:** let the keys also read the features through a per-channel,
  channel-count-invariant layer (AnyUp's feature-agnostic layer), so it stays dimension-agnostic.

### C. Label-free training (1–2 days)

- **Primary objective:** downsample the CT by 2× and run the frozen encoder on it, giving coarse features.
  Train the module, guided by the full-resolution CT, to predict the encoder's features on the
  full-resolution CT. Loss: cosine distance plus L2, on standardized features. Apply higher ratios at test
  time: JAFAR trains at low ratios and applies at higher ones, and AnyUp trained on a single encoder
  generalizes to unseen ones.
- **Alternative to compare:** targets from 2×-zoomed crops (AnyUp style).
- **Fallback, if both suffer from scale mismatch:** train through a linear segmentation head on labels.
  It is still one module for all encoders, just not label-free.
- **Data:** TotalSegmentator training volumes, labels unused. Encoders frozen. About 12–24 GPU-hours per
  trained upsampler.

### D. Evaluation code (1–2 days)

- **Main measurement:** frozen encoder → upsampler → **linear per-voxel classifier**, Dice at full
  resolution. A linear readout is right here because the method is about feature quality, and it is how
  AnyUp, JAFAR and FeatUp evaluate. Fit it with learning rate × weight decay ≪ 1, or with closed-form
  ridge regression. The earlier probes in `docs/PROBES.md` §4 were broken by AdamW at wd = 100.
- **Secondary measurement:** the upsampled features inside a light decoder, to show real-use benefit.
- **Metrics:**
  - mean Dice;
  - Dice on thin structures (under 8 mm; thickness proxy 2V/S from validation labels);
  - boundary accuracy (NSD, HD95);
  - feature fidelity (cosine similarity to held-out full-resolution features);
  - time and memory.

### E. Unit tests (about half a day)

- One set of weights runs on 384-, 512- and 1024-channel features.
- Anisotropic and non-power-of-two grids work.
- A constant CT reduces the module to smooth interpolation.
- Only the upsampler receives gradients.
- A time and memory benchmark at 96³.

---

## 2. Experiments, in order, with go/no-go points

### Stage 1: does it work at all? (days 4–7)

- Train on dino3d.
- On a small TotalSegmentator subset (e.g. 50 training and 20 validation volumes), compare the linear
  readout on upsampled features against the same readout on trilinear-upsampled features.
- **Go:** a clear thin-structure gain (e.g. at least +0.03 Dice) and no loss overall. Otherwise switch to
  the crop objective or the label-supervised fallback before going further.

### Stage 2: main results (days 7–13)

- **Generalization, the headline:** apply the dino3d-trained upsampler unchanged to SAM-Med3D and CT-CLIP,
  and to the coarse level of a CNN (ctfm) as an unseen architecture family.
- **Baselines:**
  - trilinear interpolation;
  - classic 3D joint bilateral upsampling (not learned);
  - the same module trained per encoder (the upper bound for generalization);
  - **2D AnyUp applied slice by slice.** It works on any features out of the box, so reviewers will ask
    for it. It is the test that 3D matters.
  - a raw-CT convolutional stem, as a fine-path alternative.
- **External set:** HaN-Seg, which has many tiny structures. Train linear readouts on a few cases. It needs
  downloading.

### Stage 3: ablations (days 13–17)

- attention reading only the CT vs also the features;
- neighbourhood size;
- output stride;
- training objective (downsampling vs crops);
- training on 1 vs 2 encoders;
- how the CT is windowed for the guidance encoder;
- optionally, the end-to-end decoder result.

### Stage 4: writing (days 17–28)

Freeze the experiments around **16 October**.

---

## 3. Pass bars (fill in X, Y, Z after Stage 1, before Stage 2)

- **Unseen encoders:** thin-structure Dice improves by at least **X** over trilinear, and comes within
  **Z** of the upsampler trained on that encoder.
- **2D AnyUp slice-wise:** beaten on thin-structure and boundary metrics.
- **Cost:** under **Y**% of the encoder's own forward time.

## 4. Compute and risks

- **Compute:** about 150–200 GPU-hours in total. On 2026-09-28 all four GPUs were running other jobs.
- **Risks and fallbacks:**
  - **scale mismatch in the label-free objective:** fall back to label-supervised training;
  - **memory:** stay at output stride 2;
  - **no generalization to unseen encoders:** per-encoder training, the smaller "first 3D guided
    upsampler" claim;
  - **attention reading only the CT is too weak:** add the channel-independent feature term.
- **Novelty:** re-check arXiv the week before submission. AnyUp's authors could publish a 3D version.

## 5. Open decisions

1. Which ViTs are in scope: dino3d to train on; SAM-Med3D and CT-CLIP to test. Are there public weights for
   SPECTRE or VoxelFM worth adding?
2. HaN-Seg as the external set, and whether it is available locally.
3. Is label-free training the primary objective, with the label-supervised fallback accepted?

---

## Implementation status (2026-09-28)

### Code

| piece | file | notes |
|---|---|---|
| grid geometry | `unified/upsampler/geometry.py` | per-axis cell centers; box pooling; neighbour tables; trilinear at any centers; orientation (nibabel semantics, checked against MONAI) |
| module + baselines | `unified/upsampler/module.py` | `GuidedUpsampler3D`; memory-light custom autograd for the 27-neighbour dot and aggregate; `TrilinearUpsampler`; `BilateralUpsampler` (joint bilateral, not learned) |
| encoders | `unified/upsampler/encoders.py` | `FrozenEncoder(stem)`: RAS in, last-layer features + exact centers out; per-volume intensity as clip-affine |
| data | `unified/upsampler/data.py`, `scripts/upsampler_prepare.py` | 446 volumes (300 train, 57 val, 89 test) at 1.5 mm RAS, int16/uint8 memmaps, 19 GB at `/home/lukas/data/cache/upsampler/vol15` |
| objectives | `unified/upsampler/objectives.py`, `scripts/upsampler_train.py` | `down`, `crop` (label-free), `label` (fallback through a throwaway linear head; excludes the probe bank's volumes and the val split) |
| evaluation | `unified/upsampler/probe.py`, `scripts/upsampler_probe.py`, `scripts/upsampler_report.py` | fixed class-balanced 96³ patch banks (400 train / 160 val); one linear probe per method, applied before upsampling (exact, since every method is linear in the features); Dice, thin (< 8 mm, 46 classes) and thick |
| tests | `scripts/test_upsampler.py` | 9 module checks + `--encoders` + `--store`, all passing |

### Encoder geometry as implemented (96³ patch, RAS)

| encoder | features | cell centers (voxels) |
|---|---|---|
| dino3d (last block) | 1024 × 6³ | 16i + 7.5 |
| sam_med3d | 384 × 8³ | (i + 0.5)·12 − 0.5 (resized to the 128³ canvas) |
| ctclip | 512 × 24 × 24 × 10 | in-plane (i + 0.5)·4 − 0.5; S axis 10i + 4.5 (depth padded at the end) |
| ctfm (level 4) | 512 × 6³ | 16i + 15 on R and A, 16i on S. Its stride-2 k3/p1 convs centre cells at 16i in its SPL frame; the P and L flips move them. Trilinear with `align_corners=False` assumes 16i + 7.5 everywhere. |

### Cost (A100, batch of 2, 96³, stride-2 output over all channels)

1024 × 6³: 168 ms inference, 1.3 GiB; 340 ms train step, 6.2 GiB. 384 × 8³: 81 ms. 512 × 24² × 10: 106 ms.

### Stage 1 results (dino3d, 2026-09-28)

Protocol: dino3d last-layer features (1024 × 6³ per 96³ patch at 1.5 mm); one linear probe per method,
fit on 400 class-balanced patches from 50 train volumes and scored on 160 patches from 20 val volumes;
20 epochs, AdamW lr 1e-3, wd 1e-4. Features are divided by one per-encoder std and the probe bias
starts at the log class prior (see `fit_and_eval` for the A/B that chose this: centring without the
prior, or per-channel standardization, under-fit badly). Neighbourhood methods output at stride 2,
then trilinear to stride 1. "Thin" = 46 classes under 8 mm (2V/S proxy). NSD at 3 mm.

| method | Dice | thin | thick | NSD 3 mm | Δ thin | thin classes up |
|---|---|---|---|---|---|---|
| trilinear | 0.338 | 0.154 | 0.461 | 0.386 | | |
| untrained module (Gaussian kernel) | 0.339 | 0.152 | 0.463 | 0.368 | −0.002 | 20/46 |
| joint bilateral (not learned) | 0.348 | 0.152 | 0.479 | 0.388 | −0.001 | 25/46 |
| label-free, `crop` objective | 0.341 | 0.157 | 0.464 | 0.376 | +0.003 | 22/46 |
| label-free, `down` objective | 0.362 | 0.178 | 0.485 | 0.442 | +0.024 | 32/46 |
| **label-supervised** (fallback) | **0.443** | **0.305** | **0.535** | **0.538** | **+0.152** | **43/46** |

Same labelled pool as the label-supervised module (probe fit on 288 train volumes, 4 epochs = same
step count): trilinear 0.379 / thin 0.173 → upsampler 0.488 / thin 0.343. The gain is not the extra
labels.

Where it helps and hurts (label-supervised vs trilinear, per class): ribs and skull +0.29 to +0.39;
brain −0.30, spinal cord −0.16, gluteus minimus/medius left −0.10/−0.07. The losses sit in soft
tissue enclosed by bone, which fits attention pulling bone tokens across the boundary.

**Go/no-go:** GO for the label-supervised module (Δ thin +0.152 ≫ the +0.03 bar). Label-free
objectives fail the bar (`down` +0.024, `crop` +0.003). This answers open decision 3 in favour of
the fallback; the claim becomes "trained once, with labels, on one encoder".

### Stage 2, first pass: unseen encoders (2026-09-28)

The dino3d-trained, label-supervised module applied **unchanged** (final step-4000 checkpoint; the
step-2000 snapshot is within 0.01 everywhere). Each encoder gets its own linear probe on the same
patch banks, same protocol.

| encoder (never seen by the module) | trilinear Dice / thin | module Dice / thin | Δ thin | Δ NSD 3 mm | thin up | bilateral Δ thin |
|---|---|---|---|---|---|---|
| dino3d (training encoder, for reference) | 0.338 / 0.154 | 0.443 / 0.305 | +0.152 | +0.152 | 43/46 | −0.001 |
| SAM-Med3D (ViT, 384 ch, 8³, stride 12) | 0.241 / 0.124 | 0.329 / 0.247 | +0.124 | +0.108 | 41/46 | +0.009 |
| CT-FM level 4 (CNN, 512 ch, stride 16) | 0.192 / 0.063 | 0.270 / 0.147 | +0.084 | +0.120 | 36/46 | −0.008 |
| CT-CLIP (ViT, 512 ch, 24 × 24 × 10) | 0.054 / 0.058 | 0.098 / 0.116 | +0.058 | +0.048 | 42/46 | +0.037 |

- CT-CLIP's last layer barely carries per-location class information in this setting: neighbouring
  cells differ nearly as much as random values (mean |Δ| 0.89 at std 1.0), and its probe loss plateaus at
  2.42 (dino3d about 1.1). Likely the in-plane 5× resize of a 144 mm patch onto a 480 canvas that
  CT-CLIP was pretrained at 360 mm. Its intermediate taps are the fairer test.
- Not yet done: a module trained per encoder (the
  generalization upper bound); 2D AnyUp slice-wise; the same-data control for the unseen encoders;
  full-volume sliding-window evaluation; the test split.

### Inside a trained head (started 2026-09-29)

Question: does the frozen upsampler still help when a real decoder is trained on top, or does the
decoder learn the same edge alignment itself? Design, all on frozen dino3d at 1.5 mm, full data:

| arm | config | status |
|---|---|---|
| I1: interface head, no fine path | `dino3d_i1_frz_pt_f100` | done: 0.7713 best val Dice |
| I1 + trilinear stride-2 skip (control) | `dino3d_i1t_frz_pt_f100` | running on GPU0 since 2026-09-29 11:19 |
| **I1 + guided stride-2 skip** | `dino3d_i1g_frz_pt_f100` | running on GPU2 since 2026-09-29 10:12 |
| I2: I1 + trainable raw-CT stem (+37k params) | `dino3d_i2_frz_pt_f100` | done: 0.8340 |

- The skip is the coarsest projected feature (256 ch at stride 16) delivered again at stride 2, by
  trilinear interpolation or by the frozen dino3d-trained upsampler. Zero trainable parameters are
  added (4,357,238 in all three I1 variants), so `guided − trilinear` isolates the upsampler and
  `guided − I1` is what it buys. The HU patch rides along as input channel 1 (`data.guide_hu`).
- Same 500-epoch recipe as I1/I2, so validation at any epoch (every 25) compares directly with the
  existing I1/I2 curves at that epoch. The I2 − I1 gap was already stable by epoch 100 (0.066 then,
  0.063 at 500), so epochs 100–150 give an early read; the paper numbers come at 500.
- Cost: 1.13 s/step vs about 1.05 for I1, 57 GB, about 60 h to epoch 500.
- Reading the outcome: near or above I2 → a reusable, encoder-agnostic alternative to a per-encoder
  stem; above I1 but well below I2 → partial; no gain over the trilinear control → the linear-probe
  gains do not survive a trained decoder.

Epoch 25 (first validation, one seed; `python -m scripts.upsampler_heads`):

| val mean Dice | I1 | I1 + tri | I1 + guided | I2 |
|---|---|---|---|---|
| epoch 25 | 0.5733 | 0.5825 | **0.6393** | 0.6582 |
| epoch 50 | 0.6388 | 0.6464 | **0.6772** | 0.7001 |
| epoch 75 | 0.6661 | 0.6749 | **0.6991** | 0.7286 |
| epoch 100 | 0.6766 | 0.6861 | **0.7348** | 0.7431 |
| epoch 125 | 0.6830 | 0.7034 | **0.7547** | 0.7551 |
| epoch 150 | 0.6986 | 0.7126 | **0.7710** | 0.7824 |
| epoch 175 | 0.7112 | 0.7270 | **0.7687** | 0.7918 |
| epoch 200 | 0.7200 | 0.7366 | **0.7746** | 0.7977 |
| epoch 225 | 0.7393 | 0.7427 | **0.7824** | 0.8039 |
| epoch 250 | 0.7446 | 0.7437 | **0.7875** | 0.7996 |
| epoch 275 | 0.7382 | 0.7518 | **0.7901** | 0.8066 |
| epoch 300 | 0.7508 | 0.7524 | **0.8048** | 0.8121 |
| epoch 325 | 0.7498 | 0.7575 | **0.8053** | 0.8198 |
| epoch 350 | 0.7610 | 0.7662 | **0.8102** | 0.8149 |
| epoch 375 | 0.7633 | 0.7646 | **0.8113** | 0.8248 |
| epoch 400 | 0.7666 | 0.7698 | **0.8176** | 0.8230 |
| epoch 425 | 0.7676 | 0.7719 | **0.8121** | 0.8299 |
| epoch 450 | 0.7692 | 0.7742 | **0.8148** | 0.8332 |
| epoch 475 | 0.7713 | 0.7733 | **0.8142** | 0.8328 |
| epoch 500 | 0.7707 | 0.7743 | **0.8156** | 0.8340 |
| **best** | 0.7713 (475) | 0.7743 (500) | **0.8176** (400) | 0.8340 (500) |

- trilinear − I1 +0.009 (thin +0.003, 23/46 up; ρ(Δ, mm) +0.22): the extra path alone does little.
- guided − trilinear +0.057 (thin +0.083, 42/46 up; thick +0.040, 51/69 up; ρ −0.42): the gain is
  the guidance and concentrates in thin classes, as in the probe.
- guided recovers 78% of I2 − I1 with zero trainable parameters (I2 adds 37,344).
- Per-class swings are ±0.1–0.3 this early (stomach −0.13, scapula_right +0.33 vs I1); gaps may
  still shrink with training (I2 − I1 went 0.085 → 0.063 from epoch 25 to 500).
- Epoch 50: the gaps shrink. guided − trilinear +0.031 (thin +0.045, 34/46 up; thick +0.022, 58/69
  up), guided − I1 +0.038 (63% of I2 − I1). trilinear − I1 stays small (+0.008). guided − I2 is
  about flat overall (−0.019 → −0.023) but widens on thin classes (−0.021 → −0.040): I1 catches up
  with both, while guided trails I2 by a roughly constant margin. Epoch 100 decides whether it holds.
- Epochs 75 and 100: the shrinkage was not a trend. guided − trilinear went +0.024 (75) then +0.049
  (100; thin +0.066, 39/46 up; thick +0.037, 56/69 up; ρ −0.37); guided − I1 +0.058 = 87% of
  I2 − I1, and guided matches I2 on thick classes (+0.000) while trailing on thin (−0.021).
  trilinear − I1 stays at +0.009 at every epoch. Single-epoch validation is noisy (guided moved
  +0.036 from 75 to 100, I1 +0.011), so the summary is the mean over epochs 25–100: guided − tri
  +0.040, guided − I1 +0.049, I2 − I1 +0.069 (guided recovers about 70% of the stem's gain).
- Epochs 125 and 150: the lead holds and grows. At 150, guided − trilinear +0.058 (thin +0.076,
  43/46 up; thick +0.047, 59/69 up; ρ −0.48), guided − I1 +0.072 = 86% of I2 − I1. Guided ties I2
  at 125 (0.7547 vs 0.7551) and at 150 already equals I1's best after 500 epochs (0.7710 vs
  0.7713). Guided ≥ I2 on thick classes at both epochs; the stem keeps a thin-class edge (−0.021,
  −0.039). trilinear − I1 +0.020 and +0.014. Mean over epochs 25–150: guided − tri +0.045,
  guided − I1 +0.057, I2 − I1 +0.072 (79%).
- Epoch 175: guided − trilinear +0.042 (thin +0.054, 40/46 up), guided − I1 +0.057, guided − I2
  −0.023 (thin −0.044, thick −0.009). trilinear − I1 has crept up to +0.016 (+0.017 at 200).
- 2026-09-30 10:44 the container restarted and killed both runs (guided mid-epoch 200, trilinear
  at 202). Both resumed at 10:49 with `--resume` (guided from epoch_0190.pt, trilinear from
  epoch_0200.pt; model, optimizer, scheduler and best Dice restored). One resume each: say so in
  the paper.
- Epochs 200–375 (status 2026-10-01): the guided lead is stable, not shrinking. Mean over these
  eight validations: guided − trilinear +0.044 (thin +0.061, thick +0.032, ρ −0.55), guided − I1
  +0.050, I2 − I1 +0.064 (78%), guided − I2 −0.014 (thin −0.029, thick −0.005). The trilinear
  control converges back to I1 (+0.001 at 375). At 375: guided − trilinear +0.047 with 45/46 thin
  classes up, ρ −0.73; guided ≥ I1 on all 46 thin classes; guided 0.811 already 0.04 above I1's
  500-epoch best (0.7713).
- **Final (both done 2026-10-02; best val Dice, one seed).** guided − trilinear +0.043 (thin +0.057,
  46/46 up; thick +0.034, 68/69 up; ρ −0.63); guided − I1 +0.046 (every one of 115 classes up);
  I2 − I1 +0.063, so guided recovers 74%. trilinear − I1 +0.003 (27/46 thin up): the extra path
  alone is nothing. guided − I2 −0.016 (thin −0.033, 2/46 up; thick −0.006). Mean thin Dice: I1
  0.714, tri 0.719, guided 0.775, I2 0.808. Over epochs 400–500 guided plateaued near 0.815 while
  I2 kept climbing, so the gap to I2 widened from −0.006 (epoch 400) to −0.018 (epoch 500).
  Not done yet: the test split.

### Next round (launched 2026-10-02; scope: 3D only, 2D AnyUp baseline dropped)

| # | experiment | where | status |
|---|---|---|---|
| 1 | test split (89 volumes), per case Dice + NSD 1/2 vox, for the four finished dino3d heads | GPU0, `scripts/upsampler_test_eval.py` → `results/upsampler/test_heads/` | running |
| 2 | label-supervised upsampler trained per encoder (sam_med3d, ctfm, ctclip): transfer upper bound | GPU0 after 1, `runs/upsampler/<enc>_label` | queued |
| 3 | probe with each encoder's own upsampler, merged into `stage2_<enc>.json` | GPU0 after 2 | queued |
| 4 | **I2 + guided skip** (`dino3d_i2g_frz_pt_f100`): does it stack on the stem? 4,394,582 params = I2 | GPU2, 500 epochs, 1.31 s/step, 71 GB | running since 09:50 |
| 5 | **SAM-Med3D I1 + guided** (`samMed3d_i1g_frz_pt_f100`), dino3d upsampler unchanged | GPU0 after 3, 250 epochs | queued |
| 6 | **SAM-Med3D I1** (`samMed3d_i1_frz_pt_f100`), the baseline for 5 | GPU3 (user OK), 250 epochs, 0.96 s/step × 1082 steps/epoch, 54 GB | running since 10:05, ~Oct 5 midday |

- SAM-Med3D runs at 128³ patches (its pretraining canvas at 1.5 mm): the 8³ token grid then sits
  at stride 16, as the interface needs, with no resize. Batch 1 × 6 = the dino3d effective batch;
  250 epochs of 2 × 128³ ≈ 1.19× the voxels of 500 epochs of 2 × 96³. 4,193,398 trainable
  parameters in both arms. CPU dry run passed (strides [16] / [16, 2], HU sensitivity > 0).
- Queues: `runs/upsampler/gpu0_oct2e.sh` (1–3, 5). Earlier versions: a/b hit eval OOM (four
  full-volume evals at once); c hung 10:50–13:05 waiting with `kill -0` on a finished eval that
  stayed a zombie (this container's PID 1 never reaps orphans); d was stopped because NSD sums in
  fp16 overflowed for large surfaces (fixed in `upsampler_test_eval.py` and `probe.py:surface_counts`;
  the probe's 96³ patches never reached the overflow, so its results stand). Log: `runs/upsampler/gpu0_oct2.log`. The GPU2 chain for 6 (`gpu2_after_i2g.sh`) was
  cancelled when GPU3 freed up; GPU2 is open after 4 (~Oct 5).

### Results of the 2026-10-02 round (status 2026-10-05 10:30)

- **I2 + guided** (done 09:10): best val 0.8424 vs I2 0.8340 (+0.008; thin +0.012, 38/46 up; thick
  +0.006). Ahead of I2 at 19 of 20 validations. The frozen upsampler stacks on a trainable stem.
- **SAM-Med3D, unseen encoder, in a trained head** (I1 at epoch 245/250 on GPU3; I1 + guided at
  190/250 on GPU0). Guided − I1 at every shared validation, epochs 25–175: +0.078, +0.055, +0.077,
  +0.055, +0.056, +0.041, +0.047. At 175: thin +0.058 (42/46 up), thick +0.039, ρ −0.47.
- **Test split** (89 volumes, paired bootstrap, 2000 resamples): I1 and trilinear complete; guided
  and I2 stopped at 35/89 (case 36 OOMs whenever two evals share GPU0; rerun alone). On the 35
  shared cases: guided − trilinear +0.042 [0.037, 0.047], thin +0.057 [0.048, 0.066]; trilinear − I1
  +0.005 [0.001, 0.008]; guided − I2 −0.018 [−0.023, −0.013]. NSD at 3 mm barely moves on full
  volumes (0.905–0.912; guided − trilinear +0.003 [−0.003, +0.008]): report NSD at 1.5 mm too.
- **Per-encoder upsamplers** came out *worse* than the transferred dino3d one (SAM-Med3D thin +0.039
  vs +0.124; CT-FM +0.050 vs +0.084; CT-CLIP equal on thin, better on thick). Confounded: their
  throwaway linear heads barely learned (SAM-Med3D val loss 2.54 vs trilinear 2.72; dino3d 1.22 vs
  1.55), the same feature-scale problem the probe had (SAM-Med3D std 0.14) because label training
  does not normalize features. Not a valid upper bound yet; retrain with the probe's normalization
  (global std, log-prior bias).
- **GPU access lost in the container (2026-10-05, first seen ~10:00):** `nvidia-smi` fails with
  "Failed to initialize NVML", new processes see 0 CUDA devices; running processes keep their
  contexts and train on. Needs a container restart from the host.
- **2026-10-05 13:32 container restart** (GPU access back, all four GPUs empty). SAM-Med3D I1 had
  finished at 11:57 (best val 0.7184 at epoch 250). SAM-Med3D I1 + guided was in its epoch-200
  validation, before the epoch-200 checkpoint, so it resumed from epoch 190 (≈3 h lost); ETA
  ~Oct 6 09:30. GPU2: guided and I2 test evals, one at a time (`runs/upsampler/gpu2_oct5.sh`).
  GPU3: per-encoder upsamplers with `--norm-feats` plus a dino3d control (`runs/upsampler/gpu3_oct5.sh`).
  Feature std per encoder: dino3d 1.03, SAM-Med3D 0.14, CT-FM 33.1, CT-CLIP 1.00 — so the first
  round's SAM-Med3D and CT-FM heads were mis-scaled in opposite directions.

### Round before the node wipe (queued 2026-10-05 ~14:20; wipe Wed 7 Oct morning)

Backup list and restore/relaunch commands: `docs/BACKUP_AND_RESTORE.md`.

| GPU | job | state at the wipe |
|---|---|---|
| 0 | SAM-Med3D I1 + guided to 250 epochs (~Oct 6 09:30), then its test eval (`gpu0_oct5.sh`) | done |
| 2 | dino3d I2 test eval, then **SAM-Med3D I1 + trilinear** control (`samMed3d_i1t_frz_pt_f100`, 128³, 250 epochs; `gpu2_oct5b.sh`) | ~epoch 80–90; resume after the reset |
| 3 | after the normalized per-encoder upsamplers: eight upsampler ablations on dino3d, then probes of all of them and the transfer matrix on all four encoders, then the SAM-Med3D I1 and dino3d I2 + guided test evals (`gpu3_oct5b.sh`) | done by Oct 6 morning |

Ablations: each changes one thing in the recipe of `runs/upsampler/dino3d_label` (label objective,
4000 steps, `--val-batches 12`), output `runs/upsampler/dino3d_abl_<name>`:

| name | change | question |
|---|---|---|
| `noct` | `--no-ct`: the guidance CNN sees a constant image, so the weights depend only on the neighbour geometry | is the gain the CT, or a learned interpolation kernel? |
| `seed1`, `seed2` | `--seed 1/2` | probe-level seed noise for every other row |
| `r2` | `--radius 2` (125 neighbours instead of 27) | neighbourhood size |
| `1win` | `--single-window` (only −1000..1000 HU, no soft-tissue window) | CT windowing |
| `n25`, `n75` | `--n-train-vol 25/75` (of 238 labelled volumes) | how many labelled volumes the upsampler needs |
| `fkeys` | `--feature-keys` (keys also read the features, channel-count-invariant) | CT-only attention vs also the features |

The transfer matrix probes each `<encoder>_label_norm` upsampler on every encoder (the diagonal
is the per-encoder upper bound).
