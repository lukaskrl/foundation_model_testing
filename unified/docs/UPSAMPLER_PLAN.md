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
