# The pyramid contract, the adapters, and the experiment arms

When comparing pretrained encoders, the experimentally interesting variable is the
**encoder**. Everything else — decoder, channel widths, loss, optimizer, augmentation,
batch size, patch size — is held constant. This document specifies the constant parts:
the feature contract every backbone must satisfy, the per-backbone adapter that fills
it, the shared heads that consume it, and the **arm** structure (N / S / B / W) that
separates a representation probe from a matched-capacity control, a matched-input
control, and a best-effort fine-tune.

All parameter counts here were measured on the current code (`weights=None`, CPU) with
`scripts/`-independent construction; see [§6](#6-measured-adapter-budgets).

---

## 1. The contract

Every backbone implements `BackboneInterface` (`unified/models/seg_model.py`):

```python
def forward_features(x: Tensor[B,1,D,H,W]) -> List[Tensor]   # exactly 5 tensors
```

```python
EXPECTED_STRIDES  = (1, 2, 4, 8, 16)          # finest first
EXPECTED_CHANNELS = (32, 64, 128, 256, 512)
NUM_LEVELS        = 5
```

| Level | Stride | Channels | Shape at 96³ input |
|---|---|---|---|
| `feats[0]` | 1  | 32  | (B, 32, 96, 96, 96) |
| `feats[1]` | 2  | 64  | (B, 64, 48, 48, 48) |
| `feats[2]` | 4  | 128 | (B, 128, 24, 24, 24) |
| `feats[3]` | 8  | 256 | (B, 256, 12, 12, 12) |
| `feats[4]` | 16 | 512 | (B, 512, 6, 6, 6) |

Ordering is **finest first**, matching the deep-supervision weights
(`ds_weights = [1.0, 0.5, 0.25, 0.125]` → strides 1, 2, 4, 8).
`assert_contract` checks a feature list against the contract, but note **it is not
called on the training path** — only `scripts/verify_setup.py` invokes it. In practice a
violation still fails loudly, because the head's `torch.cat` against a mismatched skip
raises immediately; the assertion is a sharper error message, not the thing standing
between you and a silent mismatch. Run `verify_setup` for any new or edited adapter.

### Why `{1, 2, 4, 8, 16}`

- **CT-FM** and **VISTA-3D** (`SegResEncoder`) natively emit exactly these five
  strides, so the strongest reference encoders need **zero** spatial resampling.
- **Swin** families natively reach `{2, 4, 8, 16, 32}`. Adding stride 32 would let
  Swin pass through untouched but would force every conv encoder to invent a level it
  does not own. Dropping stride 32 favours conv-family fidelity; Swin loses only its
  coarsest level and gains a cheap conv stem at stride 1.
- A plain **ViT** at patch 16 owns exactly one stride and cannot produce stride 32
  without upsampling something coarser, so stride 32 would buy it nothing.

---

## 2. Encoder / adapter split

```
raw input ─► [pretrained encoder] ─► native features
                                          │
                                          ▼
                                  [trainable adapter]      ← self.adapter
                                          │
                                          ▼
                                 contract pyramid (5 levels)
                                          │
                                          ▼
                                     [shared head]
                                          │
                                          ▼
                                        logits
```

Two hard rules:

1. **All trainable post-encoder modules live under `self.adapter`.** `freeze_encoder()`
   freezes every backbone parameter *except* those reachable through `self.adapter`.
   This is the only reason the channel-projection convs train at all under a frozen
   run.
2. **Adapters that split the forward** provide `encoder_forward(x)` and
   `adapter_forward(native, input_shape)`. On the **frozen** path `SegModel` uses the
   split to run the encoder under `no_grad`, detach, and run only the adapter with
   gradients — cheaper and unambiguous. Unfrozen runs always call `forward_features`
   directly.

   All 12 encoders (and `StemFusionBackbone`) implement the split. An adapter that
   omits it falls back to `forward_features` under `no_grad` with everything detached,
   which trains *nothing* after the encoder — a frozen run would then be head-only and
   not comparable with the rest of the suite. Only the dead `stunet` stub is in that
   state; implement the split for anything new.

`trainable_modules()` additionally reports which modules must stay in `train()` mode
when the encoder is frozen. The base rule returns `[self.adapter]`; **two backbones
override it**, and both overrides exist to stop a frozen run from differing from its
unfrozen twin by more than the freeze:

- `StemFusionBackbone` restores the *inner* neck too — otherwise a frozen Arm S run
  would leave a module in eval that the matching Arm N run trains, a difference between
  the arms having nothing to do with the stem.
- `DinoBackbone` in `vit_adapter` mode restores every child of `self.vit_adapter`
  except the frozen `vit_model`. Its ~25.9 M trainable Injector/Extractor parameters
  live *inside* that wrapper rather than under `self.adapter`, so the base rule left
  them in eval for the whole of a frozen run. With `drop_path_rate: 0.1` that silently
  disabled DropPath for the `frz_*` rows while the `ft_*` rows kept it — a
  regularization difference sitting inside Arm B's frozen-vs-finetuned contrast, which
  is the one comparison that arm exists to support.

**One deliberate exception to the `self.adapter` rule.** `pyramid_mode="vit_adapter"`
bundles its trainable Injector/Extractor blocks *inside* the frozen ViT wrapper, so it
overrides **both** `freeze_encoder()` (to freeze exactly the pretrained ViT) and
`trainable_modules()` (above). Its frozen-trainable budget is therefore **not**
`self.adapter` — see [§6](#6-measured-adapter-budgets).

---

## 3. Adapter modes

`pyramid_mode` selects how an encoder's native features are lifted onto the contract.
Only some modes leave a frozen run interpretable as a representation probe.

| Mode | How the five levels are produced | Probe-honest |
|---|---|---|
| `native` | encoder's own hierarchy; 1×1 channel projection per level (plus one strided conv, or one stem, where a level is missing) | **yes** |
| `upsample` | every level is a channel-projected, trilinearly resampled copy of ONE bottleneck tensor | **yes** |
| `layerwise` | each level reads a *different* transformer block, earliest tap → finest level (the DPT / UNETR convention) | **yes** |
| `multiscale` | the encoder is run at several input canvases; each level is assigned to one pass | **yes** |
| `spm` | fine levels (strides 1–8) come from a fresh `SpatialPriorModule3D` on the **raw input**; only stride 16 comes from the encoder | no |
| `vit_adapter` | 3D ViT-Adapter: SPM bidirectionally fused with the ViT through deformable Injector/Extractor blocks | no |

`NON_PROBE_MODES = {"spm", "vit_adapter"}` in `scripts/gen_lowshot_configs.py` encodes
this, and the low-shot manifest carries a `probe_comparable` column derived from it.

**Why `spm` and `vit_adapter` disqualify a probe.** Both put a randomly-initialised
conv branch on the raw input volume. With the encoder frozen, that branch is a
trainable segmentation network in its own right, so the frozen number stops being a
statement about the pretrained representation. `spm` is retained only for backward
compatibility; the generic `StemFusionBackbone` wrapper (Arm S) supersedes it by
*fusing* a stem into the encoder's own levels instead of *replacing* them.

**Why `layerwise` is the ViT default over `upsample`.** `upsample` gives all five
levels to one tensor, so the decoder's skips are re-projections of the tensor they skip
into and carry nothing new — while the other blocks are computed and discarded anyway.
`layerwise` is parameter-matched to `upsample` (identical count, see §6), so the two
form a clean single-variable pair. What `layerwise` does **not** buy is resolution:
every block emits the same token grid, so the finest level is still a resampled
stride-16 map and the tokenizer's compression is untouched.

---

## 4. How each encoder fills the contract

| Backbone | Native levels | Mode | Contract sourcing |
|---|---|---|---|
| `ctfm` | 5 @ `(1,2,4,8,16)`, ch `(32,64,128,256,512)` | `native` | 1:1, channel-only 1×1 convs |
| `vista3d` | 5 @ `(1,2,4,8,16)`, ch `(48,96,192,384,768)` | `native` | 1:1, channel-only 1×1 convs |
| `suprem_unet` | 4 @ `(1,2,4,8)` | `native` | levels 0–3 via 1×1; stride 16 via strided conv on the adapted stride-8 |
| `suprem_segresnet` | 4 @ `(1,2,4,8)` | `native` | same as `suprem_unet` |
| `voco_b` / `voco_h` | 5 @ `(2,4,8,16,32)` | `native` | Swin levels 0–3 → strides 2–16 via 1×1; stride 1 from a light conv stem on raw input |
| `suprem_swinunetr` | 5 @ `(2,4,8,16,32)` | `native` | same as `voco_b` |
| `ctclip` | 1 bottleneck (CTViT, run on a 480² resized, depth-padded copy) | `upsample` | all levels resampled from the bottleneck |
| `sam_med3d` | 1 bottleneck (input resized to 128³ so `pos_embed` needs no interpolation) | `upsample` | all levels resampled from the bottleneck |
| `dino3d` | 1 token grid @ stride 16, ViT-L/16, 24 blocks | `layerwise` | blocks `(5,11,17,23)` → one tap per level, earliest to finest |
| `merlin` | 5 I3D-ResNet-152 stages; in-plane strides `(2,4,8,16,32)`, **depth stride half the in-plane stride at every level** | `native` | `conv1`/`layer1`/`layer2`/`layer3` → strides 2–16 via 1×1; stride 1 from a `ConvStem` on raw input; `layer4` discarded |
| `biomedparse` | 4 2D FocalNet stages, HW strides `(4,8,16,32)`, full depth | `native` | strides 4–16 via depth-axis 3D conv + depth pool + 1×1; strides 1–2 from a 3D conv stem on raw input |

Two encoders resize the input inside the adapter, so patch **FOV** is constant across
the suite but effective **resolution** is not: `ctclip` (480² in-plane, depth-padded)
and `sam_med3d` (128³). This is a per-model property of the pretrained interface, not a
recipe knob, and it must be stated in any write-up.

**`merlin` and the depth-geometry problem.** Merlin's I3D inflation gives `conv1`
`time_stride=1` against an in-plane stride of 2, so the depth stride is half the
in-plane stride at every level. That is deliberate: Merlin was pretrained at
1.5 × 1.5 × **3.0** mm, so the two anisotropies cancel and one feature voxel is
physically isotropic *at its own spacing*. This framework locks `data.spacing` to
1.5 mm isotropic, so they no longer cancel, and the factor of 2 has to go somewhere.
`input_mode` decides where:

| `input_mode` | Depth reduction | Encoder sees | Native levels |
|---|---|---|---|
| `native_spacing` (default) | input depth pooled ×2 **before** the encoder | 1.5 × 1.5 × 3.0 mm — its pretraining geometry | land exactly on cubic contract strides, no per-level resampling |
| `isotropic` | each level's depth pooled ×2 **after** the encoder | 1.5 mm isotropic — depth at 2× the pretrained resolution | need a depth-only ×2 pool each |

Pooling has no parameters, so both modes have the identical 722,784-parameter adapter
and differ in exactly one thing. `configs/models/merlin_isotropic.yaml` is the A/B
partner, registered as the `merlinISO` ablation row. Full-resolution depth still
reaches the decoder through the stride-1 stem in both modes.

Discarding `layer4` costs Merlin something the Swin analogy understates: `layer4` →
`avgpool` → `contrastive_head` is the tap the vision-language objective actually acted
on, so the coarsest level the decoder sees is one stage shallower than the one the loss
shaped. The modules are kept (so the checkpoint loads strictly and the CT-RATE
classification track can still pool that tap) but are never executed in the
segmentation forward.

`biomedparse` is the least apples-to-apples member: FocalNet runs per axial slice with
no native depth context, and only the adapter's depth mixers and stem inject any.

---

## 5. Heads

Two heads are registered in `HEAD_REGISTRY` (`unified/models/head.py`); `head.name`
selects one from config. Both consume the contract and nothing else, so they are
backbone-independent by construction.

```python
head(x_in: Tensor, feats: List[Tensor]) -> Tensor | List[Tensor]
```

### `unified_seg_head` (default) — U-Net decoder

```
d4 = f_4                                    # stride 16
d3 = UpBlock(d4, f_3)                       # stride 8
d2 = UpBlock(d3, f_2)                       # stride 4
d1 = UpBlock(d2, f_1)                       # stride 2
d0 = UpBlock(d1, f_0)                       # stride 1
out = 1×1 conv (32 → 118)                   # stride-1 logits
```

`UpBlock` is MONAI's `UnetrUpBlock` (ConvTranspose ×2 + concat skip + 2 residual conv
blocks). `num_classes = 118` (117 structures + background). Because real features fill
every skip down to stride 1, the head reads **only** the contract — it has no
`enc0`-style fresh conv on the raw input.

### `mask2former_head` — query-based decoder

A denser query-based alternative, selected purely by config. Registered behind the
same signature, so it is a drop-in at matched budget.

### Deep supervision

With `deep_supervision: true` and in `.train()` mode, the head returns
`[logits_s1, logits_s2, logits_s4, logits_s8]` (finest first), each from a 1×1
`UnetOutBlock` on the corresponding decoder activation. Weighted by
`head.ds_weights = [1.0, 0.5, 0.25, 0.125]`. This is model-independent and applies
uniformly, so it does not break the fairness lock.

**Shared head size: 8,674,072 parameters** — identical for every backbone, since it
depends only on the contract.

---

## 6. Measured adapter budgets

Measured on current code, `weights=None`. `trainable when frozen` is the number that
decides probe honesty: parameters still receiving gradients after `freeze_encoder()`.

| Backbone (config) | Mode | Adapter module | Trainable when frozen |
|---|---|---:|---:|
| `voco_b` | `native` | 291,680 | 291,680 |
| `suprem_swinunetr` | `native` | 291,680 | 291,680 |
| `ctfm` | `native` | 351,168 | 351,168 |
| `sam_med3d` | `upsample` | 382,912 | 382,912 |
| `ctclip` | `upsample` | 509,888 | 509,888 |
| `ctclip_layerwise` | `layerwise` | 509,888 | 509,888 |
| `ctclip_multiscale` | `multiscale` | 509,888 | 509,888 |
| `vista3d` | `native` | 525,760 | 525,760 |
| `biomedparse` | `native` | 609,600 | 609,600 |
| `merlin` | `native` | 722,784 | 722,784 |
| `merlin_isotropic` | `native` | 722,784 | 722,784 |
| `dino3d_layerwise` | `layerwise` | 1,017,792 | 1,017,792 |
| `dino3d_upsample` | `upsample` | 1,017,792 | 1,017,792 |
| `voco_h` | `native` | 1,075,040 | 1,075,040 |
| `suprem_segresnet` | `native` | 1,094,080 | 1,094,080 |
| `suprem_unet` | `native` | 1,224,640 | 1,224,640 |
| *any* Arm S row | `<mode>+stem` | its Arm N twin **+ 349,584** | twin **+ 349,584** |
| `dino3d_vitadapter` (Arm B) | `vit_adapter` | 1,247,168 | **25,857,888** |

Three properties this table is meant to demonstrate:

- **Probe adapters are thin and comparable.** Across the 12 probe-comparable encoders
  the budget spans 291,680 → 1,224,640 — a 4.2× spread against a shared head of
  8.67 M, so no encoder's frozen result is explained by adapter capacity.
- **The A/B pairs are exactly parameter-matched.** `dino3d_layerwise` vs
  `dino3d_upsample` are both 1,017,792; all three `ctclip` modes are 509,888; both
  `merlin` input modes are 722,784. Each pair differs only in *which* tensor a level
  reads, or *where* a parameter-free resample happens.
- **Arm S adds a bit-identical constant.** Verified on **all 12** encoders: total
  parameters and frozen-trainable parameters each rise by exactly **349,584**
  (348,624 SPM + 960 GroupNorm affine), on hierarchical CNNs, Swin, an inflated-2D
  ResNet, a per-slice 2D backbone and columnar ViTs alike. That constant is what makes
  Arm S a controlled comparison rather than per-backbone tuning, and it is why the arm
  is *generated* from each Arm N config rather than hand-written per encoder.
- **`vit_adapter` is 21× the largest probe adapter** (25.86 M vs 1.22 M) and is why it
  cannot sit in the probe column.

---

## 7. The experiment arms

**Four** arms, tagged in `configs/lowshot/MANIFEST.csv` via the `arm` column. Each arm
runs the full 4 conditions × 5 data fractions, so `frozen`-vs-`finetuned` and
`pretrained`-vs-`scratch` stay meaningful **within** an arm. Each holds the encoder
fixed and varies exactly one thing, so each is a controlled subtraction against its
Arm N twin.

| Arm | What it varies | What it controls for | Probe | Members |
|---|---|---|---|---|
| **N** | nothing — the encoder **as delivered** | — | **yes** | all 12 encoders (`native` ×9, `upsample` ×2, `layerwise` ×1) + the `dino3dU` and `merlinISO` ablations |
| **S** | + a shared `SpatialPriorModule3D` fused into levels 0–3 | **capacity** for fine detail | no | all 12 encoders |
| **B** | best-effort dense recipe: 3D ViT-Adapter, SPM bidirectionally fused | — (upper bound) | no | `dino3d_vitadapter` |
| **W** | the HU window, forced to one of two shared values | the **input** interface | no | 18 cells (12 encoders × 2 windows − 6 no-ops) |

S and W are deliberately **orthogonal, not crossed**: crossing them would quadruple the
matrix while identifying nothing the two separate subtractions do not already.

**Never merge S, B or W rows into the Arm N column.** Read arms via the `arm` column;
the launcher's `PROBE=True` filter enforces it. `probe_comparable` means "belongs in the
headline Arm N column" and is `False` on three independent grounds — a non-probe pyramid
mode, a fused stem (guarded by an explicit `not stem_fused` check, since the
adapter-string test alone would not catch `"native+stem"`), or a forced window. An Arm W
row *is* a clean thin-neck probe; it is excluded because it probes a **different input**,
and merging it would compare encoders across two normalizations at once.

### Arm N — the measurement

Arm N compares encoders as their authors shipped them: a hierarchical CNN hands the
decoder five genuinely different tensors, a columnar ViT hands it five views of one.
That gap is real and worth measuring, but it conflates two things — how good the
pretrained semantics are, and whether the architecture happens to route fine detail to
a decoder.

### Arm S — the matched-stem control

`StemFusionBackbone` (`unified/models/stem_fusion.py`) bolts the **same**
`SpatialPriorModule3D` onto *any* backbone and adds it into the fine levels:

```
fused_k = ReLU(GroupNorm(inner_k + spm_k))    k = 0..3   (strides 1, 2, 4, 8)
fused_4 = inner_4                             (stride 16, untouched)
```

The deliverable is not the level Arm S reaches, it is the **delta**:

```
compensable(m) = gap_N(m) - gap_S(m)     missing fine detail — a stem fixes it
irreducible(m) = gap_S(m)                missing pretrained mid-level features
```

Both arms must exist for the same (condition, fraction) cell for the subtraction to be
defined.

**Read the delta against what Arm N's fine levels already were.** `compensable ≈ 0`
means three different things depending on the encoder, which is why the manifest carries
an `arm_n_fine_source` column. The values are verified against the code, not asserted —
the numbers below are the raw-input stem parameters already present in each Arm N
adapter:

| `arm_n_fine_source` | Encoders | Arm N raw-stem params | What Arm S contrasts |
|---|---|---:|---|
| `pretrained` | `ctfm`, `vista3d`, `suprem_unet`, `suprem_segresnet` | 0 | pretrained-stem vs **+** fresh-stem — a CNN's own stride-1 level already *is* a two-conv stem on raw voxels, so the honest contrast is not stem-vs-no-stem |
| `fresh_stem_s0` | `voco_b`, `voco_h`, `suprem_swinunetr`, `merlin` | 28,640 | one fresh stem vs **two**. `compensable ≈ 0` here means "detail was already routed", **not** "this encoder routes detail well" |
| `fresh_stem_s0s1` | `biomedparse` | 84,992 | two fresh stems already (levels 0 *and* 1) vs three |
| `resampled_bottleneck` | `ctclip`, `sam_med3d` | 0 | **no raw-input wire at all** vs one |
| `resampled_tokens` | `dino3d` | 0 | same |

Expect `compensable ≈ 0` for hierarchical encoders and large for the `resampled_*`
group — those four have no path finer than their bottleneck, and the shared head has no
raw-input branch of its own (`enc0` was deliberately removed, §9), so their Arm N score
is bounded by interpolation rather than by representation quality.

`GroupNorm` (8 groups) replaces upstream `SyncBatchNorm` so the module is independent
of distributed training and batch size. Wrapping a mode that already sources *most* of
the contract from raw voxels is refused (`_ALREADY_HAS_STEM = {"spm", "vit_adapter"}`) —
stacking two full SPMs would double the added capacity and destroy the constant the arm
depends on. The small `fresh_stem_*` stems above are a different case and are wrapped
normally: the added capacity is still the same constant, and the contrast is simply one
fresh stem vs two. Note the guard reads the *effective* mode via
`_effective_pyramid_mode()`, which checks both the backbone and its inner adapter —
`ctclip` and `sam_med3d` record `pyramid_mode` only on the adapter, so reading the
backbone attribute alone silently returned `None` and would have let a `spm` variant of
exactly those two through.

**Scope: all 12 encoders.** It began as a two-model pilot (`ctfm`, `dino3d` — the
maximal-contrast pair), but `compensable` is meant to be a property of an architecture
*family*, and with n=1 per family there is no way to separate "columnar ViTs need a
stem" from "dino3d needs a stem". Full coverage gives every family at least two members
and, more importantly, covers `ctclip` and `sam_med3d` — the two encoders with the
largest expected delta, which the pilot did not include. Arm S rows are generated from
each Arm N config plus `{stem_fusion, stem_inplanes}` rather than from hand-written
`<name>_stem.yaml` twins, for the same reason the wrapper is one generic module: twelve
hand-maintained twins would be twelve chances to drift.

### Arm B — the best-effort upper bound

The faithful plain-ViT dense-prediction recipe, so the pretrained representation feeds
every scale instead of only stride 16. It answers "if you are going to use a plain ViT
densely, what is available and what does it cost" — a different and legitimate
question from Arm N's. Report it beside the `dino3d` Arm N row, never inside it.

### Arm W — the matched-input control

Every other arm feeds each encoder the intensity normalization it was pretrained with,
on the principle that the window is part of the encoder's input interface rather than a
tunable knob. That principle is right, but it is not free, and the cost is wildly
uneven. Measured on 12 TotalSegmentator subjects (raw HU, post-`Spacingd` /
`CropForegroundd`), the fraction of each class's voxels **clipped** by its own window:

| Window | Encoders | Classes >90 % clipped | >50 % | Foreground voxels clipped |
|---|---|---:|---:|---:|
| `[-175, 250]` | `voco_b/h`, `suprem_unet/segresnet/swinunetr` | 6 / 116 | 39 | **27.6 %** |
| `[-1000, 1000]` | `merlin`, `biomedparse` | 0 / 116 | 0 | 0.7 % |
| `[-1024, 2048]` | `ctfm` | 0 / 116 | 0 | 0.1 % |

Five of the twelve Arm N encoders see a whole-body task through a soft-tissue window
that flattens the lungs outright (97–99 % of lung-lobe and trachea voxels) and a third
of the skeleton. Nothing downstream recovers what was clipped before the first conv, so
a cross-encoder ranking read off Arm N alone cannot separate "worse representation"
from "narrower input window".

Arm W separates them. A linear window is `clamp → affine`; the affine half is undone by
any layer that can learn, the clamp half is permanent. **That makes the prediction
arm-dependent**, which is exactly why it is worth running:

- `ft_*` — the encoder adapts, so the affine shift costs ~nothing and any remaining gap
  is the clipped information. The cleanest cross-encoder number available.
- `frz_*` — the frozen first layers *cannot* absorb the affine shift, so a gap here
  mixes lost information with distribution mismatch. For an encoder shipping a narrow
  recipe the frozen probe has **no unconfounded form**, because the information loss is
  a property of the delivered model. That is a legitimate finding, but it is the
  sentence "this encoder's shipped preprocessing discards most of the lung signal", not
  "this encoder's representation is weak".

Two forced windows give the full 2×2 on every encoder rather than a one-way test, and
the **cross-over** is what identifies the mechanism:

```
window_cost(m, w) = score(m, w) - score(m, native)

narrowing costs ctfm ≈ what broadening gains voco   -> INFORMATION; Arm N ranking
                                                       is confounded, report both
broadening HURTS voco while narrowing hurts ctfm    -> input fidelity dominates;
                                                       the native window is right
```

`wshared` is expressed as *"delete the per-encoder override"* so it equals `base.yaml`'s
window by construction rather than by a duplicated literal that could drift. `wnarrow`
forces the SuPreM / VoCo soft-tissue window onto encoders that never saw it. Six of the
24 cells are no-ops (`ctfm × wshared`, and the five soft-tissue encoders × `wnarrow`)
and are detected *behaviourally* — an absent `mode:` key means `range` — then skipped,
because for those the Arm N row **is** that cell and `window_cost = 0` by construction.

`axcodes` is deliberately **not** touched: orientation is a separate part of the input
interface, and `merlin` / `ctclip` are geometrically wrong under anything but `SRA`.
Cost is config-only — the window runs *after* the disk cache and
`preprocessing_fingerprint` hashes only spacing / crop margin / `num_classes`, so every
window variant shares the one existing cache.

---

## 8. What is fair, and what is an honest artifact

**Locked by the config loader.** `train:` is globally locked — a per-model config that
touches it raises `ConfigError`. So optimizer, LR, schedule, loss, AMP, gradient
clipping and epoch budget are identical across every run. `head:`, `data:` and `eval:`
deep-merge, which is what lets a *deliberate* protocol run change FOV or
`train_fraction`; leave them at `base.yaml` defaults for any cross-backbone comparison.

**Fair:** same dataset and split, same resampling, same patch size and sampler, same
loss, same schedule, same decoder architecture and init scheme, same contract, same
uniform batch (bs 2 × 2 accumulation = 8 patches) across the low-shot matrix.

**Asserted, not assumed:** every adapter's checkpoint load. `strict=False` is required
(each release carries decoders / SSL heads / text towers this framework has no place
for) but it also swallows a key-layout mismatch that leaves the encoder partly or wholly
random while the run still reports `pretrained: true`. `assert_encoder_loaded`
(`backbones/_loading.py`) is the single standard: **every parameter and persistent
buffer of the constructed encoder must be populated**. Guard strength used to vary from
`strict=True` down to no check at all, which meant the encoders most likely to fail
silently were ranked beside the ones that could not. Measured against the real released
checkpoints all twelve load 100 % of their keys, so the guard costs no legitimate load.

**Deliberately per-model:** the HU window and orientation (`model.preprocessing`). The
normalization a model was pretrained with is part of its input interface, not a tunable
knob, and matching it is a defensible default — but it is **not** unambiguously the fair
choice, and this document used to claim it was. For five encoders the pretrained window
clips 27.6 % of foreground voxels and flattens the lungs entirely (Arm W, §7), and no
representation recovers information destroyed before the first conv. Treat the window as
a *factor with a measured cost*, not a settled question: report `window_cost` beside any
cross-encoder ranking. Orientation is genuinely per-model and is not varied — `merlin`
and `ctclip` are geometrically wrong under anything but `SRA`.

**Honest artifacts, to be reported not hidden:**

- Pretraining budgets and supervision differ enormously (CT-CLIP: ~50 k volumes +
  reports; VoCo: ~160 k unlabeled; SuPreM / VISTA: voxel labels).
- `ctclip` and `sam_med3d` resize inside the adapter, so effective resolution differs
  even though FOV does not.
- `biomedparse` is 2D per-slice; its depth context is entirely adapter-supplied.
- Adapter budgets differ by up to 4.2× within the probe column (§6).
- The stride-1 level is a fresh stem for the Swin family, `merlin` and `biomedparse`,
  since none owns one natively — see the `arm_n_fine_source` table in §7, which is the
  key for reading `compensable`. `biomedparse` goes further and fills strides 1 **and**
  2 from fresh convs (84,992 params, 2 of 5 levels) while still carrying
  `probe_comparable=True`, whereas `spm` is disqualified for filling 4 of 5. That line
  is a threshold, not a principle; the column reports it so a reader can apply their own.
- A single LR (2e-4) covers both frozen probes (0.3–1.2 M trainable) and full finetunes
  (100 M+). Uniform, but uniform is not the same as unbiased across that spread.
- Only the low-shot matrix is batch-matched. `model.batch_size`, `model.grad_accum_steps`
  and `model.amp` sit in the *unlocked* `model:` block, so the headline
  `configs/models/*.yaml` sweep runs each encoder at its own batch and is a separate
  table.

---

## 9. Deprecated — removed from this framework

- **The 4-level `{4, 8, 16, 32}` / `{64, 128, 256, 512}` contract.** Superseded by the
  5-level contract above. It octave-shifted conv encoders (discarding their stride-1
  and stride-2 features) and left the fine levels to fresh convs on the raw input,
  which meant a frozen run measured those convs.
- **`enc0` / `enc0_down2`** — the head's fresh `UnetrBasicBlock` on the raw image and
  its strided companion. Real features now fill every skip.
- **The frozen `PyramidNeck`-inside-the-frozen-backbone bug.** Channel adapters were
  registered as submodules of the frozen encoder and stayed at random init for whole
  runs. Fixed by the `self.adapter` rule (§2).
- **Unguarded checkpoint loads.** `voco` and `suprem_swinunetr` checked only the
  *unexpected* list, `ctclip` and `vista3d` checked nothing, and `sam_med3d`'s check
  suppressed itself whenever any missing key mentioned `rel_pos`. A wrong-prefix
  checkpoint constructed silently into a fully random encoder labelled
  `pretrained: true`. Fixed by `assert_encoder_loaded` (§8).
- **Arm B frozen runs left 25.9 M trainable parameters in eval mode.** `DinoBackbone`
  overrode `freeze_encoder()` but not `trainable_modules()`, so DropPath was off in
  `frz_*` and on in `ft_*`. Fixed by the second override (§2).
- **The Arm S stem-stacking guard read the wrong attribute.** It checked
  `getattr(inner, "pyramid_mode")`, which is `None` for `ctclip` and `sam_med3d` (they
  record it on their inner adapter), so a `spm` variant of either would have been
  wrapped and stacked two SPMs. Fixed by `_effective_pyramid_mode()` (§7).
- **STU-Net** (`stunet_small` / `_base` / `_large` / `_huge`). Registered but not
  implemented: `_build_stunet_encoder` raises `NotImplementedError`, it needs the
  upstream repo vendored plus a PyTorch 1.10 / nnU-Net V1 environment, and it is not
  part of the 11-encoder benchmark. Treat the configs and `requirements-stunet.txt` as
  dead unless someone finishes the adapter.
- **`docs/PYRAMID_REFRACTOR.md`** — the migration note for the 4→5 level change. Its
  still-live content (native pyramid inventory, contract rationale, adapter strategy,
  measured budgets) is folded into this document; the historical before/after framing
  was dropped.
