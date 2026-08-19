# The pyramid contract, the adapters, and the experiment arms

When comparing pretrained encoders, the experimentally interesting variable is the
**encoder**. Everything else — decoder, channel widths, loss, optimizer, augmentation,
batch size, patch size — is held constant. This document specifies the constant parts:
the feature contract every backbone must satisfy, the per-backbone adapter that fills
it, the shared heads that consume it, and the **arm** structure (N / S / B) that
separates a representation probe from a best-effort fine-tune.

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
`forward_features` is shape-asserted against the contract on every call, so a
mis-specified adapter fails loudly rather than training into a silent mismatch.

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
when the encoder is frozen. `StemFusionBackbone` overrides it so the *inner* neck is
restored too — otherwise a frozen Arm S run would leave a module in eval that the
matching Arm N run trains, which is a difference between the arms having nothing to do
with the stem.

**One deliberate exception.** `pyramid_mode="vit_adapter"` bundles its trainable
Injector/Extractor blocks *inside* the frozen ViT wrapper, so it overrides
`freeze_encoder()` to freeze exactly the pretrained ViT. Its frozen-trainable budget is
therefore **not** `self.adapter` — see [§6](#6-measured-adapter-budgets).

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
| `ctfm_stem` (Arm S) | `native+stem` | 700,752 | 700,752 |
| `dino3d_stem` (Arm S) | `layerwise+stem` | 1,367,376 | 1,367,376 |
| `dino3d_vitadapter` (Arm B) | `vit_adapter` | 1,247,168 | **25,857,888** |

Three properties this table is meant to demonstrate:

- **Probe adapters are thin and comparable.** Across the 12 probe-comparable encoders
  the budget spans 291,680 → 1,224,640 — a 4.2× spread against a shared head of
  8.67 M, so no encoder's frozen result is explained by adapter capacity.
- **The A/B pairs are exactly parameter-matched.** `dino3d_layerwise` vs
  `dino3d_upsample` are both 1,017,792; all three `ctclip` modes are 509,888; both
  `merlin` input modes are 722,784. Each pair differs only in *which* tensor a level
  reads, or *where* a parameter-free resample happens.
- **Arm S adds a bit-identical constant.** 700,752 − 351,168 = **349,584**, and
  1,367,376 − 1,017,792 = **349,584**. Same wrapper, same cost, on a hierarchical CNN
  and on a columnar ViT alike. That constant is what makes Arm S a controlled
  comparison rather than per-backbone tuning.
- **`vit_adapter` is 21× the largest probe adapter** (25.86 M vs 1.22 M) and is why it
  cannot sit in the probe column.

---

## 7. The experiment arms

Three arms, tagged in `configs/lowshot/MANIFEST.csv` via the `arm` column. Each arm
runs the full 4 conditions × 5 data fractions, so `frozen`-vs-`finetuned` and
`pretrained`-vs-`scratch` stay meaningful **within** an arm.

| Arm | What it is | Raw-input branch | Probe | Members |
|---|---|---|---|---|
| **N** | the encoder **as delivered**: pretrained weights → thin neck → shared head | no | **yes** | all 12 encoders (`native` ×9, `upsample` ×2, `layerwise` ×1) + the `dino3dU` and `merlinISO` ablations |
| **S** | Arm N **plus** a shared `SpatialPriorModule3D` fused into contract levels 0–3 | yes, additive | no | `ctfm_stem`, `dino3d_stem` |
| **B** | best-effort dense recipe: 3D ViT-Adapter, SPM bidirectionally fused | yes, bidirectional | no | `dino3d_vitadapter` |

**Never merge S or B rows into the Arm N column.** Read arms via the `arm` column;
the launcher's `PROBE=True` filter enforces it (`probe_comparable` is `False` for every
S and B row, guarded by an explicit `not stem_fused` check — the adapter-string test
alone would not catch `"native+stem"`).

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

For a columnar ViT this supplies the stride-1 wire it never had. For a CNN it is a
fresh stem *alongside* the pretrained one — because a CNN's own stride-1 level already
*is* a two-conv stem on raw voxels, so the honest contrast is pretrained-stem vs
fresh-stem, not stem vs no-stem.

The deliverable is not the level Arm S reaches, it is the **delta**:

```
compensable(m) = gap_N(m) - gap_S(m)     missing fine detail — a stem fixes it
irreducible(m) = gap_S(m)                missing pretrained mid-level features
```

Expect `compensable ≈ 0` for hierarchical encoders and large for columnar ViTs. Both
arms must exist for the same (condition, fraction) cell for the subtraction to be
defined.

`GroupNorm` (8 groups) replaces upstream `SyncBatchNorm` so the module is independent
of distributed training and batch size. Wrapping a mode that already runs a raw-input
branch is refused (`_ALREADY_HAS_STEM = {"spm", "vit_adapter"}`) — stacking two stems
would double the added capacity and destroy the constant the arm depends on.

Current pilot scope is the maximal-contrast pair (`ctfm`, `dino3d`), where the delta
should be largest. Extending to more members per architecture family is what turns the
pilot into a family-level claim.

### Arm B — the best-effort upper bound

The faithful plain-ViT dense-prediction recipe, so the pretrained representation feeds
every scale instead of only stride 16. It answers "if you are going to use a plain ViT
densely, what is available and what does it cost" — a different and legitimate
question from Arm N's. Report it beside the `dino3d` Arm N row, never inside it.

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

**Deliberately per-model, and correctly so:** the HU window and orientation
(`model.preprocessing`). The normalization a model was pretrained with is part of its
input interface, not a tunable knob — matching it is what makes the comparison fair
rather than what breaks it.

**Honest artifacts, to be reported not hidden:**

- Pretraining budgets and supervision differ enormously (CT-CLIP: ~50 k volumes +
  reports; VoCo: ~160 k unlabeled; SuPreM / VISTA: voxel labels).
- `ctclip` and `sam_med3d` resize inside the adapter, so effective resolution differs
  even though FOV does not.
- `biomedparse` is 2D per-slice; its depth context is entirely adapter-supplied.
- Adapter budgets differ by up to 4.2× within the probe column (§6).
- The stride-1 level is a fresh stem for the Swin family and for `biomedparse`, since
  neither owns one natively.

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
- **STU-Net** (`stunet_small` / `_base` / `_large` / `_huge`). Registered but not
  implemented: `_build_stunet_encoder` raises `NotImplementedError`, it needs the
  upstream repo vendored plus a PyTorch 1.10 / nnU-Net V1 environment, and it is not
  part of the 11-encoder benchmark. Treat the configs and `requirements-stunet.txt` as
  dead unless someone finishes the adapter.
- **`docs/PYRAMID_REFRACTOR.md`** — the migration note for the 4→5 level change. Its
  still-live content (native pyramid inventory, contract rationale, adapter strategy,
  measured budgets) is folded into this document; the historical before/after framing
  was dropped.
