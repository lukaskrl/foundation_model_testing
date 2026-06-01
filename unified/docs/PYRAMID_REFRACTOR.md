# Pyramid contract refactor — design note

## 1. Problem summary

The previous backbone-adapter contract demanded **4** feature maps at strides
`{4, 8, 16, 32}` with channels `{64, 128, 256, 512}`. Two structural
flaws hurt segmentation quality:

1. **Resolution octave-shift on conv backbones.** SegResEncoder-based backbones
   (CT-FM, VISTA-3D) natively emit 5 features at strides `{1, 2, 4, 8, 16}`.
   The old adapter sliced indices `[1..5)` and `PyramidNeck` then
   trilinearly *downsampled* every level by 2× to fit `{4, 8, 16, 32}`. The
   stride-1 feature was discarded; the stride-2 feature was thrown away in
   the resample.
2. **Fresh full-res convs on raw input did the fine-detail work.** The head
   added two `UnetrBasicBlock` layers (`enc0` on the raw image, `enc0_down2`
   = a strided conv on `enc0`) and then ran 4 transposed-conv stages.
   The stride-1 and stride-2 skips were *random-init convs on the input volume*,
   not features from the pretrained model. With the backbone frozen, those
   shallow random-init convs were the **only** trainable source of detail.

In addition, `PyramidNeck` was registered as a sub-module of the frozen
backbone, so its 1×1 channel-adapter convs were stuck at random init for the
entire training run.

## 2. Native pyramids — measured

All shapes confirmed by running the encoders on a `[1, 1, 96, 96, 96]` tensor.
For Swin/Conv backbones, channels follow a strict ×2 ladder, so they are
written as `(c_0, ..., c_N)` with `c_i = base · 2^i`.

| Backbone | Native levels | Native strides | Native channels |
|---|---|---|---|
| `ctfm`             (VISTA `SegResEncoder`, `init_filters=32`, 5 stages)  | 5 | `(1, 2, 4, 8, 16)`  | `(32, 64, 128, 256, 512)` |
| `vista3d`          (VISTA `SegResEncoder`, `init_filters=48`, 5 stages)  | 5 | `(1, 2, 4, 8, 16)`  | `(48, 96, 192, 384, 768)` |
| `suprem_segresnet` (MONAI `SegResNet.encode`, `init_filters=16`, 4 stages) | 4 | `(1, 2, 4, 8)`      | `(16, 32, 64, 128)` |
| `suprem_unet`      (SuPreM `UNet3D` encoder, 4 stages)                   | 4 | `(1, 2, 4, 8)`      | `(64, 128, 256, 512)` |
| `voco_b`           (MONAI `SwinTransformer`, `feature_size=48`, `patch=2`) | 5 | `(2, 4, 8, 16, 32)` | `(48, 96, 192, 384, 768)` |
| `suprem_swinunetr` (MONAI `SwinTransformer`, `feature_size=48`, `patch=2`) | 5 | `(2, 4, 8, 16, 32)` | `(48, 96, 192, 384, 768)` |
| `dino3d`           (`DinoVisionTransformer3d`, `vit_large_3d`, `patch=16`) | 1 | `(16,)`              | `(1024,)` |

## 3. The new pyramid contract

```
EXPECTED_STRIDES  = (1, 2, 4, 8, 16)     # finest first
EXPECTED_CHANNELS = (32, 64, 128, 256, 512)
NUM_LEVELS        = 5
```

Ordering: **finest first**. `feats[0]` is the stride-1 feature; `feats[-1]` is
the coarsest level. Matches the existing deep-supervision convention
(`ds_weights = [1.0, 0.5, 0.25, 0.125]` corresponds to predictions at strides
1, 2, 4, 8 — finest first).

### Why `{1, 2, 4, 8, 16}` and not `{1, 2, 4, 8, 16, 32}`?

- **CT-FM** and **VISTA-3D** native pyramids fit `{1, 2, 4, 8, 16}` *exactly*,
  so the dominant-quality reference (CT-FM ~0.898 native) requires **zero**
  spatial resampling. Matching the native CT-FM `SegResNetDS` decoder
  topology is the goal.
- **SwinTransformer** natively reaches `{2, 4, 8, 16, 32}`. Adding the
  stride-32 level *would* let Swin pass through with zero resampling, but it
  forces every conv backbone to invent a stride-32 level it does not own
  natively. Dropping stride-32 favours conv-family fidelity; Swin only loses
  the very-coarsest level (which an FPN typically over-weights) and gains a
  conv stem to fill stride-1 (cheap, ~50 k params).
- **ViT** (`dino3d`) cannot produce stride-32 from tokens without
  upsampling a coarser map, so stride-32 would buy nothing.

The contract is documented in `BackboneInterface.__doc__` and is a stable
interface for both adapters and heads. Future backbones implement it; future
heads consume it.

## 4. Adapter strategy per backbone

Two cleanly-separated layers per backbone:

```
raw input  ─► [pretrained encoder]  ─► native features
                                           │
                                           ▼
                                   [trainable adapter]
                                           │
                                           ▼
                                  contract pyramid (5 lvl)
                                           │
                                           ▼
                                   [trainable head]
                                           │
                                           ▼
                                       logits
```

- The pretrained `encoder` is frozen.
- The **adapter** is **trainable** (was previously frozen — bug fix). It
  contains all channel-projection 1×1 convs and any per-backbone synthesizer
  for missing strides.
- The shared **head** is trainable and backbone-independent.

`SegModel.freeze_backbone` now freezes everything in the backbone *except*
`backbone.adapter`. Each backbone is required to register its trainable
post-encoder modules under `self.adapter` (an `nn.Module` or `nn.ModuleDict`).

### Per-backbone breakdown

| Backbone | Source of each contract level |
|---|---|
| `ctfm`             | `(s1, s2, s4, s8, s16)` ← `(L0,L1,L2,L3,L4)` of `SegResEncoder` — channel-only 1×1 conv |
| `vista3d`          | same; 1×1 channel adapters (`48→32, 96→64, …`) |
| `suprem_segresnet` | `(s1..s8)` ← native skip[0..3] (1×1 conv); `s16` ← strided 3×3 conv on the adapted `s8` |
| `suprem_unet`      | `(s1..s8)` ← native skip[0..3] (1×1 conv); `s16` ← strided 3×3 conv on the adapted `s8` |
| `voco_b`           | `(s2..s16)` ← native Swin levels `0..3` (1×1 conv); `s1` ← lightweight conv stem on raw input |
| `suprem_swinunetr` | same as `voco_b` |
| `dino3d`           | `s16` ← `vit.get_intermediate_layers()[-1]` → 1×1 to 512ch; `(s1..s8)` ← `SpatialPriorModule3D` (4 stages, GroupNorm) on raw input |

**No spatial upsampling of pretrained features anywhere.** When a level is
finer than any native feature provides (Swin-`s1`, ViT-`s1..s8`), it is
generated by a *new* conv branch on the raw input. When a level is coarser
than the deepest native feature (SuPreM `s16`), it is generated by a strided
conv on the deepest native level.

### SpatialPriorModule for `dino3d`

A ported, simplified version of `3DINO/.../adapter_modules.py:SpatialPriorModule`,
adapted for our 5-level contract:

- `stem` (3 × 3×3×3 convs, **stride 1**, `GroupNorm`, ReLU) → `s1` features
- `down2` (3×3 stride-2 conv) → `s2`
- `down3` (3×3 stride-2 conv) → `s4`
- `down4` (3×3 stride-2 conv) → `s8`
- 1×1 channel projection per level to `(32, 64, 128, 256)`

`GroupNorm` (8 groups) is used instead of upstream `SyncBatchNorm` so the
module is independent of distributed-training and batch size.

No deformable Injector/Extractor blocks are included; those are part of the
deformable-head path, not this refactor.

## 5. Head — convolutional U-Net (first consumer of the contract)

`UnifiedSegHead` is rewritten as a standard U-Net decoder driven entirely by
the contract pyramid. Pseudocode for the decode path (`f_i` is contract level
`i`, with `f_0 @ stride 1, …, f_4 @ stride 16`):

```
d4 = f_4                                         # stride 16
d3 = UpBlock(d4, f_3)        in=c4, skip=c3 → c3 # stride 8
d2 = UpBlock(d3, f_2)        in=c3, skip=c2 → c2 # stride 4
d1 = UpBlock(d2, f_1)        in=c2, skip=c1 → c1 # stride 2
d0 = UpBlock(d1, f_0)        in=c1, skip=c0 → c0 # stride 1
out = 1×1 conv (c0 → 118)                        # stride 1 logits
```

Where `UpBlock = UnetrUpBlock` (ConvTranspose ×2 + concat skip + 2× residual
conv block, both reused from MONAI).

Removed: `enc0` (`UnetrBasicBlock` on raw input) and `enc0_down2` (strided
conv on `enc0`). With the new contract, real pretrained features fill every
skip down to stride 1, so the fresh-input-conv hack is gone.

### Deep supervision

Returns `[logits_s1, logits_s2, logits_s4, logits_s8]` (finest first) at
train time when `deep_supervision=True`. Each level reuses a 1×1
`UnetOutBlock` on the corresponding decoder activation. This is unchanged
from the previous head and keeps the `ds_weights` contract intact.

### Head registry

A simple `HEAD_REGISTRY` registers heads by name. `head.name`
(`unified_seg_head`) selects the head from config. The forward signature is
fixed:

```python
head(x_in: Tensor, feats: List[Tensor]) -> Tensor | List[Tensor]
```

with `feats` matching `BackboneInterface.EXPECTED_*`. A future
`mask_transformer_head` (deformable / Mask2Former-style) drops in behind the
same contract and is selected purely by config.

## 6. Trainable parameter counts (measured)

Counted at `[1, 1, 96, 96, 96]` input, `freeze_backbone=True`,
`deep_supervision=True`. The shared head accounts for **8,674,072**
trainable parameters in every backbone (it depends only on the contract,
not on the encoder).

| Backbone | Trainable BEFORE | Trainable AFTER | Adapter (trainable) | Notes |
|---|---:|---:|---:|---|
| `ctfm`             | 9,069,208 | 9,025,240 |   351,168 | 5× 1×1 channel adapters (one per native level) |
| `vista3d`          | 9,069,208 | 9,199,832 |   525,760 | 5× 1×1 channel adapters |
| `voco_b`           | 9,069,208 | 8,965,752 |   291,680 | raw→s1 conv stem + 4× 1×1 channel adapters |
| `suprem_swinunetr` | 9,069,208 | 8,965,752 |   291,680 | same as `voco_b` |
| `dino3d`           | 9,069,208 | 9,548,008 |   873,936 | SpatialPriorModule3D (strides 1..8) + 1×1 conv on tokens (stride 16) |
| `suprem_unet`      | 9,069,208 | 9,898,712 | 1,224,640 | 4× 1×1 channel adapters + s8→s16 stride-2 conv (2³ kernel) |
| `suprem_segresnet` | 9,069,208 | 9,768,152 | 1,094,080 | 4× 1×1 channel adapters + s8→s16 stride-2 conv |

Before/after totals stay in a tight band (8.97–9.90 M, **±5 %** around the
mean). Adapter sizes vary by family but the largest (`suprem_unet`,
1.22 M) is only ~4× the smallest (`voco_b`, 0.29 M) — comfortably within
the matched-capacity intent. The "before" column came from the pre-refactor
adapter+head that included a fresh-input `UnetrBasicBlock` (`enc0`) and the
4-level UNETR-style decoder; the "after" column comes from the new
5-level U-Net head plus the trainable adapters described above.

The verification matrix on the next section confirms shapes, deep-supervision
list lengths and final logit dimensions for every backbone.

## 7. Fairness lock — what did NOT change

- `data` block: untouched (spacing, intensity, patch size, sampler, etc.)
- `train` block: untouched (optimizer, lr, loss, AMP, batch size policy)
- `eval` block: untouched (sliding-window roi, overlap, metrics)
- Per-model configs: still only override `model:` (verified by the config
  loader's diff check)

The `head:` block in `configs/base.yaml` is updated to the new contract
(channels `(32, 64, 128, 256, 512)`, strides `(1, 2, 4, 8, 16)`). This is
model-independent and applies uniformly to every backbone — the fairness
lock permits this.

## 8. Verification

Each backbone is built without weights, run on a `[1, 1, 96, 96, 96]`
tensor, contract-asserted, and exercised in both `.eval()` mode (expecting a
single `[B, 118, 96, 96, 96]` tensor) and `.train()` mode (expecting a
4-element finest-first deep-supervision list at strides 1, 2, 4, 8).

| Backbone | Native shapes (96³ input) | Contract OK | Eval shape | DS shapes |
|---|---|---|---|---|
| `ctfm`             | 32@96, 64@48, 128@24, 256@12, 512@6 | ✓ | (1,118,96,96,96) | [(1,118,96³),(1,118,48³),(1,118,24³),(1,118,12³)] |
| `vista3d`          | 48@96, 96@48, 192@24, 384@12, 768@6 | ✓ | (1,118,96,96,96) | same |
| `voco_b`           | 48@48, 96@24, 192@12, 384@6, +stem | ✓ | (1,118,96,96,96) | same |
| `suprem_swinunetr` | 48@48, 96@24, 192@12, 384@6, +stem | ✓ | (1,118,96,96,96) | same |
| `dino3d`           | tokens 1024@6 + SPM @ 96,48,24,12 | ✓ | (1,118,96,96,96) | same |
| `suprem_unet`      | 64@96, 128@48, 256@24, 512@12, +s16 | ✓ | (1,118,96,96,96) | same |
| `suprem_segresnet` | 16@96, 32@48, 64@24, 128@12, +s16  | ✓ | (1,118,96,96,96) | same |

The standalone `scripts/smoke_compute.py` test (random tensor, 2 forward+
backward steps, AdamW step) and `scripts/smoke_train.py` (one real
TotalSegmentator volume from the train split through the full transform
pipeline) both `SMOKE OK` for every backbone. Loss decreased between step 0
and step 1 in every case, e.g. `ctfm` `11.32 → 10.81` over one volume.

Gradient flow is exactly as designed: with `freeze_backbone=True`, every
encoder parameter is `requires_grad=False` and receives no gradient; every
parameter under `backbone.adapter` receives gradient; every head parameter
receives gradient.
