# Unified segmentation head — design

## Goal

When comparing pretrained encoders, the experimentally interesting variable is the
**encoder**. Everything else — decoder architecture, channel widths, loss, optimizer,
augmentation, batch size — must be held constant. This document specifies the constant
parts: input contract, decoder architecture, and the per-backbone "neck" that adapts an
encoder's native features into the contract.

## The contract

Every backbone in this framework implements `BackboneInterface` (see
`unified/models/seg_model.py`):

```python
def forward_features(x: Tensor[B,1,D,H,W]) -> List[Tensor]
```

The returned list has **exactly 4 tensors** with these shapes:

| Pyramid level | Stride | Channels | Shape (input 96³) |
|---|---|---|---|
| `feat[0]` | 4  | 64  | (B, 64, 24, 24, 24) |
| `feat[1]` | 8  | 128 | (B, 128, 12, 12, 12) |
| `feat[2]` | 16 | 256 | (B, 256, 6, 6, 6) |
| `feat[3]` | 32 | 512 | (B, 512, 3, 3, 3) |

A small **neck** (1×1×1 convs + interpolation as needed) inside each adapter does the
channel and stride adaptation. The neck is the same flavor of operation across all
backbones — it has very few learnable parameters relative to the decoder, so it
shouldn't dominate the comparison.

The decoder (`UnifiedSegHead`) is the **same module instance type** in every
configuration. It is a UNETR-style U-Net decoder:

```
feat[3] (B,512,D/32) ──┐
                       ├─ UnetrUpBlock → feat[2]
feat[2] (B,256,D/16) ──┤
                       ├─ UnetrUpBlock → feat[1]
feat[1] (B,128,D/8) ───┤
                       ├─ UnetrUpBlock → feat[0]
feat[0] (B,64,D/4) ────┤
                       └─ UnetrUpBlock → (B,32,D)
                                          ↓ 1×1×1 conv
                                         (B, num_classes, D, H, W)
```

`num_classes = 118` (117 organs + background).

## How each backbone fills the contract

### SwinUNETR (VoCo)

Native hierarchy from `SwinUNETR.swinViT`:
`[(B,48,D), (B,96,D/2), (B,192,D/4), (B,384,D/8), (B,768,D/16)]`.

We take levels [1, 2, 3, 4] and apply a 1×1×1 conv per level to remap
channels {96, 192, 384, 768} → {64, 128, 256, 512}. Strides {2, 4, 8, 16} → we
**downsample once more** (stride-2 conv) so the pyramid is {4, 8, 16, 32}.

### SegResNet-DS encoder (VISTA-3D)

Native hierarchy from `SegResEncoder._forward`:
`[(B,32,D/2), (B,64,D/4), (B,128,D/8), (B,256,D/16)]`.

Take all four levels; 1×1×1 conv to remap {32, 64, 128, 256} → {64, 128, 256, 512};
downsample once to bring strides {2, 4, 8, 16} → {4, 8, 16, 32}.

### DinoVisionTransformer3d (3DINO)

ViT has one spatial scale (patch_size=16 → stride 16, all blocks). Extract four
intermediate blocks `[5, 11, 17, 23]` via `get_intermediate_layers(..., reshape=True)`.
This gives four tensors of shape `(B, 1024, D/16, H/16, W/16)`.

To synthesize a multi-scale pyramid we use the **UNETR projection trick** (same as
`segmentation_heads.py:UNETRHead`): apply a stack of `UnetrPrUpBlock` upsamplers with
{2, 1, 0} extra upsample layers to the shallower blocks to produce features at strides
{4, 8, 16, 32}. Then a 1×1×1 conv pins channel counts to {64, 128, 256, 512}.

### CTViT (CT-CLIP)

CTViT is a VAE encoder — a single bottleneck `(B, 512, T/4, H/16, W/16)`. There are no
intermediate skip features. We synthesize a four-scale pyramid by **learned
upsampling**: take the bottleneck and progressively `ConvTranspose3d` it to strides
{32, 16, 8, 4}, with channel counts {512, 256, 128, 64}.

**Caveat:** This decoder has more parameters at the pyramid stage than other backbones'
necks. The CT-CLIP comparison is the loosest in the suite — flag it in any paper
write-up. CT-CLIP was not pretrained for segmentation; this is the most generous head
we can give it that still respects the framework contract.

### nnU-Net V1 encoder (STU-Net)

Native hierarchy `[(B,32,D), (B,64,D/2), (B,128,D/4), (B,256,D/8), (B,320,D/16)]`
(canonical nnU-Net V1 with 5 downsampling stages). Take levels [1, 2, 3, 4];
1×1×1 conv to {64, 128, 256, 512}; downsample once so strides {2, 4, 8, 16} →
{4, 8, 16, 32}. **Note:** STU-Net's encoder needs the upstream code at
`vendor/STU-Net/` since the local checkout is docs-only.

### Focal backbone (BiomedParse) — 2D

BiomedParse's backbone is 2D (per slice). The four stages give 2D features
`{(B,96,H/4), (B,192,H/8), (B,384,H/16), (B,768,H/32)}` per slice.

The adapter handles this by:
1. Reshape (B, 1, D, H, W) → (B·D, 1, H, W). Convert grayscale CT to 3-channel by
   broadcasting (BiomedParse's input expectation).
2. Run the 2D backbone per slice.
3. Reshape back: per-stage features → (B, C, D, H/s, W/s).
4. 1×1×1 conv to remap channels; 1D conv along D to give the depth dimension some
   receptive field (a small ConvTranspose1d isn't enough — see code).

**Caveat:** BiomedParse processes slices independently. We give it the same
receptive field budget as other models by adding small depth-axis 3D convs after each
stage, but this is the **least apples-to-apples backbone**. Document it as such.

## What's "fair" and what isn't

**Fair:**
- Same data, same train/val/test split, same preprocessing.
- Same patch size, batch size, optimizer, schedule, epochs.
- Same loss (DiceCE).
- Same decoder architecture and weights initialization scheme.

**Not perfectly fair:**
- Pretraining budgets differ wildly (CT-CLIP saw ~50k CT-RATE volumes + text;
  STU-Net saw TotalSegmentator itself; VoCo saw 160k unlabeled volumes; etc.).
- BiomedParse's 2D-per-slice forward is fundamentally lower-receptive-field along
  the D axis.
- CT-CLIP's pyramid has the most "fresh" parameters per backbone.

These are **honest comparison artifacts**, not bugs. We document them; we don't try to
hide them. The goal is to measure "given identical fine-tuning conditions, which
pretrained encoder transfers best to TotalSegmentator?" — that's a useful question
even with the asterisks above.
