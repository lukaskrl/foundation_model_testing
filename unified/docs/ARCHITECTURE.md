# Architecture

## Components

```
┌──────────────────────────────────────────────────────────────────────┐
│  scripts/train.py                                                    │
│  - load config (base ⊕ model), enforce the train: lock               │
│  - apply data.train_fraction (nested, seeded low-shot subset)        │
│  - apply model.pretrained (false -> weights=None, scratch twin)      │
│  - build TotalSegmentatorDataset(train) / (val)                      │
│  - build SegModel = Backbone(model_cfg) + head from HEAD_REGISTRY    │
│  - Trainer(model, loaders, cfg).run()                                │
└──────────────────────────────────────────────────────────────────────┘
              │
              ├─ unified.data
              │     TotalSegmentatorDataset, build_transforms (MONAI)
              │     CachedDataset — disk cache of the MODEL-INDEPENDENT prefix
              │     ctrate.py — CT-RATE streaming cacher (classification track)
              │     CLASSES: 117 structures + background, alphabetical
              │
              ├─ unified.models
              │     BackboneInterface  — forward_features -> 5 tensors, contract-asserted
              │     backbones/         — one file per encoder + _neck.py (shared necks)
              │     stem_fusion.py     — StemFusionBackbone, the Arm S wrapper
              │     head.py            — HEAD_REGISTRY, UnifiedSegHead (default)
              │     mask2former_head.py— query-based alternative head
              │     seg_model.py       — SegModel = backbone + head
              │     registry.py        — build_backbone(), consumes stem_fusion kwargs
              │
              ├─ unified.training
              │     Trainer (AdamW + WarmupCosine, DiceCE, AMP, grad accum/clip)
              │
              └─ unified.evaluation
                    Evaluator (sliding_window_inference + Dice/HD95)
```

The feature contract, the adapter modes and the N / S / B arm structure are specified in
[HEAD_DESIGN.md](HEAD_DESIGN.md). Read that first — it is the load-bearing document.

## Configs

`configs/base.yaml` holds everything shared: data root and split source, spacing,
intensity window, patch size and sampler, augmentation, optimizer, LR schedule, loss,
evaluation settings, and the `head:` block.

Precedence rules enforced by the config loader:

| Block | A model config may… | Why |
|---|---|---|
| `model:` | set freely | this is the variable under study |
| `head:` `data:` `eval:` | **deep-merge** onto the base | lets a deliberate protocol run change FOV, `train_fraction`, head choice |
| `train:` | **nothing — raises `ConfigError`** | the optimizer / schedule / loss recipe must be identical across every run |

That lock is the mechanism behind the fairness claim. For any **cross-backbone**
comparison, leave `head:`, `data:` and `eval:` at their base defaults too; override them
only for an explicitly labelled best-effort or protocol run.

Two knobs drive the benchmark matrix:

- **`data.train_fraction`** (+ `train_fraction_seed`, default 1234) — keeps a seeded
  **nested** prefix of one permutation of the train split (1 % ⊂ 5 % ⊂ 10 % ⊂ 25 % ⊂
  100 %). Model-independent, so every encoder sees the identical subset at a given
  fraction. The validation split is never subset. The disk cache is subject-keyed, so
  it is not invalidated by changing the fraction.
- **`model.pretrained`** — `false` passes `weights=None`, giving the random-init twin.
  Every backbone guards its load with `if weights:`, so the scratch arm is free.

## Lifecycle of a forward pass

```
batch["image"]  : (B, 1, 96, 96, 96)     float32, in the model's own intensity range
batch["label"]  : (B, 1, 96, 96, 96)     int64, values in [0, 117]

SegModel.forward(batch["image"]):
    native = backbone.encoder_forward(image)             # under no_grad when frozen
    feats  = backbone.adapter_forward(native, shape)     # 5 tensors, contract-asserted
    logits = head(image, feats)                          # (B, 118, 96, 96, 96)

loss = DiceCE(batch_dice=True, include_background=True, ce_class_weights=...)
```

With `head.deep_supervision: true` the head returns four logit maps at strides
1, 2, 4, 8 during training, weighted by `head.ds_weights`.

## Preprocessing and caching

The pipeline is deliberately split so one cache serves every encoder:

```
cached (model-independent)          on-the-fly (per-model)
─────────────────────────          ──────────────────────
EnsureTyped                        Orientationd (model.preprocessing.axcodes)
Spacingd  -> 1.5 mm iso            intensity window (range|percentile|sigmoid|znorm)
CropForegroundd (margin 10)        class-balanced patch sampling + augmentation
```

Orientation and intensity are **not** cached because they are part of each model's
input interface. Cache location is `data.cache.dir`; the key includes a fingerprint of
the cached prefix (`unified/data/cache.py:preprocessing_fingerprint`).

## Lifecycle of evaluation

Validation runs at native resolution (after 1.5 mm resampling) with MONAI's
`sliding_window_inference`, `roi_size = (96,96,96)`, `overlap = 0.5`, gaussian blending.

**Dice is averaged over gt-present classes only.** A TotalSegmentator case contains
roughly 60 of the 117 structures; averaging over all 117 and scoring absent classes 0.0
understates results by roughly a factor of two. The evaluator derives gt-presence from
the label one-hot and masks on it, matching CT-FM's `ignore_empty=True`. HD95 requires a
class present in **both** prediction and reference, so it approaches an upper bound
rather than diverging when a model detects almost nothing.

In-training validation computes `dice` only (`eval.val_metrics`) — HD95 builds
full-resolution CPU one-hot volumes and spikes host RAM hard enough to attract the
shared node's OOM killer. `scripts/evaluate.py` runs the full metric list for final
numbers.

Checkpoint selection is on mean Dice, with early stopping after
`train.early_stop_patience` validation rounds without a gain of at least
`train.early_stop_min_delta`.

## Benchmark tracks

**Segmentation (primary).** TotalSegmentatorV2, 117 structures, split 1082 / 57 / 89
from `meta.csv`. The low-shot matrix lives in `configs/lowshot/` — generated by
`scripts/gen_lowshot_configs.py` as 4 conditions (`frz_pt`, `frz_sc`, `ft_pt`, `ft_sc`)
× 5 fractions × arm, with `MANIFEST.csv` as the index and `scripts/run_lowshot_matrix.sh`
as a resume-resilient, filterable launcher:

```bash
GPU=1 COND=frz_pt FRAC=1.0 PROBE=True bash scripts/run_lowshot_matrix.sh
```

`PROBE=True` restricts to probe-comparable rows — pass it for any headline launch, or
Arm S / Arm B rows leak into the probe column.

**Classification (secondary).** CT-RATE 18-way multi-label chest abnormality linear
probe — the counterpart test of whether conclusions hold beyond dense segmentation.
`scripts/ctrate_cache.py` builds a model-independent HU-int16 cache in canonical RAS;
`scripts/ctrate_linprobe.py` extracts frozen pooled embeddings and fits one
`Linear(dim, 18)` per encoder.

## Adding a model

See [ADDING_A_MODEL.md](ADDING_A_MODEL.md). The short version: implement
`BackboneInterface` returning 5 contract tensors, put every trainable post-encoder
module under `self.adapter`, register with `@register_backbone`, add
`configs/models/<name>.yaml`, then add it to the right group in
`scripts/gen_lowshot_configs.py`.
