# Architecture

## Components

```
┌──────────────────────────────────────────────────────────────────────┐
│  scripts/train.py                                                    │
│  - parse config (base + model)                                       │
│  - build TotalSegmentatorDataset(train) / (val)                      │
│  - build SegModel = Backbone(model_cfg) + UnifiedSegHead(head_cfg)   │
│  - call Trainer(model, loaders, cfg).run()                           │
└──────────────────────────────────────────────────────────────────────┘
              │
              ├─ unified.data
              │     TotalSegmentatorDataset, build_transforms (MONAI)
              │     CLASSES: 117 organs + background, alphabetical
              │
              ├─ unified.models
              │     BackboneInterface (forward_features → 4 tensors)
              │     UnifiedSegHead (UNETR decoder, num_classes=118)
              │     SegModel (backbone + head)
              │     backbones/{voco,vista,dino,stunet,biomedparse,ctclip}.py
              │
              ├─ unified.training
              │     Trainer (AdamW + WarmupCosine, DiceCELoss, AMP)
              │
              └─ unified.evaluation
                    Evaluator (sliding_window_inference + Dice/HD95)
```

## Configs

`configs/base.yaml` is the **shared** config. It defines:
- data root and split source
- patch size, spacing, intensity range, augmentations
- batch size, num_workers
- optimizer (AdamW), LR, weight decay
- LR schedule (warmup + cosine), epochs
- loss (DiceCE with softmax)
- evaluation (sliding window roi, overlap, metrics)
- decoder (UnifiedSegHead) channel widths and num_classes

Per-model configs in `configs/models/*.yaml` define **only** the backbone:
- `model.name` — registry key
- `model.weights` — path under `weights/`
- `model.kwargs` — backbone constructor arguments

The framework merges `base.yaml ⊕ models/<name>.yaml` with the model's section taking
precedence only over its own subsection. If a model config tries to override a
training/data field, the loader raises `ConfigError` — that's by design (fair
comparison).

## Lifecycle of a forward pass

```
batch["image"]  : (B, 1, 96, 96, 96)         float32, normalized to [-1, 1]
batch["label"]  : (B, 1, 96, 96, 96)         int64, values in [0, 117]

SegModel.forward(batch["image"]):
    feats = self.backbone.forward_features(image)  # List[4 tensors]
    logits = self.head(feats)                       # (B, 118, 96, 96, 96)
    return logits

loss = DiceCELoss(softmax=True)(logits, batch["label"])
```

## Lifecycle of evaluation

Validation/test scans are run at native resolution (after resampling to 1.5 mm) using
MONAI's `sliding_window_inference` with the same `roi_size = (96,96,96)` and
`overlap = 0.5`. Per-class Dice and HD95 are computed against the full ground-truth
volume (not per-patch). Reported metrics: mean Dice, median Dice, per-class Dice for
all 117 organs, plus HD95.

## Adding a model

See [ADDING_A_MODEL.md](ADDING_A_MODEL.md). The short version:
1. Write `unified/models/backbones/<name>.py` exposing a class that inherits from
   `BackboneInterface` and implements `forward_features(x) → List[4 Tensor]`.
2. Register it: `@register_backbone("my_model")`.
3. Add `configs/models/<name>.yaml`.
