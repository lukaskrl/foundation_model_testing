# Unified TotalSegmentator Fine-Tuning Framework

A single, config-driven fine-tuning and evaluation framework for comparing medical-imaging
foundation models on the [TotalSegmentator v2](https://github.com/wasserth/TotalSegmentator)
CT dataset. The premise: **swap the pretrained encoder, keep everything else identical.**

```
config = pick(unified/configs/models/*.yaml)
→ same TotalSegmentator preprocessing
→ same patch size, batch size, optimizer, LR schedule, loss, augmentation, # epochs
→ same uniform segmentation head (decoder)
→ same evaluator (Dice/HD95 per class, on the same held-out subjects)
→ different pretrained backbone
```

## Status

This repo is a **scaffold**. It compiles a clean interface around six foundation-model
encoders, a shared U-Net-style decoder, a shared trainer, and a shared evaluator. Some
per-model adapters are reference implementations (3DINO, VoCo, VISTA) and some are
stubs that still need wiring (STU-Net, BiomedParse, CT-CLIP) — see
[docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md) for the contract each adapter must
satisfy.

| Model | Backbone | Status | Notes |
|---|---|---|---|
| `voco_b` / `voco_h` | SwinUNETR-B / H | adapter implemented | Native 5-level hierarchy |
| `vista3d` | SegResNet-DS encoder | adapter implemented | Drops VISTA's class/point heads |
| `dino3d` | DinoVisionTransformer3d | adapter implemented | UNETR-style pyramid from ViT layers [5,11,17,23] |
| `stunet_small` / `stunet_huge` | nnU-Net V1 encoder | adapter stub | Needs upstream `uni-medical/STU-Net` cloned to access encoder code |
| `biomedparse` | Focal backbone (2D) | adapter stub | Slice-wise; not strictly comparable, see `docs/HEAD_DESIGN.md` |
| `ctclip` | CTViT spatial encoder | adapter stub | Single-scale bottleneck; head uses learned upsampling pyramid |

Models intentionally excluded: **SAM-Med3D** (prompt-based — no automatic full-volume
seg), **LVM-Med** (2D SAM, single-mask decoder).

## Layout

```
unified/
├── configs/
│   ├── base.yaml                  # shared data/training/eval — never change to make a model "fit"
│   └── models/
│       ├── voco_b.yaml
│       ├── voco_h.yaml
│       ├── vista3d.yaml
│       ├── dino3d.yaml
│       ├── stunet_small.yaml
│       ├── stunet_huge.yaml
│       ├── biomedparse.yaml
│       └── ctclip.yaml
├── unified/
│   ├── data/                      # TotalSegmentator dataset + shared preprocessing
│   ├── models/
│   │   ├── backbones/             # per-model wrappers (one file per foundation model)
│   │   ├── head.py                # the shared UnifiedSegHead
│   │   ├── seg_model.py           # SegModel = backbone + head, the thing being trained
│   │   └── registry.py
│   ├── training/                  # trainer, loss, optimizer/scheduler
│   ├── evaluation/                # sliding-window inference + Dice/HD95
│   └── utils/                     # config loader, logging, checkpoint
├── scripts/
│   ├── prepare_data.py            # one-shot: build train/val/test manifests from meta.csv
│   ├── train.py                   # CLI: python -m scripts.train --config configs/models/voco_b.yaml
│   ├── evaluate.py
│   └── verify_setup.py
└── docs/
    ├── ARCHITECTURE.md            # the framework design, in detail
    ├── HEAD_DESIGN.md             # why this decoder, what's "fair", what's not
    └── ADDING_A_MODEL.md          # how to add a new backbone in <100 LoC
```

## Why a unified head?

Pretrained backbones differ in feature hierarchy: SwinUNETR gives 5 native scales,
SegResNet gives 4, a ViT gives 1 (or N at the same scale), a VAE encoder gives 1
bottleneck. If each model used its native decoder, you'd be comparing
backbone + decoder + loss + schedule — not just the backbone. So we standardize the
decoder: every backbone is wrapped to expose **four feature maps at strides
{4, 8, 16, 32}** with **fixed channel counts {64, 128, 256, 512}**, and a single
U-Net-style decoder is trained on top. See `docs/HEAD_DESIGN.md` for the channel/stride
math and the known limitations of this choice for CT-CLIP and BiomedParse.

## Environment

This scaffold installs nothing. Once you decide to run it:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Two environments are unavoidable:

- **Default** (`requirements.txt`): PyTorch ≥ 2.2, MONAI ≥ 1.3, nibabel, tqdm,
  pyyaml. Covers 3DINO, VoCo, VISTA-3D, CT-CLIP, BiomedParse.
- **STU-Net** (`requirements-stunet.txt`): PyTorch 1.10 + nnU-Net V1 (1.7.0). STU-Net's
  weights are nnU-Net V1 checkpoints — they need the upstream `uni-medical/STU-Net` code
  cloned at `vendor/STU-Net/` and the old PyTorch.

## Data

Path is hard-coded in `configs/base.yaml`:
`/store/Datasets/TotalSegmentatorDataset/`. The dataset directory is **read-only**;
all preprocessing happens in-memory at training time (resample → intensity normalize →
random crop). Train/val/test split comes from `meta.csv`'s `split` column
(1082 / 57 / 89 subjects).

The 117-class label map is built by enumerating `segmentations/*.nii.gz` in any subject
(class index = sorted alphabetical order of organ names + background as class 0).
See `unified/data/totalsegmentator.py` for the canonical class list.

## Running

```bash
# build the train/val/test manifest (one-time)
python -m scripts.prepare_data

# fine-tune VoCo-B
python -m scripts.train --config configs/models/voco_b.yaml --output runs/voco_b_run1

# evaluate
python -m scripts.evaluate --config configs/models/voco_b.yaml \
    --checkpoint runs/voco_b_run1/best.pt
```
