# Unified Fine-Tuning Framework for 3D CT Foundation Models

A config-driven framework for comparing pretrained medical-imaging encoders under one
adaptation interface. The premise: **swap the pretrained encoder, keep everything else
identical** — then measure what pretraining is actually worth, at which annotation
budget, and for which structures.

```
config = configs/models/*.yaml
→ same TotalSegmentator preprocessing, patch size, sampler, augmentation
→ same optimizer, LR schedule, loss, epoch budget          (train: is config-locked)
→ same 5-level feature contract and shared decoder
→ same evaluator (gt-present Dice / HD95 per class, same held-out subjects)
→ different pretrained encoder
```

Two benchmark tracks: **segmentation** on TotalSegmentatorV2 (117 structures) and a
**classification** linear probe on CT-RATE (18-way chest abnormalities).

## Encoders

All 12 adapters below are implemented, weight-loading asserted, and shape-verified
against the contract. Two are report-supervised (`ctclip`, `merlin`) and they differ in
architecture family, so "vision-language pretraining" is not confounded with "columnar
ViT".

| Config | Encoder | Pretraining | Contract mode |
|---|---|---|---|
| `ctfm` | SegResNet-DS encoder | SSL, ~148 k CT volumes | `native` (1:1, 5 levels) |
| `vista3d` | SegResNet-DS encoder | supervised segmentation | `native` (1:1, 5 levels) |
| `voco_b` / `voco_h` | SwinUNETR-B / H | SSL, ~160 k volumes | `native` + stride-1 conv stem |
| `suprem_unet` | UNet3D encoder | supervised (AbdomenAtlas) | `native` + stride-16 strided conv |
| `suprem_segresnet` | MONAI SegResNet | supervised (AbdomenAtlas) | `native` + stride-16 strided conv |
| `suprem_swinunetr` | MONAI Swin | supervised (AbdomenAtlas) | `native` + stride-1 conv stem |
| `biomedparse` | FocalNet (2D, per-slice) | supervised, multi-modality | `native` + depth mixers + stem |
| `ctclip` | CTViT image tower | vision-language (reports) | `upsample` |
| `merlin` | I3D-ResNet-152 image tower | vision-language (reports + EHR codes) | `native` + stride-1 conv stem |
| `sam_med3d` | ViT (promptable SAM) | promptable segmentation | `upsample` |
| `dino3d` | ViT-L/16 (3DINO) | SSL (DINOv2-style) | `layerwise` |

Variant configs for controlled A/B pairs: `ctclip_layerwise`, `ctclip_multiscale`,
`dino3d_upsample`, `dino3d_vitadapter`, `merlin_isotropic`, `ctfm_stem`, `dino3d_stem`,
`ctfm_mask2former`, `ctfm_unet_128`.

**Not implemented:** `stunet_*` is a registered stub that raises `NotImplementedError`.
It needs the upstream repo vendored plus a PyTorch 1.10 / nnU-Net V1 environment, and it
is not part of the benchmark. Treat those configs and `requirements-stunet.txt` as dead.

## The contract

Encoders disagree about feature hierarchy: SegResNet gives 5 native scales, Swin gives 5
at shifted strides, a plain ViT gives one scale (or N at the same scale), a VAE-style
tower gives one bottleneck. If each used its native decoder you would be comparing
encoder + decoder + schedule, not the encoder. So every backbone is wrapped to expose

**5 feature maps at strides `(1, 2, 4, 8, 16)` with channels `(32, 64, 128, 256, 512)`,
finest first**

and one shared decoder (8,674,072 parameters) is trained on top. Everything trainable
after the encoder lives under `self.adapter`, which is what makes a frozen run a
representation probe. Adapter budgets span 292 k – 1.22 M across the 12 encoders.

See [docs/HEAD_DESIGN.md](docs/HEAD_DESIGN.md) for the contract rationale, the adapter
modes, measured parameter counts, and the arm structure.

## Experiment arms

| Arm | What it measures | Probe? |
|---|---|---|
| **N** | the encoder as delivered — pretrained weights → thin neck → shared head | **yes** |
| **S** | Arm N plus a shared raw-input stem fused into strides 1–8, at a bit-identical +349,584 params | no |
| **B** | best-effort dense recipe (3D ViT-Adapter, bidirectional fusion) | no |

Arm S exists for the delta, not the level: `compensable = gap_N − gap_S` separates
"missing fine detail, a cheap stem fixes it" from "missing pretrained semantics."
Never merge S or B rows into the Arm N column — filter on the manifest's `arm` /
`probe_comparable` columns.

## Layout

```
unified/
├── configs/
│   ├── base.yaml              # shared recipe; train: is locked against overrides
│   ├── models/                # one per encoder + variant configs
│   └── lowshot/               # generated matrix + MANIFEST.csv
├── unified/
│   ├── data/                  # TotalSegmentator + CT-RATE datasets, transforms, cache
│   ├── models/
│   │   ├── backbones/         # per-encoder adapters + _neck.py (shared necks)
│   │   ├── stem_fusion.py     # Arm S wrapper
│   │   ├── head.py            # HEAD_REGISTRY + UnifiedSegHead
│   │   ├── mask2former_head.py
│   │   ├── seg_model.py       # SegModel = backbone + head
│   │   └── registry.py
│   ├── training/              # trainer, loss, optimizer/scheduler
│   ├── evaluation/            # sliding-window inference + Dice/HD95
│   └── utils/                 # config loader (the fairness lock), logging, checkpoint
├── scripts/                   # train, evaluate, matrix generation + launchers
└── docs/
    ├── HEAD_DESIGN.md         # contract, adapter modes, arms  ← start here
    ├── ARCHITECTURE.md        # module map, config precedence, both tracks
    └── ADDING_A_MODEL.md      # how to add an encoder
```

## Environment

```bash
python -m venv env && source env/bin/activate
pip install -r requirements.txt
```

PyTorch ≥ 2.2, MONAI ≥ 1.3, nibabel, tqdm, pyyaml. Several adapters import from sibling
checkouts in the parent directory (`3DINO/`, `BiomedParse/`, `CT-CLIP/`, `Merlin/`), so keep the
repo layout intact.

## Data

`configs/base.yaml` points at `/store/Datasets/TotalSegmentatorDataset/` (read-only).
Split comes from `meta.csv`'s `split` column: **1082 train / 57 val / 89 test**. The
117-class map is the alphabetical enumeration of `segmentations/*.nii.gz` with background
as class 0 — see `unified/data/totalsegmentator.py`.

CT-RATE is streamed and cached separately by `scripts/ctrate_cache.py`.

## Running

```bash
# one-time manifest build
python -m scripts.prepare_data

# single run
python -m scripts.train --config configs/models/ctfm.yaml --output runs/ctfm_run1

# final evaluation (full metric list incl. HD95)
python -m scripts.evaluate --config configs/models/ctfm.yaml \
    --checkpoint runs/ctfm_run1/best.pt

# low-shot matrix: regenerate configs, then launch a filtered slice
python -m scripts.gen_lowshot_configs
GPU=1 COND=frz_pt FRAC=1.0 PROBE=True bash scripts/run_lowshot_matrix.sh

# classification track
python -m scripts.ctrate_linprobe --models ctfm vista3d --roi 128 128 128
```

The launcher is manifest-driven and resume-resilient (it skips completed `.done` runs),
which matters on a shared node where the OOM killer intervenes. Filters: `COND`, `FRAC`,
`BACKBONE`, `ADAPTER`, `PROBE`.

## Two results worth knowing before you read numbers

- **Dice must be averaged over gt-present classes only.** A case holds ~60 of 117
  structures; scoring absent classes as 0.0 halves the apparent result. The evaluator
  masks on label-derived gt-presence, matching CT-FM's `ignore_empty=True`.
- **A frozen run is only a probe if every trainable parameter sits in a thin neck.**
  `pyramid_mode` values `spm` and `vit_adapter` put a randomly-initialised conv branch on
  the raw input and are flagged `probe_comparable=False` throughout.
