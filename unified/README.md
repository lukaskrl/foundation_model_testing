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

All 12 adapters below are implemented, shape-verified against the contract, and
weight-loading asserted through one shared guard (`assert_encoder_loaded`) that requires
**every** encoder key to be populated — all twelve load 100 % of their keys from the
real released checkpoints. Two are report-supervised (`ctclip`, `merlin`) and they differ in
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
`dino3d_upsample`, `dino3d_vitadapter`, `merlin_isotropic`, `ctfm_mask2former`,
`ctfm_unet_128`. The Arm S and Arm W variants are *generated* per encoder by
`scripts/gen_lowshot_configs.py` rather than hand-written, so every one gets a
bit-identical stem / window (`ctfm_stem.yaml` and `dino3d_stem.yaml` remain only for
standalone one-off runs).

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

Each arm holds the encoder fixed and varies exactly one thing, so each is a controlled
subtraction against its Arm N twin.

| Arm | What it varies | What it controls for | Probe? |
|---|---|---|---|
| **N** | nothing — the encoder as delivered | — | **yes** |
| **S** | + a shared raw-input stem fused into strides 1–8, a bit-identical +349,584 params | **capacity** for fine detail | no |
| **B** | best-effort dense recipe (3D ViT-Adapter, bidirectional fusion) | — (upper bound) | no |
| **W** | the HU window, forced to one of two shared values | the **input** interface | no |

The arms exist for the deltas, not the levels:

```
compensable(m)    = gap_N(m) - gap_S(m)          missing fine detail, a stem fixes it
irreducible(m)    = gap_S(m)                     missing pretrained mid-level features
window_cost(m, w) = score(m, w) - score(m, native)   what the input interface costs
```

Read `compensable` against the manifest's `arm_n_fine_source` column — for the four
encoders whose Arm N stride-1 level was *already* a fresh stem, `compensable ≈ 0` means
"detail was already routed", not "this encoder routes detail well".

Never merge S, B or W rows into the Arm N column — filter on the manifest's `arm` /
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
repo layout intact:

```bash
git submodule update --init --depth 1            # the 10 pinned upstream repos
git clone --depth 1 https://github.com/StanfordMIMI/Merlin.git ../Merlin
```

`unified/utils/paths.py` resolves both roots from its own location, so a fresh clone
needs no edits. Override with `FM_ROOT` (upstream checkouts) or `WEIGHTS_ROOT`
(checkpoints) when the layout differs.

## Weights

```bash
bash scripts/download_weights.sh                 # all encoders, ~15 GB
bash scripts/download_weights.sh ctfm voco_b     # or a subset, by key
```

Resumable and idempotent — every public file is sha256-verified, so re-running after
an interruption costs nothing. Transfers go through the `hf` CLI when it is on PATH
(xet-chunked and parallel; one checkpoint served at ~50 kB/s over the plain resolve
endpoint moved at ~85 MB/s through `hf`) and fall back to `curl` otherwise.
**Three repos are gated on HuggingFace**
(`microsoft/BiomedParse`, `AICONSlab/3DINO-ViT`, and the `ibrahimhamamci/CT-RATE`
dataset that holds the CT-CLIP checkpoint): accept the terms on each repo page once,
run `hf auth login`, then re-run the script — without a token those three are skipped
and listed in the summary.

`model.weights` in each config is an absolute path under `WEIGHTS_ROOT`'s default
location. Merlin is the one derived file: the script slices the image tower out of the
released CLIP checkpoint via `scripts/slim_merlin_tower.py` (the adapter accepts either).

## Data

```bash
bash scripts/download_totalsegmentator.sh        # ~22 GiB, into $DATA_ROOT (default ~/data)
python -m scripts.prepare_data                   # splits + merged label.nii.gz per subject
```

**v2.0.1 is pinned deliberately** (Zenodo [10047292](https://doi.org/10.5281/zenodo.10047292),
CC BY 4.0): 1228 subjects / 117 structures, which is the split the committed
`unified/data/splits/*.txt` and `configs/base.yaml` were built against. A different
release silently changes the benchmark's subject set. The script cross-checks the
extracted `meta.csv` against the expected **1082 train / 57 val / 89 test** counts.

Zenodo drops long connections partway through the archive, so the script re-invokes
`curl -C -` until the byte count matches and then verifies the md5 — a plain
one-shot `curl` will usually die around 25% with error 18.

`data.dataset_root` / `data.meta_csv` in `configs/base.yaml` name the extracted tree.
Note the dataset root must be **writable**: `prepare_data` writes each subject's merged
`label.nii.gz` next to its `ct.nii.gz`, which is what keeps the loader from gunzipping
118 files per sample. The 117-class map is the alphabetical enumeration of
`segmentations/*.nii.gz` with background as class 0 — see
`unified/data/totalsegmentator.py`.

CT-RATE (the classification track) is a separate gated HuggingFace dataset, streamed
and cached by `scripts/ctrate_cache.py`.

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
`BACKBONE`, `ADAPTER`, `PROBE`, `ARM`, `WINDOW`.

```bash
GPU=2 ARM=S COND='frz_pt|ft_pt' FRAC=1.0 bash scripts/run_lowshot_matrix.sh   # 24 runs
GPU=3 ARM=W COND='frz_pt|ft_pt' FRAC=1.0 bash scripts/run_lowshot_matrix.sh   # 36 runs
```

## Three results worth knowing before you read numbers

- **Dice must be averaged over gt-present classes only.** A case holds ~60 of 117
  structures; scoring absent classes as 0.0 halves the apparent result. The evaluator
  masks on label-derived gt-presence, matching CT-FM's `ignore_empty=True`.
- **A frozen run is only a probe if every trainable parameter sits in a thin neck.**
  `pyramid_mode` values `spm` and `vit_adapter` put a randomly-initialised conv branch on
  the raw input and are flagged `probe_comparable=False` throughout.
- **The per-encoder HU window is not a neutral choice.** Five encoders
  (`voco_b/h`, `suprem_*`) were pretrained with a `[-175, 250]` soft-tissue window; on
  this whole-body task that clips **27.6 % of foreground voxels** and flattens the lungs
  entirely (97–99 % of lung-lobe and trachea voxels). Matching each encoder's pretraining
  normalization is a defensible default, but it means an Arm N ranking cannot separate
  "worse representation" from "narrower input window". Arm W measures the difference —
  quote `window_cost` beside any cross-encoder claim.
