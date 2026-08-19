# Adding a backbone

The contract: from a `(B, 1, D, H, W)` CT input, produce **five** 3D feature maps at
strides `(1, 2, 4, 8, 16)` with channels `(32, 64, 128, 256, 512)`, **finest first**.
Everything downstream — decoder, loss, schedule, evaluator — is provided by the
framework. See [HEAD_DESIGN.md](HEAD_DESIGN.md) for why the contract is shaped this way.

Two rules are not optional:

1. **Every trainable post-encoder parameter lives under `self.adapter`.**
   `freeze_encoder()` freezes all backbone parameters *except* those reachable through
   `self.adapter`. Get this wrong and your channel projections either never train
   (frozen at random init) or your "frozen" run silently fine-tunes the encoder.
2. **If a level cannot come from the pretrained encoder, say so.** A fresh conv branch
   on the raw input volume makes a frozen run un-interpretable as a representation
   probe. That is a legitimate design (see `spm`, `vit_adapter`, and Arm S) but it must
   be declared, because `scripts/gen_lowshot_configs.py` uses it to set
   `probe_comparable` in the low-shot manifest.

## Skeleton

`unified/models/backbones/my_model.py`:

```python
import torch
import torch.nn as nn

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from ._neck import ChannelNeck, UpsampleNeck, LayerTapNeck, SpatialPriorModule3D


class _MyAdapter(nn.Module):
    """Everything trainable that sits after the pretrained encoder."""

    def __init__(self, native_channels, contract_channels):
        super().__init__()
        # One 1x1x1 projection per contract level is the cheapest honest lift.
        self.necks = nn.ModuleList([
            nn.Conv3d(c_in, c_out, kernel_size=1)
            for c_in, c_out in zip(native_channels, contract_channels)
        ])

    def forward(self, native, input_shape):
        return [neck(f) for neck, f in zip(self.necks, native)]


@register_backbone("my_model")
class MyModelBackbone(BackboneInterface):

    def __init__(self, weights: str = None, **kwargs):
        super().__init__()
        self.encoder = build_my_encoder(**kwargs)

        # `weights=None` must be legal: it is how the scratch twin
        # (`model.pretrained: false`) and CPU shape tests are built.
        if weights:
            state = torch.load(weights, map_location="cpu", weights_only=False)
            state = state.get("state_dict", state)
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            _assert_loaded(missing, unexpected)     # see gotchas below

        self.adapter = _MyAdapter(
            native_channels=(...),
            contract_channels=self.EXPECTED_CHANNELS,
        )

    # Preferred: split the forward so SegModel can run the encoder under no_grad.
    def encoder_forward(self, x):
        return self.encoder.get_features(x)        # whatever API it exposes

    def adapter_forward(self, native, input_shape):
        return self.adapter(native, input_shape)

    def forward_features(self, x):
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])
```

`BackboneInterface.forward_features` is shape-asserted against
`EXPECTED_STRIDES` / `EXPECTED_CHANNELS` on every call, so a mismatch raises rather
than training silently.

### Filling levels the encoder does not own

Reusable pieces in `unified/models/backbones/_neck.py`:

| Situation | Use |
|---|---|
| native hierarchy, channels only need remapping | `ChannelNeck` |
| a level is **coarser** than the deepest native feature | strided 3×3×3 conv on the adapted deepest level |
| a level is **finer** than any native feature, hierarchical encoder | a light conv stem on raw input (declare it) |
| single bottleneck, want all five levels | `UpsampleNeck` (probe-honest) |
| transformer with many same-scale blocks | `LayerTapNeck` — one block per level, earliest → finest (probe-honest, preferred) |
| you explicitly want a raw-input pyramid | `SpatialPriorModule3D` (**not** probe-honest) |

Prefer `LayerTapNeck` over `UpsampleNeck` for any multi-block transformer: same
parameter count, and the blocks are computed either way.

### Arm S compatibility

If your adapter does **not** already run a raw-input branch, it is automatically
eligible for Arm S — `StemFusionBackbone` will wrap it and fuse a shared stem into
levels 0–3 for a fixed 349,584 parameters. Nothing is required of you. If your adapter
*does* run one, add its `pyramid_mode` to `_ALREADY_HAS_STEM` in
`unified/models/stem_fusion.py` so the wrapper refuses instead of stacking two stems.

## Config

`configs/models/my_model.yaml`:

```yaml
model:
  name: my_model
  weights: /store/home/skrljl/projects/foundation_models/weights/MyModel/best.pt
  batch_size: 2
  grad_accum_steps: 2
  # The normalization the model was PRETRAINED with. This is part of the model's
  # input interface, not a recipe knob — matching it is what makes the comparison
  # fair. Modes: range | percentile | sigmoid | znorm (see data/transforms.py).
  preprocessing:
    axcodes: RAS
    intensity:
      a_min: -1024.0
      a_max: 2048.0
      b_min: 0.0
      b_max: 1.0
      clip: true
  kwargs:
    depth: 12
    embed_dim: 768
```

A model config may set `model:` freely and deep-merge onto `head:` / `data:` / `eval:`.
It may **not** touch `train:` — the loader raises `ConfigError`. That lock is what keeps
the optimizer, schedule and loss identical across encoders.

## Test it

```bash
# construct, load weights, dummy (1,1,96,96,96) forward, assert contract + logits
python -m scripts.verify_setup --config configs/models/my_model.yaml

# forward + backward + optimizer step on a random tensor (CPU-sized patch)
python -m scripts.smoke_compute --config configs/models/my_model.yaml --no-weights

# same, but exercising the frozen path
python -m scripts.smoke_compute --config configs/models/my_model.yaml \
    --no-weights --freeze-backbone

# one real TotalSegmentator volume through the full transform pipeline
python -m scripts.smoke_train --config configs/models/my_model.yaml
```

Then check the frozen path specifically: every encoder parameter must have
`requires_grad=False` and every `self.adapter` parameter must receive gradient. Print the
frozen-trainable count and compare it against the table in
[HEAD_DESIGN.md §6](HEAD_DESIGN.md#6-measured-adapter-budgets) — if it is millions
rather than hundreds of thousands, something in the encoder is still training.

## Add it to the low-shot matrix

`scripts/gen_lowshot_configs.py` generates the 4 conditions × 5 fractions per arm:

- probe-comparable encoder → add to `BACKBONES`
- one-neck-varied ablation of an existing entry → `ABLATIONS`
- matched-stem twin → `ARM_S`
- best-effort, non-probe recipe → `EXTRA_ARMS`

Then regenerate and check the manifest columns are what you expect:

```bash
python -m scripts.gen_lowshot_configs
awk -F, 'NR>1{print $10, $8, $9}' configs/lowshot/MANIFEST.csv | sort | uniq -c
```

## Gotchas

- **Silent partial weight loads are the worst bug in this repo's history.** `dino3d`
  constructed a ViT-L with DINOv2's default `block_chunks=1` (expecting
  `blocks.0.{0..23}`) while the checkpoint stored `blocks.{0..3}.{global_idx}`, so
  `strict=False` matched only 6 of 24 blocks — **26 % of the encoder loaded** — and
  produced a plausible-looking benchmark number that was meaningless. Always assert on
  the `missing` / `unexpected` lists (`_assert_vit_loaded` is the pattern) and never
  ship a loader whose failure mode is a slightly-worse Dice.
- **Weight prefixes.** Checkpoints wrap state under `state_dict`, `network_weights`,
  `student`, `teacher`, `model`, or prefix keys with `module.`, `backbone.`,
  `teacher_backbone.`. Strip, then assert.
- **`weights=None` must work.** The scratch twin (`model.pretrained: false`) passes
  `weights=None`; guard the load with `if weights:`.
- **`SyncBatchNorm`.** Upstream research code often uses it. Convert to `GroupNorm`
  (see `_syncbn_to_gn`) so single-GPU small-batch training is unaffected by batch size.
- **Resizing inside the adapter.** Legal (`ctclip` → 480², `sam_med3d` → 128³) but it
  breaks the "identical input" invariant: FOV stays constant, effective resolution does
  not. Document it in the adapter docstring — it belongs in any write-up.
- **Memory.** ViT-L and SwinUNETR-H at 96³ / batch 2 already sit around 24 GB. Set
  `model.batch_size` and `model.grad_accum_steps` in the model config so the *effective*
  batch matches the rest of the suite, and say why.
