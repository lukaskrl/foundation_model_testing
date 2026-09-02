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
from ._loading import assert_encoder_loaded
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
            # Mandatory. strict=False is required (releases carry decoders / SSL
            # heads this framework has no place for) but it also swallows a
            # key-layout mismatch that leaves the encoder random while the run
            # still reports `pretrained: true`. See gotchas below.
            assert_encoder_loaded(
                "my_model", self.encoder,
                self.encoder.load_state_dict(state, strict=False),
            )

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

`BackboneInterface.assert_contract` checks a feature list against
`EXPECTED_STRIDES` / `EXPECTED_CHANNELS`, but it is **not** called on the training path —
only `scripts/verify_setup.py` invokes it. A violation still fails loudly (the head's
`torch.cat` against a mismatched skip raises), but run `verify_setup` on any new adapter
to get the sharper error.

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

### Arm S / Arm W compatibility

Both arms are generated from your Arm N config, so adding your encoder to `BACKBONES`
gets you both for free. Two things are required of you:

- **If your adapter sources *most* of the contract (strides 1–8) from a raw-input conv
  branch**, add its `pyramid_mode` to `_ALREADY_HAS_STEM` in
  `unified/models/stem_fusion.py` so the wrapper refuses instead of stacking two stems.
  Set `self.pyramid_mode` on the **backbone**, not only on the inner adapter — the guard
  checks both (`_effective_pyramid_mode`), but the backbone attribute is the one to
  prefer. A *small* stem for the one or two levels your encoder cannot reach is fine to
  wrap and must **not** go in that set.
- **Declare what your Arm N fine levels are** in `ARM_N_FINE_SOURCE`
  (`scripts/gen_lowshot_configs.py`): `pretrained`, `fresh_stem_s0`, `fresh_stem_s0s1`,
  or `resampled_*`. It becomes the manifest's `arm_n_fine_source` column and is the key
  for reading `compensable` — `≈ 0` means "detail already routed" for an encoder that
  already had a stem, and "a stem does not help" for one that did not.

Arm S adds a bit-identical **349,584** parameters to every encoder. If your wrapped
model does not show exactly that delta, something is stacking or missing.

## Config

`configs/models/my_model.yaml`:

```yaml
model:
  name: my_model
  weights: /home/lukas/projects/foundation_model_testing/weights/MyModel/best.pt
  batch_size: 2
  grad_accum_steps: 2
  # The normalization the model was PRETRAINED with — part of the model's input
  # interface, not a recipe knob. Matching it is the defensible default, but it is
  # not automatically "fair": if your window clips information the task needs, an
  # Arm N ranking cannot separate that from representation quality. Arm W measures
  # the difference (HEAD_DESIGN.md §7), so state your window's clipping cost.
  # Modes: range | percentile | sigmoid | znorm (see data/transforms.py).
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
# construct, dummy (1,1,96,96,96) forward, assert contract + logits.
# NOTE: --load-weights is OFF by default, so this does NOT exercise your checkpoint
# guard. Pass it to test the real load.
python -m scripts.verify_setup --config configs/models/my_model.yaml --load-weights

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

- probe-comparable encoder → add to `BACKBONES` **and** to `ARM_N_FINE_SOURCE`
- one-neck-varied ablation of an existing entry → `ABLATIONS`
- best-effort, non-probe recipe → `EXTRA_ARMS`

Arm S and Arm W are **not** lists you add to: both are generated over every `BACKBONES`
entry, so a single `BACKBONES` line yields the Arm N, Arm S and Arm W rows together.
That is deliberate — one shared code path is what guarantees every encoder gets a
bit-identical stem and a bit-identical window.

Then regenerate and check the manifest columns are what you expect (columns are
positional and **append-only** — `run_lowshot_matrix.sh` reads `$1..$11` by index):

```bash
python -m scripts.gen_lowshot_configs
awk -F, 'NR>1{print $10, $11, $9, $12}' configs/lowshot/MANIFEST.csv | sort | uniq -c
```

## Gotchas

- **Silent partial weight loads are the worst bug in this repo's history.** `dino3d`
  constructed a ViT-L with DINOv2's default `block_chunks=1` (expecting
  `blocks.0.{0..23}`) while the checkpoint stored `blocks.{0..3}.{global_idx}`, so
  `strict=False` matched only 6 of 24 blocks — **26 % of the encoder loaded** — and
  produced a plausible-looking benchmark number that was meaningless. It has happened
  more than once: `voco` and `suprem_swinunetr` checked only the *unexpected* list,
  `ctclip` and `vista3d` checked nothing at all, and a wrong-prefix checkpoint
  constructed silently into a fully random encoder still labelled `pretrained: true`.
  **Always route the load through `assert_encoder_loaded`** — it requires every encoder
  key to be populated, and all twelve current encoders satisfy that against their real
  checkpoints, so it costs no legitimate load. Never ship a loader whose failure mode is
  a slightly-worse Dice.
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
