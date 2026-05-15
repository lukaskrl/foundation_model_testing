# Adding a backbone

The contract: produce four 3D feature maps at strides {4, 8, 16, 32} with channel
counts {64, 128, 256, 512}, from a (B, 1, D, H, W) CT input. Everything downstream is
provided by the framework.

## Skeleton

`unified/models/backbones/my_model.py`:

```python
import torch
import torch.nn as nn
from unified.models.seg_model import BackboneInterface
from unified.models.registry import register_backbone

@register_backbone("my_model")
class MyModelBackbone(BackboneInterface):

    NECK_OUT_CHANNELS = (64, 128, 256, 512)
    NECK_STRIDES = (4, 8, 16, 32)

    def __init__(self, weights: str, **kwargs):
        super().__init__()
        # 1. build the pretrained encoder
        self.encoder = build_my_encoder(**kwargs)

        # 2. load weights
        state = torch.load(weights, map_location="cpu")
        state = state.get("state_dict", state)  # unwrap if wrapped
        self.encoder.load_state_dict(state, strict=False)

        # 3. build the per-level neck (1x1x1 conv + optional resize) so that
        #    the encoder's native channels/strides become (64,128,256,512)
        #    at strides (4,8,16,32).
        self.necks = nn.ModuleList([
            self._make_neck(in_ch=..., out_ch=64,  resize_factor=...),
            self._make_neck(in_ch=..., out_ch=128, resize_factor=...),
            self._make_neck(in_ch=..., out_ch=256, resize_factor=...),
            self._make_neck(in_ch=..., out_ch=512, resize_factor=...),
        ])

    def forward_features(self, x):
        # forward through encoder, get native multi-scale features
        raw = self.encoder.get_features(x)   # whatever API the encoder exposes
        # select 4 levels and pass through necks
        out = []
        for level, neck in zip(self._select_levels(raw), self.necks):
            out.append(neck(level))
        return out  # List[Tensor], length 4, strides (4,8,16,32), channels (64,128,256,512)
```

## Config

`configs/models/my_model.yaml`:

```yaml
model:
  name: my_model
  weights: /store/home/skrljl/projects/foundation_models/weights/MyModel/best.pt
  kwargs:
    depth: 12
    embed_dim: 768
    # ... whatever the encoder constructor needs
```

## Test it

```bash
python -m scripts.verify_setup --config configs/models/my_model.yaml
```

This will construct the model, load weights, run a dummy (1,1,96,96,96) forward, and
check that the output is (1, 118, 96, 96, 96).

## Common gotchas

- **Weight prefixes.** Pretrained checkpoints often wrap state dicts under `state_dict`,
  `network_weights`, `student`, `teacher`, or prefix keys with `module.` or `backbone.`.
  Strip these before `load_state_dict`. Use `strict=False` only after manually checking
  the missing/unexpected list — silent partial loads are a footgun.
- **Frozen encoder.** Don't freeze the encoder unless your config asks for it. The
  default in `base.yaml` is full fine-tuning.
- **Memory.** ViT-Large and SwinUNETR-H at patch 96³ batch 2 already use ~24 GB. If
  you add a larger backbone, also add `train.batch_size_override` to its config and
  document why it deviates.
