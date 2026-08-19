"""3DINO-ViT backbone adapter with a SpatialPriorModule for fine levels.

A plain ViT has only one spatial scale (stride 16 with ``patch_size=16``).
Upsampling tokens cannot invent fine-grained detail, so the adapter splits
the contract pyramid:

  * ``feats[0..3]`` (strides 1, 2, 4, 8) come from a lightweight
    SpatialPriorModule3D — a small conv stem that runs on the *raw input
    volume*. The module's design is ported (and simplified) from
    ``3DINO/dinov2/eval/segmentation_3d/adapter_modules.py:SpatialPriorModule``
    but uses ``GroupNorm`` so it trains at small batch size on one GPU.
    Deformable Injector/Extractor blocks are intentionally omitted — they
    belong to the future deformable-head path.

  * ``feats[4]`` (stride 16) comes from the ViT's last (or last-of-extract)
    intermediate layer with a 1×1 channel projection (``embed_dim → 512``).
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from ._neck import (  # noqa: F401
    UpsampleNeck, ChannelNeck, LayerTapNeck, SpatialPriorModule3D,
)
# SpatialPriorModule3D moved to _neck.py (the generic StemFusionBackbone needs it
# too, and a shared wrapper must not import from a specific backbone). Re-exported
# here so `from .dino3d import SpatialPriorModule3D` in ctclip / sam_med3d keeps
# working.

DINO_REPO = Path("/store/home/skrljl/projects/foundation_models/3DINO")


def _import_dino_vit():
    if str(DINO_REPO) not in sys.path:
        sys.path.insert(0, str(DINO_REPO))
    from dinov2.models.vision_transformer import DinoVisionTransformer3d  # type: ignore
    return DinoVisionTransformer3d


def _import_vit_adapter():
    """The 3D port of ViT-Adapter (Chen et al.) shipped in the 3DINO repo.

    Provides bidirectional Injector/Extractor deformable-attention interaction
    between a SpatialPriorModule pyramid and the plain ViT, so the pretrained
    representation feeds every output scale rather than only stride-16.
    """
    if str(DINO_REPO) not in sys.path:
        sys.path.insert(0, str(DINO_REPO))
    from dinov2.eval.segmentation_3d.vit_adapter import ViTAdapter  # type: ignore
    return ViTAdapter


def _load_dino_weights(vit, weights):
    """Load a 3DINO checkpoint into a DinoVisionTransformer3d (strict=False)."""
    state = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for k in ("teacher", "student", "model"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    new = {}
    for k, v in state.items():
        nk = k
        for pref in ("module.", "backbone.", "teacher_backbone."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        new[nk] = v
    return vit.load_state_dict(new, strict=False)


def _assert_vit_loaded(missing, unexpected) -> None:
    """Fail loudly if a 3DINO checkpoint did not fully populate the ViT.

    ``load_state_dict(strict=False)`` is required because the checkpoint carries
    the SSL projection heads (``dino_head.*``, ``ibot_head.*``) that the backbone
    has no place for. But strict=False also silently swallows a *key-layout*
    mismatch, which is how the ``block_chunks`` bug went unnoticed: the ViT was
    built expecting ``blocks.0.{0..23}`` while the checkpoint stores
    ``blocks.{0..3}.{global_idx}``, so 18 of 24 transformer blocks stayed randomly
    initialized and training proceeded on a 26%-pretrained encoder.

    A correct load leaves ``missing`` empty and ``unexpected`` equal to exactly
    the SSL head tensors, so anything else is a construction/checkpoint mismatch.
    """
    if missing:
        raise RuntimeError(
            f"dino3d: {len(missing)} checkpoint keys did not load into the ViT — "
            f"construction does not match the checkpoint layout (block_chunks? "
            f"arch? init_values/LayerScale?). Sample: {list(missing)[:5]}"
        )
    stray = [u for u in unexpected if not u.startswith(("dino_head.", "ibot_head."))]
    if stray:
        raise RuntimeError(
            f"dino3d: {len(stray)} unexpected checkpoint keys beyond the SSL "
            f"heads — wrong checkpoint for this arch? Sample: {stray[:5]}"
        )


def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


def _syncbn_to_gn(module: nn.Module) -> nn.Module:
    """Replace every SyncBatchNorm/BatchNorm3d with GroupNorm in place.

    ViT-Adapter's SPM and final norms use ``SyncBatchNorm``, which needs an
    initialized process group and large batches. GroupNorm trains at batch
    size 1–2 on a single GPU and is batch-independent, matching the rest of
    the unified adapters. None of these norms are pretrained (only the ViT
    blocks are), so swapping the norm type loses no weights.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.SyncBatchNorm, nn.BatchNorm3d)):
            setattr(module, name, _gn(child.num_features))
        else:
            _syncbn_to_gn(child)
    return module


class _ViTAdapterContractNeck(nn.Module):
    """Lift ViT-Adapter's 4-level pyramid onto the 5-level contract.

    ViT-Adapter returns ``[f1, f2, f3, f4]`` at strides {2, 4, 8, 16}, all at
    ``embed_dim`` channels. The contract wants ``[g0..g4]`` at strides
    {1, 2, 4, 8, 16} with channels (32, 64, 128, 256, 512). We channel-project
    the four native levels and synthesize the missing stride-1 level with a
    learned 2× transposed conv off the finest level — so every contract level
    stays a function of the ViT-Adapter output, with no fresh raw-input branch.
    """

    def __init__(self, embed_dim: int, contract_channels):
        super().__init__()
        c0, c1, c2, c3, c4 = contract_channels
        self.up0 = nn.ConvTranspose3d(embed_dim, c0, kernel_size=2, stride=2, bias=False)
        self.gn0 = _gn(c0)
        self.act0 = nn.ReLU(inplace=True)
        self.neck1 = ChannelNeck(embed_dim, c1)
        self.neck2 = ChannelNeck(embed_dim, c2)
        self.neck3 = ChannelNeck(embed_dim, c3)
        self.neck4 = ChannelNeck(embed_dim, c4)

    def forward(self, feats4):
        f1, f2, f3, f4 = feats4                       # strides 2, 4, 8, 16
        g0 = self.act0(self.gn0(self.up0(f1)))        # stride 1
        return [g0, self.neck1(f1), self.neck2(f2), self.neck3(f3), self.neck4(f4)]


def _arch_kwargs(arch: str):
    # init_values enables LayerScale (ls1/ls2.gamma). The 3DINO checkpoints were
    # pretrained WITH LayerScale, so the module must exist for the gammas to load
    # and for the residual math to match pretraining. The magnitude is irrelevant
    # here (loaded gammas overwrite the init); it just has to be truthy.
    if arch == "vit_large_3d":
        return dict(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
                    qkv_bias=True, init_values=1.0e-5)
    if arch == "vit_base_3d":
        return dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                    qkv_bias=True, init_values=1.0e-5)
    raise ValueError(f"unknown 3DINO arch {arch}")


class _DinoAdapter(nn.Module):
    """Adapt ViT features onto the contract pyramid.

    ``pyramid_mode``:
      * ``"layerwise"`` (default): each contract level reads a DIFFERENT
        transformer block, earliest tap to the finest level — the DPT / UNETR
        convention for plain-ViT dense prediction. Probe-honest (no raw-input
        branch) and parameter-matched to ``upsample``, so the two are a clean
        single-variable pair.
      * ``"upsample"``: every level is a channel-projected, trilinearly
        upsampled copy of ONE block's tokens (``UpsampleNeck``). Kept as the
        ablation that quantifies what reading a single block costs.
      * ``"spm"``: SPM on the raw input (strides 1..8) + 1×1 projection of the
        ViT tokens (stride 16). The fine levels come from a fresh conv branch,
        so a frozen run is NOT a clean representation probe. Superseded by the
        generic ``StemFusionBackbone`` wrapper, which *fuses* rather than
        replaces; kept for backward compatibility.

    Why ``layerwise`` is the default. ``upsample`` gives all five levels to one
    tensor, so the decoder's skips are re-projections of the tensor that produced
    the thing they skip into — they carry nothing new. Worse, it reads 1 of the
    ViT's 24 blocks while the published recipes read 4, and the discarded blocks
    are computed either way. That was an unforced handicap, not a fairness
    choice.

    What it does NOT buy: resolution. Every block emits the same token grid, so
    the finest level is still a resampled stride-16 map and the tokenizer's 4:1
    compression is untouched. And because the residual stream is nested, a late
    tap still carries what an early one contributed — the fine levels get
    less-abstracted access to information that largely survived anyway, not
    information that would otherwise be gone. Expect a modest effect.
    """

    def __init__(self, embed_dim: int, contract_channels, spm_inplanes: int = 16,
                 pyramid_mode: str = "layerwise", level_sources=None):
        super().__init__()
        self.pyramid_mode = pyramid_mode
        if pyramid_mode == "layerwise":
            if level_sources is None:
                raise ValueError("pyramid_mode='layerwise' requires level_sources")
            self.layer_neck = LayerTapNeck(
                in_channels=embed_dim,
                level_sources=level_sources,
                contract_channels=contract_channels,
            )
        elif pyramid_mode == "spm":
            # Fine levels come from a SpatialPriorModule on raw input.
            self.spm = SpatialPriorModule3D(
                in_channels=1,
                inplanes=spm_inplanes,
                out_channels=contract_channels[:4],
            )
            # Semantic level (stride 16) is the ViT's tokens, channel-projected.
            c_top = contract_channels[4]

            def gn(ch):
                g = min(8, ch)
                while ch % g != 0:
                    g -= 1
                return nn.GroupNorm(g, ch)
            self.vit_proj = nn.Sequential(
                nn.Conv3d(embed_dim, c_top, kernel_size=1, bias=False),
                gn(c_top),
                nn.ReLU(inplace=True),
            )
        elif pyramid_mode == "upsample":
            self.upsample_neck = UpsampleNeck(
                in_channels=embed_dim,
                contract_channels=contract_channels,
            )
        else:
            raise ValueError(
                f"unknown pyramid_mode {pyramid_mode!r}; expected 'layerwise', "
                "'upsample' or 'spm'"
            )


@register_backbone("dino3d")
class DinoBackbone(BackboneInterface):
    def __init__(
        self,
        weights: str,
        arch: str = "vit_large_3d",
        img_size: int = 112,
        patch_size: int = 16,
        in_chans: int = 1,
        extract_blocks=(5, 11, 17, 23),
        spm_inplanes: int = 16,
        pyramid_mode: str = "layerwise",
        # vit_adapter-mode knobs (ignored by layerwise/spm/upsample modes)
        conv_inplane: int = 32,
        deform_num_heads: int = 8,
        drop_path_rate: float = 0.0,
        with_cp: bool = True,
    ):
        super().__init__()
        if patch_size != 16:
            raise ValueError("dino3d adapter assumes patch_size=16")
        self.pyramid_mode = pyramid_mode
        embed_dim = _arch_kwargs(arch)["embed_dim"]

        if pyramid_mode == "vit_adapter":
            self._build_vit_adapter(
                weights=weights, arch=arch, img_size=img_size,
                patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                conv_inplane=conv_inplane, deform_num_heads=deform_num_heads,
                drop_path_rate=drop_path_rate, with_cp=with_cp,
            )
            return

        DinoVisionTransformer3d = _import_dino_vit()
        self.vit = DinoVisionTransformer3d(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            # block_chunks=4 matches the checkpoint layout (blocks.{0..3}.{global_idx}).
            # The DINOv2 default of 1 expects blocks.0.{0..23}, so only chunk 0's
            # six blocks would match and the other 18 would be silently dropped by
            # strict=False (26% of the ViT loaded). See _assert_vit_loaded.
            block_chunks=4,
            **_arch_kwargs(arch),
        )
        # `layerwise` reads every entry; `upsample` / `spm` use only the deepest
        # for their single stride-16 tensor. The default (5, 11, 17, 23) is the
        # DPT / UNETR tap set for a 24-block ViT-L.
        self.extract_blocks = tuple(int(b) for b in extract_blocks)
        if not self.extract_blocks:
            raise ValueError("dino3d requires at least one extract_block")
        depth = _arch_kwargs(arch)["depth"]
        bad = [b for b in self.extract_blocks if not 0 <= b < depth]
        if bad:
            raise ValueError(
                f"extract_blocks {bad} out of range for a {depth}-block {arch}"
            )
        if list(self.extract_blocks) != sorted(self.extract_blocks):
            raise ValueError(
                f"extract_blocks must be ascending so the earliest tap feeds the "
                f"finest contract level; got {list(self.extract_blocks)}"
            )

        # Tap -> contract level. Ascending taps, earliest to the finest level:
        # the DPT / UNETR convention, since early blocks are less abstracted and
        # the fine levels are where local detail belongs. With fewer taps than
        # levels the deepest tap serves the remaining coarse levels — so the
        # COARSEST level always receives exactly the tensor `upsample` would have
        # used, which keeps upsample-vs-layerwise a clean single-variable A/B.
        n_levels = len(self.EXPECTED_CHANNELS)
        n_taps = len(self.extract_blocks)
        self.level_sources = tuple(min(i, n_taps - 1) for i in range(n_levels))

        if weights:
            _assert_vit_loaded(*_load_dino_weights(self.vit, weights))

        self.adapter = _DinoAdapter(
            embed_dim=embed_dim,
            contract_channels=self.EXPECTED_CHANNELS,
            spm_inplanes=spm_inplanes,
            pyramid_mode=pyramid_mode,
            level_sources=self.level_sources if pyramid_mode == "layerwise" else None,
        )

    def _build_vit_adapter(self, *, weights, arch, img_size, patch_size,
                           in_chans, embed_dim, conv_inplane, deform_num_heads,
                           drop_path_rate, with_cp):
        """Wrap the plain ViT in the repo's 3D ViT-Adapter.

        ``block_chunks=4`` matches both the checkpoint layout (``blocks.{0..3}.*``)
        and ViT-Adapter's 4-stage interaction schedule (``blocks[i]`` per stage).
        """
        DinoVisionTransformer3d = _import_dino_vit()
        ViTAdapter = _import_vit_adapter()
        vit = DinoVisionTransformer3d(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            block_chunks=4,
            **_arch_kwargs(arch),
        )
        if weights:
            _assert_vit_loaded(*_load_dino_weights(vit, weights))
        self.vit_adapter = ViTAdapter(
            vit_model=vit,
            input_channels=in_chans,
            pretrain_size=img_size,
            conv_inplane=conv_inplane,
            deform_num_heads=deform_num_heads,
            drop_path_rate=drop_path_rate,
            with_cp=with_cp,
            use_cls=True,
            add_vit_feature=True,
            use_extra_extractor=True,
        )
        # Adapt for single-GPU small-batch training (repo uses SyncBatchNorm).
        _syncbn_to_gn(self.vit_adapter)
        # Trainable contract lift lives under self.adapter so a frozen run
        # (freeze_encoder) keeps it learnable alongside the interaction blocks.
        self.adapter = _ViTAdapterContractNeck(embed_dim, self.EXPECTED_CHANNELS)

    def encoder_forward(self, x):
        if self.pyramid_mode == "vit_adapter":
            # The interaction blocks are interleaved *inside* the ViT forward,
            # so there is nothing to precompute under no_grad here — the whole
            # ViT-Adapter runs in adapter_forward (ViT params are frozen via
            # requires_grad, not via a no_grad wrapper).
            return [x]
        # `get_intermediate_layers` returns a tuple in the order of
        # `extract_blocks` (ascending), each already reshaped to (B, C, d, h, w)
        # and normed. The blocks are computed by the ViT either way — `upsample`
        # and `spm` simply discard all but the deepest.
        layers = self.vit.get_intermediate_layers(
            x,
            n=self.extract_blocks,
            reshape=True,
            return_class_token=False,
            norm=True,
        )
        if self.pyramid_mode == "layerwise":
            # Every tap reaches the adapter, ascending by block index.
            return [x, *layers]
        # Pass raw input alongside the ViT tokens so the trainable SPM can
        # consume it in the adapter step.
        return [x, layers[-1]]

    def adapter_forward(self, native, input_shape):
        if self.pyramid_mode == "vit_adapter":
            (x_in,) = native
            feats4 = self.vit_adapter(x_in)      # [stride2,4,8,16] @ embed_dim
            return self.adapter(feats4)          # -> contract [stride1..16]
        x_in, *rest = native
        if self.adapter.pyramid_mode == "layerwise":
            return self.adapter.layer_neck(rest, input_shape)
        (tokens,) = rest
        if self.adapter.pyramid_mode == "upsample":
            return self.adapter.upsample_neck(tokens, input_shape)
        fine = self.adapter.spm(x_in)            # 4 tensors @ strides 1..8
        s16 = self.adapter.vit_proj(tokens)      # stride 16
        return [*fine, s16]

    def forward_features(self, x):
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])

    def freeze_encoder(self) -> None:
        """Freeze only the pretrained ViT blocks; keep the adapter learnable.

        In ``vit_adapter`` mode the trainable machinery (SPM + Injector/
        Extractor interaction blocks) is bundled inside ``self.vit_adapter``
        alongside the frozen ViT, so the base-class rule ("freeze everything
        not under self.adapter") would wrongly freeze the interaction blocks.
        Here we freeze exactly the pretrained ViT and leave everything else —
        interaction blocks and the contract neck — trainable.
        """
        if self.pyramid_mode == "vit_adapter":
            for p in self.vit_adapter.vit_model.parameters():
                p.requires_grad_(False)
            return
        super().freeze_encoder()
