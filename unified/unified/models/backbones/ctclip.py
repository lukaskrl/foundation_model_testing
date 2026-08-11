"""CT-CLIP adapter.

CT-CLIP's image encoder is CTViT — a 3D VAE-ish ViT producing a single
bottleneck `(B, dim, T/temporal_patch, H/patch, W/patch)`. CTViT was
pretrained on **480×480** in-plane patches with temporal patches of
``temporal_patch_size`` frames, so the encoder is run on a resized,
depth-padded copy of the input and the bottleneck is anisotropic
(different temporal vs spatial strides relative to the resized input).

There are no intermediate skip features. Three ``pyramid_mode`` options:

``spm``
    Fine levels (strides 1, 2, 4, 8) come from a lightweight
    ``SpatialPriorModule3D`` running on the **raw input volume** — a fresh
    conv branch, the only source of native-resolution detail. Mirrors the
    dino3d adapter. NOT probe-honest: a frozen run partly measures that
    fresh branch rather than the pretrained representation.

``upsample``
    Every level is a channel-projected, trilinearly resampled copy of the one
    bottleneck. Probe-honest but the fine levels carry no information the
    coarse one lacks.

``layerwise``
    Read the residual stream at several transformer layers of a **single**
    pass and give each contract level its own tap, earliest tap to the finest
    level — the convention plain-ViT dense prediction settled on (UNETR, DPT).
    Probe-honest (no raw-input branch), parameter-matched to ``upsample`` (one
    1×1 projection per level), and one encoder pass, so it is strictly cheaper
    than ``multiscale``.

    What it does and does not buy. CTViT is **columnar**: it never downsamples,
    so all eight layers emit the same token grid at the same width (512). Taps
    therefore differ in *context*, not scale — the first stack has mixed
    in-plane but not yet along depth, and each further layer adds abstraction on
    the same grid. This removes the ``upsample`` defect that every level is the
    same tensor (the decoder's skips carried nothing), but it does NOT relax the
    token-scale bound: the finest level is still a resampled stride-16 grid.
    Nothing in CTViT is finer than one token — resolution exists only at the
    tokenizer and its linear transpose (``to_pixels``, a per-token expansion to
    a 10×20×20 voxel block), so every representation the network holds, encoder
    or reconstruction decoder, is bounded by 512 dims per such block.

    Because the stream is residual, taps are nested rather than independent: a
    late tap still carries what an early one contributed. The fine levels get
    less-abstracted access to information that largely survives downstream, not
    information that would otherwise be lost — so expect a smaller effect than a
    CNN pyramid's skips. Worth running as an A/B against ``upsample`` rather
    than assuming.

    Default taps are the last ``NUM_LEVELS`` layers in execution order, which
    for CTViT's 4 spatial + 4 temporal stacks is the in-plane stack's output
    plus each of the four depth layers. The last tap is *exactly* the
    ``upsample`` bottleneck, so the two modes are a clean single-variable pair.
    Note the depth-context ladder this induces: the finest level has seen no
    depth mixing at all, the coarsest has seen all of it.

``multiscale``
    Run the pretrained encoder **several times at different input canvases**
    and let each contract level draw from the pass whose native token grid is
    closest to it. Probe-honest (no raw-input branch) *and* genuinely
    multi-scale. Parameter-matched to ``upsample`` — one 1×1 projection per
    level — so the two are a clean A/B pair.

    This is possible because CTViT's spatial position bias is an MLP over
    relative coordinates (``cache_rel_pos=False``) and its PEG is a conv, so
    both are size-agnostic; the fixed 480 canvas is an artifact of upstream
    ``encode()`` reading ``patch_height_width`` rather than an architectural
    constraint. ``_encode`` below derives the token grid from the tensor
    instead, which is a strict generalization (identical at 480).

    Canvas choice is physical, not arbitrary. CTViT pretrained on
    480×480×240 at 0.75/0.75/1.5 mm = a 360 mm isotropic cube at a 24³ token
    grid, i.e. **15 mm of anatomy per token on every axis**. For a 96³ patch
    at 1.5 mm (144 mm FOV) the default canvases give 9.0 / 14.4 / 24.0 mm per
    token — bracketing the pretraining footprint. The default 480 canvas puts
    it at 6.0 mm/token, 2.5x off, and then discards 4x of the in-plane token
    grid resampling 24 -> 6. The three default passes together cost roughly a
    quarter of one 480 pass (spatial attention is quadratic in token count:
    36 + 100 + 256 tokens vs 576).

    Note the depth axis is NOT multi-scale: the z token count is
    ``ceil(D / temporal_patch_size)`` regardless of canvas, so every level
    resamples z from the same 10 tokens (15 mm granularity). Fixing that needs
    shifted-offset passes, which this mode does not do.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_backbone
from ..seg_model import BackboneInterface
from .dino3d import SpatialPriorModule3D
from ._neck import LayerTapNeck, MultiCanvasNeck, UpsampleNeck

CTCLIP_REPO = Path("/store/home/skrljl/projects/foundation_models/CT-CLIP")


def _import_ctvit():
    """Load CTViT bypassing transformer_maskgit/__init__.py.

    The package __init__.py pulls in transformers, T5, and a lot of other
    things we don't need for the image encoder. We load ctvit.py and its only
    sibling dependency (attention.py) by file path.
    """
    pkg_dir = CTCLIP_REPO / "transformer_maskgit" / "transformer_maskgit"
    if not pkg_dir.exists():
        raise NotImplementedError(
            f"CT-CLIP source missing at {pkg_dir}. Ensure the CT-CLIP repo is "
            "cloned at /store/home/skrljl/projects/foundation_models/CT-CLIP."
        )
    import importlib.util
    import types
    pkg_name = "transformer_maskgit"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_dir)]
        sys.modules[pkg_name] = pkg

    def _load(modname, filename):
        full = f"{pkg_name}.{modname}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, pkg_dir / filename)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("attention", "attention.py")
    ctvit_mod = _load("ctvit", "ctvit.py")
    return ctvit_mod.CTViT


def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


def _assign_level_sources(n_canvases: int, n_levels: int = 5):
    """Map each contract level to one encoder pass (canvases sorted ascending).

    Smallest canvas -> coarsest level (few tokens, large physical extent per
    token = global context); largest canvas -> finest levels (many tokens,
    small extent = local detail). Levels finer than the largest canvas can
    serve share it and are resampled up.

    With a single canvas this degenerates to every level using that one pass,
    i.e. exactly ``UpsampleNeck`` behaviour.
    """
    sources = []
    for level in range(n_levels):
        # level 0 = finest. Coarsest level takes canvas 0 (the smallest).
        idx = (n_levels - 1) - level
        sources.append(min(idx, n_canvases - 1))
    return sources


class _CTClipAdapter(nn.Module):
    """Adapt the CTViT bottleneck(s) onto the contract pyramid.

    ``pyramid_mode``: ``"spm"`` = SPM on raw input (strides 1..8) + 1×1
    projection of the CTViT bottleneck (stride 16); ``"upsample"`` = every
    level is a channel-projected, trilinearly resampled copy of the single
    bottleneck (``UpsampleNeck``); ``"multiscale"`` = per-level projection of
    several encoder passes at different canvases (``MultiCanvasNeck``);
    ``"layerwise"`` = per-level projection of several residual-stream taps from
    one pass (``LayerTapNeck``). All but ``spm`` have no raw-input path and are
    the probe-honest options.
    """

    def __init__(self, ctvit_dim: int, contract_channels, spm_inplanes: int = 16,
                 pyramid_mode: str = "spm", n_canvases: int = 1):
        super().__init__()
        self.pyramid_mode = pyramid_mode
        if pyramid_mode == "spm":
            self.spm = SpatialPriorModule3D(
                in_channels=1,
                inplanes=spm_inplanes,
                out_channels=contract_channels[:4],
            )
            c_top = contract_channels[4]
            self.vit_proj = nn.Sequential(
                nn.Conv3d(ctvit_dim, c_top, kernel_size=1, bias=False),
                _gn(c_top),
                nn.ReLU(inplace=True),
            )
        elif pyramid_mode == "upsample":
            self.upsample_neck = UpsampleNeck(
                in_channels=ctvit_dim,
                contract_channels=contract_channels,
            )
        elif pyramid_mode == "multiscale":
            self.multi_neck = MultiCanvasNeck(
                in_channels=ctvit_dim,
                level_sources=_assign_level_sources(n_canvases, len(contract_channels)),
                contract_channels=contract_channels,
            )
        elif pyramid_mode == "layerwise":
            # Identity mapping: taps arrive ascending by layer index and the
            # contract is finest-first, so the earliest tap feeds the finest
            # level. Ascending order is enforced in CTClipBackbone.__init__.
            self.layer_neck = LayerTapNeck(
                in_channels=ctvit_dim,
                level_sources=tuple(range(len(contract_channels))),
                contract_channels=contract_channels,
            )
        else:
            raise ValueError(
                f"unknown pyramid_mode {pyramid_mode!r}; expected 'spm', "
                "'upsample', 'multiscale' or 'layerwise'"
            )


@register_backbone("ctclip")
class CTClipBackbone(BackboneInterface):
    def __init__(
        self,
        weights: str,
        dim: int = 512,
        codebook_size: int = 8192,
        image_size: int = 480,
        patch_size: int = 20,
        temporal_patch_size: int = 10,
        spatial_depth: int = 4,
        temporal_depth: int = 4,
        dim_head: int = 32,
        heads: int = 8,
        use_image_encoder_only: bool = True,
        spm_inplanes: int = 16,
        pyramid_mode: str = "spm",
        canvases=None,
        tap_layers=None,
    ):
        super().__init__()
        CTViT = _import_ctvit()
        self.encoder = CTViT(
            dim=dim,
            codebook_size=codebook_size,
            image_size=image_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_depth=spatial_depth,
            temporal_depth=temporal_depth,
            dim_head=dim_head,
            heads=heads,
        )

        if weights:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            visual = {
                k.split("visual_transformer.", 1)[1]: v
                for k, v in state.items()
                if "visual_transformer." in k
            }
            if not visual:
                visual = state
            self.encoder.load_state_dict(visual, strict=False)

        # CTViT's ContinuousPositionBias has a hard-coded device=cuda; patch
        # so it respects the passed `device` argument.
        self._patch_pos_bias(self.encoder.spatial_rel_pos_bias)

        # Encoder input canvases. Only used by pyramid_mode="multiscale"; the
        # other modes run the single pretrained canvas. Sorted ascending so
        # `_assign_level_sources` can map smallest -> coarsest level.
        if canvases is None:
            canvases = (120, 200, 320) if pyramid_mode == "multiscale" else (image_size,)
        self.canvases = tuple(sorted(int(c) for c in canvases))
        bad = [c for c in self.canvases if c % patch_size != 0]
        if bad:
            raise ValueError(
                f"canvases {bad} are not divisible by patch_size {patch_size}; "
                "CTViT tokenizes with a non-overlapping patch grid"
            )

        # Residual-stream tap points. Only used by pyramid_mode="layerwise".
        # Indices run over BOTH stacks in execution order: 0..spatial_depth-1
        # after each in-plane layer, then the temporal layers. The default takes
        # the last NUM_LEVELS of them, which for CTViT's 4+4 is the in-plane
        # stack's output plus each of the four depth layers -- so the coarsest
        # level receives exactly the `upsample` bottleneck.
        self.n_encoder_layers = int(spatial_depth) + int(temporal_depth)
        self.tap_layers = None
        if pyramid_mode == "layerwise":
            n_levels = len(self.EXPECTED_CHANNELS)
            if tap_layers is None:
                if self.n_encoder_layers < n_levels:
                    raise ValueError(
                        f"pyramid_mode='layerwise' needs at least {n_levels} encoder "
                        f"layers to tap, but spatial_depth + temporal_depth = "
                        f"{self.n_encoder_layers}. Pass explicit `tap_layers` (taps may "
                        "repeat) or use pyramid_mode='upsample'."
                    )
                tap_layers = list(range(self.n_encoder_layers - n_levels,
                                        self.n_encoder_layers))
            tap_layers = [int(i) for i in tap_layers]
            if len(tap_layers) != n_levels:
                raise ValueError(
                    f"tap_layers must have one entry per contract level "
                    f"({n_levels}); got {len(tap_layers)}"
                )
            bad = [i for i in tap_layers if not 0 <= i < self.n_encoder_layers]
            if bad:
                raise ValueError(
                    f"tap_layers {bad} out of range for {self.n_encoder_layers} "
                    "encoder layers"
                )
            if tap_layers != sorted(tap_layers):
                raise ValueError(
                    f"tap_layers must be ascending so the earliest tap feeds the "
                    f"finest contract level; got {tap_layers}"
                )
            self.tap_layers = tuple(tap_layers)

        self.adapter = _CTClipAdapter(
            ctvit_dim=dim,
            contract_channels=self.EXPECTED_CHANNELS,
            spm_inplanes=spm_inplanes,
            pyramid_mode=pyramid_mode,
            n_canvases=len(self.canvases),
        )

    @staticmethod
    def _patch_pos_bias(pos_bias_module):
        import types
        from einops import rearrange

        def forward(self, *dimensions, device=None):
            if device is None:
                device = torch.device("cpu")
            if not getattr(self, "rel_pos", None) is not None or not self.cache_rel_pos:
                positions = [torch.arange(d, device=device) for d in dimensions]
                grid = torch.stack(torch.meshgrid(*positions, indexing="ij"))
                grid = rearrange(grid, "c ... -> (...) c")
                rel_pos = rearrange(grid, "i c -> i 1 c") - rearrange(grid, "j c -> 1 j c")
                if self.log_dist:
                    rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)
                self.register_buffer("rel_pos", rel_pos, persistent=False)
            rel_pos = self.rel_pos.to(torch.float32).to(device)
            for layer in self.net:
                rel_pos = layer(rel_pos.float())
            return rearrange(rel_pos, "i j h -> h i j")

        pos_bias_module.forward = types.MethodType(forward, pos_bias_module)

    def _forward_stacks(self, tokens, tap_layers=None):
        """Run both encoder stacks; optionally read the residual stream between
        layers.

        Reimplementation of ``CTViT.encode`` that differs from upstream in two
        ways, both strict generalizations:
          * no hard-coded ``torch.device('cuda')`` (upstream ctvit.py lines
            292, 332);
          * the token grid ``(h, w)`` is read off the token tensor rather than
            from ``self.encoder.patch_height_width``, so a canvas other than
            the pretrained 480 works. Legal because the spatial position bias
            is a coordinate MLP and the PEG is a conv — neither is tied to a
            token count. Identical behaviour at the 480 canvas.

        The per-layer loop is also inlined from ``Transformer.forward``
        (attention.py) because upstream returns only ``norm_out(x)`` and offers
        no hidden-state hook. Block order — PEG, self-attention, feed-forward,
        each residual — is copied exactly; ``cross_attn`` is always None here
        (CTViT builds its stacks with ``has_cross_attn=False``), and ``encode``
        passes no ``self_attn_mask``, so both are omitted rather than threaded
        through. The final output is bit-identical to upstream's.

        ``tap_layers``: iterable of layer indices over BOTH stacks in execution
        order (``0 .. spatial_depth-1`` in-plane, then the temporal layers).
        Each requested tap gets its own stack's ``norm_out`` applied — the
        residual stream is unnormalized between layers and its scale grows with
        depth — while the stream itself continues un-normalized, so collecting
        taps cannot change the final output.

        Returns ``(bottleneck, taps)``, all in ``(b, t, h, w, d)`` layout, taps
        ascending by layer index (empty when ``tap_layers`` is None). The tap at
        the last temporal layer is exactly the bottleneck.
        """
        from einops import rearrange
        wanted = set() if tap_layers is None else {int(i) for i in tap_layers}
        b, _, h, w, _ = tokens.shape
        video_shape = tuple(tokens.shape[:-1])
        taps = {}

        # --- in-plane stack: attention within each slab, no depth mixing ---
        sp = self.encoder.enc_spatial_transformer
        x = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        attn_bias = self.encoder.spatial_rel_pos_bias(h, w, device=x.device)
        for i, (peg, self_attn, _cross_attn, ff) in enumerate(sp.layers):
            if peg is not None:
                x = peg(x, shape=video_shape) + x
            x = self_attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x
            if i in wanted:
                taps[i] = rearrange(
                    sp.norm_out(x), "(b t) (h w) d -> b t h w d", b=b, h=h, w=w
                )
        x = rearrange(sp.norm_out(x), "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)

        # --- depth stack: attention along t, no spatial bias (as upstream) ---
        tp = self.encoder.enc_temporal_transformer
        offset = len(sp.layers)
        x = rearrange(x, "b t h w d -> (b h w) t d")
        for i, (peg, self_attn, _cross_attn, ff) in enumerate(tp.layers):
            if peg is not None:
                x = peg(x, shape=video_shape) + x
            x = self_attn(x) + x
            x = ff(x) + x
            if offset + i in wanted:
                taps[offset + i] = rearrange(
                    tp.norm_out(x), "(b h w) t d -> b t h w d", b=b, h=h, w=w
                )
        x = rearrange(tp.norm_out(x), "(b h w) t d -> b t h w d", b=b, h=h, w=w)

        return x, [taps[i] for i in sorted(taps)]

    def _encode(self, tokens):
        """The single bottleneck — ``CTViT.encode`` behaviour. See
        ``_forward_stacks``."""
        bottleneck, _ = self._forward_stacks(tokens)
        return bottleneck

    def _to_tokens(self, x, canvas=None):
        """Resize input to ``canvas``, pad depth to the temporal patch, patchify.

        ``canvas=None`` uses the pretrained in-plane size (``image_size``).
        Depth is padded, never resized, so the z token count is
        ``ceil(D / temporal_patch_size)`` regardless of canvas.
        """
        B, C, D, H, W = x.shape
        if C != 1:
            raise ValueError("ct-clip adapter expects 1-channel CT input")
        if canvas is not None:
            target_hw = (int(canvas), int(canvas))
        else:
            target_hw = (
                (self.encoder.image_size, self.encoder.image_size)
                if isinstance(self.encoder.image_size, int)
                else tuple(self.encoder.image_size)
            )
        flat = x.reshape(B * D, 1, H, W)
        flat = F.interpolate(flat, size=target_hw, mode="bilinear", align_corners=False)
        x_resized = flat.reshape(B, 1, D, *target_hw)

        tps = self.encoder.temporal_patch_size
        pad_d = (tps - (D % tps)) % tps
        if pad_d > 0:
            x_resized = F.pad(x_resized, (0, 0, 0, 0, 0, pad_d))

        return self.encoder.to_patch_emb(x_resized)     # (B, t', h', w', d)

    def _run_ctvit(self, x, canvas=None):
        """One encoder pass; return the bottleneck as (B, dim, t', h', w')."""
        tokens = self._encode(self._to_tokens(x, canvas))
        return tokens.permute(0, 4, 1, 2, 3).contiguous()

    def _run_ctvit_taps(self, x):
        """One encoder pass; return ``self.tap_layers`` residual-stream reads,
        each as (B, dim, t', h', w'), ascending by layer index."""
        _, taps = self._forward_stacks(self._to_tokens(x), tap_layers=self.tap_layers)
        return [t.permute(0, 4, 1, 2, 3).contiguous() for t in taps]

    def encoder_forward(self, x):
        mode = self.adapter.pyramid_mode
        if mode == "multiscale":
            # One pretrained-encoder pass per canvas, ascending. No raw input:
            # every contract level stays a function of the pretrained weights.
            return [self._run_ctvit(x, canvas=c) for c in self.canvases]
        if mode == "layerwise":
            # One pass, several reads of its residual stream. Also no raw input.
            return self._run_ctvit_taps(x)
        bottleneck = self._run_ctvit(x)
        # Pass raw input alongside the bottleneck so the trainable SPM in the
        # adapter can consume it.
        return [x, bottleneck]

    def adapter_forward(self, native, input_shape):
        mode = self.adapter.pyramid_mode
        if mode == "multiscale":
            return self.adapter.multi_neck(native, input_shape)
        if mode == "layerwise":
            return self.adapter.layer_neck(native, input_shape)
        x_in, bottleneck = native
        if mode == "upsample":
            return self.adapter.upsample_neck(bottleneck, input_shape)
        D, H, W = input_shape
        fine = self.adapter.spm(x_in)                       # strides 1..8
        s16 = self.adapter.vit_proj(bottleneck)             # 512 ch, anisotropic
        target = (D // 16, H // 16, W // 16)
        if tuple(s16.shape[2:]) != target:
            s16 = F.interpolate(s16, size=target, mode="trilinear", align_corners=False)
        return [*fine, s16]

    def forward_features(self, x):
        return self.adapter_forward(self.encoder_forward(x), x.shape[2:])
