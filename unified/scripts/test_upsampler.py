"""Assert the properties the upsampler paper rests on (docs/UPSAMPLER_PLAN.md §1E).

  1. The two memory-saving autograd functions are exact: gradcheck in float64, and
     forward/backward equal to a naive implementation that gathers all neighbours.
  2. Partition of unity: constant features come out unchanged, for any CT and any
     grid. The output is a convex mix of the encoder's own vectors.
  3. Encoder-agnostic: one set of weights runs on 384, 512 and 1024 channels, and
     permuting the input channels permutes the output channels (also with the
     feature-agnostic key term on).
  4. Linear in the features for fixed weights, so a linear probe may be applied
     before upsampling: aggregate(w, W F) == W aggregate(w, F).
  5. At initialisation the module ignores the CT (zero queries): it starts as a
     fixed Gaussian interpolation. A constant CT gives the same output at any HU.
  6. Anisotropic, non-power-of-two and padded (CT-CLIP-like) grids work, and an
     output restricted to a sub-crop equals the same crop of the full output.
  7. Only the upsampler receives gradients; the features are never modified.
  8. Orientation helpers match MONAI's Orientation, and reoriented centers still
     describe where the reoriented cells are.

  9. (--encoders) each wrapped encoder returns RAS features with the stated cell
     centers; CT-FM's centers are derived from its conv layers.
 10. (--store) per-volume normalization reproduces the benchmark's MONAI transforms.
 11. (--interface CKPT) the upsampled skip in the I1 head adds no trainable parameters,
     trains the projection through the skip, and reads the HU channel only if guided.

Usage:  python -m scripts.test_upsampler              # CPU, ~20 s
        python -m scripts.test_upsampler --encoders   # + random-init encoders, CPU
        python -m scripts.test_upsampler --store /home/lukas/data/cache/upsampler/vol15
        python -m scripts.test_upsampler --bench      # + time/memory at 96^3 (CUDA)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from unified.upsampler.geometry import (  # noqa: E402
    apply_ornt, centers_resize, centers_stride, neighbor_table, ornt_between,
    pool_matrices, pool_to_grid, reorient_centers, default_centers)
from unified.upsampler.module import (  # noqa: E402
    GuidedUpsampler3D, BilateralUpsampler, TrilinearUpsampler, aggregate, neighbor_dot,
    _gather, _offset_indices)

PASSED = []


def check(name):
    def deco(fn):
        def run():
            t0 = time.time()
            fn()
            PASSED.append(name)
            print(f"  ok  {name}  ({time.time() - t0:.1f}s)", flush=True)
        run.__name__ = fn.__name__
        return run
    return deco


def _rand_ct(B, shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, 1, *shape, generator=g) * 400.0


def _trained_like(m: GuidedUpsampler3D, seed=0):
    """Give the zero-initialised query projection random weights, so the CT matters."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.to_q.weight.copy_(torch.randn(m.to_q.weight.shape, generator=g) * 0.5)
    return m


# --------------------------------------------------------------------- 1
@check("custom autograd: gradcheck (float64)")
def t_gradcheck():
    tab = neighbor_table(default_centers((3, 4, 2), (6, 8, 5)),
                         default_centers((5, 7, 4), (6, 8, 5)), radius=1)
    q = torch.randn(1, 3, 5, 7, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 3, 3, 4, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: neighbor_dot(a, b, tab), (q, k), fast_mode=True)
    w = torch.rand(1, 27, 5, 7, 4, dtype=torch.float64, requires_grad=True)
    f = torch.randn(1, 5, 3, 4, 2, dtype=torch.float64, requires_grad=True)
    # chunk=2 exercises the channel-chunk loop
    assert torch.autograd.gradcheck(lambda a, b: aggregate(a, b, tab, 2), (w, f), fast_mode=True)


@check("custom autograd and dense GEMM path: equal naive all-neighbour gather")
def t_naive():
    tab = neighbor_table(default_centers((4, 3, 5), (16, 12, 20)),
                         default_centers((8, 6, 10), (16, 12, 20)))
    q0, k0, f0 = torch.randn(2, 4, 8, 6, 10), torch.randn(2, 4, 4, 3, 5), torch.randn(2, 7, 4, 3, 5)
    r = torch.randn(2, 7, 8, 6, 10)

    def fresh():
        return [t.clone().requires_grad_() for t in (q0, k0, f0)]

    q, k, f = fresh()
    out = aggregate(neighbor_dot(q, k, tab).softmax(1), f, tab, chunk=3)
    grads = torch.autograd.grad((out * r).sum(), (q, k, f))

    q2, k2, f2 = fresh()
    K = torch.stack([_gather(k2, *ix) for ix in _offset_indices(tab)], 1)   # B,27,C,*
    Fg = torch.stack([_gather(f2, *ix) for ix in _offset_indices(tab)], 1)
    out2 = ((q2[:, None] * K).sum(2).softmax(1)[:, :, None] * Fg).sum(1)
    grads2 = torch.autograd.grad((out2 * r).sum(), (q2, k2, f2))

    assert torch.allclose(out, out2, atol=1e-5)
    for x, y in zip(grads, grads2):
        assert torch.allclose(x, y, atol=1e-5), (x - y).abs().max()

    # the dense GEMM path gives the same values and gradients as the offset loop
    grads_by_path = []
    for dense in (False, True):
        q, k, f = fresh()
        o = aggregate(neighbor_dot(q, k, tab).softmax(1), f, tab, chunk=3, dense=dense)
        grads_by_path.append((o, torch.autograd.grad((o * r).sum(), (q, k, f))))
    (o1, g1), (o2, g2) = grads_by_path
    assert torch.allclose(o1, o2, atol=1e-5)
    for x, y in zip(g1, g2):
        assert torch.allclose(x, y, atol=1e-5), (x - y).abs().max()


# --------------------------------------------------------------------- 2
@check("partition of unity: constant features pass through unchanged")
def t_unity():
    for fk in (False, True):
        m = _trained_like(GuidedUpsampler3D(feature_keys=fk)).eval()
        ct = _rand_ct(2, (32, 24, 40))
        f = torch.ones(2, 16, 2, 3, 5) * torch.arange(16.0)[None, :, None, None, None]
        with torch.no_grad():
            out = m(ct, f)
        assert torch.allclose(out, f[:, :, :1, :1, :1].expand_as(out), atol=1e-4)
    b = BilateralUpsampler()
    with torch.no_grad():
        out = b(ct, f)
    assert torch.allclose(out, f[:, :, :1, :1, :1].expand_as(out), atol=1e-4)


# --------------------------------------------------------------------- 3
@check("encoder-agnostic: one module on 384/512/1024 channels, channel-equivariant")
def t_channels():
    for fk in (False, True):
        m = _trained_like(GuidedUpsampler3D(feature_keys=fk)).eval()
        ct = _rand_ct(1, (48, 48, 48))
        for C in (384, 512, 1024):
            f = torch.randn(1, C, 3, 3, 3)
            perm = torch.randperm(C)
            with torch.no_grad():
                o = m(ct, f)
                op = m(ct, f[:, perm])
            assert o.shape == (1, C, 24, 24, 24)
            assert torch.allclose(o[:, perm], op, atol=1e-4)


# --------------------------------------------------------------------- 4
@check("linear in features; linear probe commutes with upsampling")
def t_linear():
    m = _trained_like(GuidedUpsampler3D()).eval()
    ct = _rand_ct(2, (32, 32, 32))
    f, g = torch.randn(2, 64, 4, 4, 4), torch.randn(2, 64, 4, 4, 4)
    with torch.no_grad():
        lhs = m(ct, 2.0 * f - 3.0 * g)
        rhs = 2.0 * m(ct, f) - 3.0 * m(ct, g)
        assert torch.allclose(lhs, rhs, atol=1e-3)
        W = torch.randn(10, 64)
        bias = torch.randn(10)
        w, tab = m.weights(ct, f)
        a = aggregate(w, torch.einsum("kc,bcdhw->bkdhw", W, f) + bias[None, :, None, None, None], tab)
        b = torch.einsum("kc,bcdhw->bkdhw", W, aggregate(w, f, tab)) + bias[None, :, None, None, None]
        assert torch.allclose(a, b, atol=1e-3)


# --------------------------------------------------------------------- 5
@check("init ignores the CT; a constant CT gives the same output at any HU")
def t_constant_ct():
    m = GuidedUpsampler3D().eval()
    f = torch.randn(1, 32, 4, 4, 4)
    with torch.no_grad():
        a, b = m(_rand_ct(1, (32, 32, 32), 1), f), m(_rand_ct(1, (32, 32, 32), 2), f)
    assert torch.allclose(a, b, atol=1e-5), "untrained module should not read the CT"
    m = _trained_like(GuidedUpsampler3D()).eval()
    outs = []
    for hu in (-900.0, 0.0, 40.0, 700.0):
        with torch.no_grad():
            outs.append(m(torch.full((1, 1, 32, 32, 32), hu), f))
    for o in outs[1:]:
        assert torch.allclose(outs[0], o, atol=1e-5)
    with torch.no_grad():
        o_rand = m(_rand_ct(1, (32, 32, 32), 3), f)
    assert not torch.allclose(outs[0], o_rand, atol=1e-3), "trained module should read the CT"


# --------------------------------------------------------------------- 6
@check("odd, anisotropic and padded grids; sub-crop == crop of full; trilinear exact")
def t_grids():
    m = _trained_like(GuidedUpsampler3D(feature_keys=True)).eval()
    # CT-CLIP-like in RAS: 24 x 24 in-plane (resize), depth padded to 10-slice cells
    fine = (96, 96, 96)
    fc = (centers_resize(24, 96), centers_resize(24, 96), centers_stride(10, 10.0))
    ct = _rand_ct(1, fine)
    f = torch.randn(1, 8, 24, 24, 10)
    with torch.no_grad():
        out = m(ct, f, feat_centers=fc)
    assert out.shape == (1, 8, 48, 48, 48) and torch.isfinite(out).all()
    # odd sizes, non-integer strides
    ct2, f2 = _rand_ct(1, (37, 50, 23)), torch.randn(1, 5, 3, 7, 2)
    with torch.no_grad():
        o2 = m(ct2, f2, out_shape=(11, 50, 9))
    assert o2.shape == (1, 5, 11, 50, 9) and torch.isfinite(o2).all()
    # sub-crop: output cells 6..11 of a stride-8 grid over the patch
    full_c = default_centers((12, 12, 12), fine)
    sub_c = tuple(c[6:12] for c in full_c)
    f3 = torch.randn(1, 8, 6, 6, 6)
    with torch.no_grad():
        full = m(ct, f3, out_centers=full_c)
        sub = m(ct, f3, out_centers=sub_c)
    assert torch.allclose(full[:, :, 6:12, 6:12, 6:12], sub, atol=1e-5)
    # trilinear baseline equals F.interpolate on the default layout, any sizes
    for fs, os_ in (((6, 6, 6), (48, 48, 48)), ((3, 7, 2), (11, 50, 9))):
        ff = torch.randn(1, 8, *fs)
        ref = F.interpolate(ff, size=os_, mode="trilinear", align_corners=False)
        assert torch.allclose(TrilinearUpsampler()(ct, ff, out_shape=os_), ref, atol=1e-5)


# --------------------------------------------------------------------- 7
@check("gradients reach every upsampler parameter and never the features")
def t_grads():
    m = GuidedUpsampler3D(feature_keys=True)
    ct = _rand_ct(1, (32, 32, 32))
    f = torch.randn(1, 64, 2, 2, 2)
    f0 = f.clone()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for step in range(2):          # step 0 moves to_q off zero; step 1 reaches to_k
        opt.zero_grad(set_to_none=True)
        out = m(ct, f)
        (out - torch.randn_like(out)).pow(2).mean().backward()
        opt.step()
    assert f.grad is None and torch.equal(f, f0)
    missing = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, f"no gradient: {missing}"


# --------------------------------------------------------------------- 8
@check("orientation: matches MONAI; reoriented centers follow the cells")
def t_orientation():
    from monai.data import MetaTensor
    from monai.transforms import Orientation
    x = torch.randn(1, 5, 6, 7)
    for codes in ("SPL", "SRA", "LPS", "RAS", "PIR"):
        ref = Orientation(axcodes=codes)(MetaTensor(x.clone(), affine=torch.eye(4))).as_tensor()
        mine = apply_ornt(x, ornt_between("RAS", codes))
        assert mine.shape == ref.shape and torch.equal(mine, ref), codes
        assert torch.equal(apply_ornt(mine, ornt_between(codes, "RAS")), x), codes
        # centers: box-pool, then reorient == reorient, then box-pool at moved centers
        fine = (20, 24, 28)
        V = torch.randn(1, 1, *fine)
        cs = (centers_stride(3, 7.0, 3.0), centers_resize(4, 24), centers_stride(5, 5.0, 1.0))
        pooled = pool_to_grid(V, pool_matrices(cs, fine))
        o = ornt_between("RAS", codes)
        cs2, fine2 = reorient_centers(cs, fine, o)
        V2 = apply_ornt(V, o)
        assert tuple(V2.shape[2:]) == fine2
        assert torch.allclose(apply_ornt(pooled, o), pool_to_grid(V2, pool_matrices(cs2, fine2)),
                              atol=1e-5), codes


# --------------------------------------------------------------- encoders
ENCODERS = {
    # stem: (RAS feature shape at a 96^3 patch, expected RAS centers per axis)
    "dino3d_layerwise": ((1024, 6, 6, 6), [centers_stride(6, 16)] * 3),
    "sam_med3d": ((384, 8, 8, 8), [centers_resize(8, 96)] * 3),
    # SRA encoder: depth (S) is padded to 10-slice cells and lands on RAS axis 2
    "ctclip": ((512, 24, 24, 10), [centers_resize(24, 96), centers_resize(24, 96),
                                   centers_stride(10, 10.0, 4.5)]),
    # SPL encoder, cells at 16 i in its own frame; the P and L flips move them to
    # 16 i + 15 on RAS axes 0 and 1
    "ctfm": ((512, 6, 6, 6), [centers_stride(6, 16, 15.0), centers_stride(6, 16, 15.0),
                              centers_stride(6, 16, 0.0)]),
}


def t_encoders(stems):
    from unified.upsampler.encoders import FrozenEncoder
    for stem in stems:
        t0 = time.time()
        enc = FrozenEncoder(stem, device="cpu", pretrained=False)
        assert not any(p.requires_grad for p in enc.bb.parameters())
        shape, want = ENCODERS[stem]
        f, cs = enc(torch.randn(1, 1, 96, 96, 96))
        assert tuple(f.shape[1:]) == shape, (stem, f.shape)
        for c, w in zip(cs, want):
            assert torch.allclose(c.double(), w.double()), (stem, c, w)
        print(f"  ok  {stem}: RAS features {tuple(f.shape[1:])}, centers as expected "
              f"({time.time() - t0:.0f}s)", flush=True)
        if stem == "ctfm":
            _conv_chain_check(enc)
        PASSED.append(f"encoder {stem}")


def _conv_chain_check(enc):
    """Derive CT-FM's cell centers from its layers instead of trusting the comment.

    A gradient centroid does not work here: InstanceNorm couples every voxel, so at
    random init a deep cell's input gradient covers the whole patch. The layers do:
    for a conv (k, s, p) after cumulative stride S, the output center moves by
    S * ((k - 1) / 2 - p), so a chain of k3/p1 convs keeps offset 0 at any stride."""
    S, offset = 1, 0.0
    for name, m in enc.bb.encoder.named_modules():
        if isinstance(m, (torch.nn.MaxPool3d, torch.nn.AvgPool3d)):
            raise AssertionError(f"ctfm has a pooling layer {name}; revisit its centers")
        if isinstance(m, torch.nn.Conv3d):
            k, st, p = m.kernel_size[0], m.stride[0], m.padding[0]
            offset += S * ((k - 1) / 2 - p)
            S *= st
    assert S == 16 and offset == 0.0, (S, offset)
    print(f"  ok  ctfm layers: cumulative stride {S}, center offset {offset} (cells at 16 i)",
          flush=True)


def t_intensity(store_root, sid="s0000"):
    """Per-volume NormParams reproduce the benchmark's MONAI intensity transforms."""
    from monai.data import MetaTensor
    from monai.transforms import Compose, Orientationd
    from unified.data.transforms import _orient_intensity_list
    from unified.upsampler.data import VolumeStore
    from unified.upsampler.encoders import load_model_cfg, norm_params
    from unified.data.transforms import _resolved_preprocessing
    store = VolumeStore(store_root)
    vol = torch.from_numpy(store.image(sid).astype("float32"))[None]
    for stem in ("dino3d_layerwise", "sam_med3d", "ctclip", "ctfm"):
        cfg = load_model_cfg(stem)
        _, intensity = _resolved_preprocessing(cfg)
        tx = Compose([t for t in _orient_intensity_list(cfg) if not isinstance(t, Orientationd)])
        ref = tx({"image": MetaTensor(vol.clone()), "label": MetaTensor(vol.clone())})["image"]
        ref = ref.as_tensor() if hasattr(ref, "as_tensor") else ref
        mine = norm_params(intensity, vol[0].numpy()).apply(vol)
        err = float((mine - ref).abs().max())
        span = float(ref.max() - ref.min())
        assert err < 1e-3 * span, (stem, err, span)
        print(f"  ok  intensity {stem:17s} ({intensity.get('mode', 'range')}): "
              f"max err {err:.2e} of range {span:.2f}", flush=True)
    PASSED.append("intensity normalization")


# ------------------------------------------------------------ interface head
def t_interface(ckpt):
    """The upsampled skip inside the I1 interface head (dino3d, random init, CPU).

    Checks: the skip adds zero trainable parameters (guided and trilinear alike),
    the frozen upsampler receives no gradient, gradient reaches the stride-16
    projection through the skip, and the output depends on the HU channel for
    'guided' and not for 'trilinear'."""
    from unified.models import build_backbone, build_head, SegModel
    from unified.utils import load_config
    cfg = load_config(str(REPO / "configs/interface/dino3d_i1_frz_pt_f100.yaml"))
    m = cfg["model"]
    counts = {}
    torch.manual_seed(0)
    x = torch.randn(1, 1, 96, 96, 96)
    hu_a, hu_b = _rand_ct(1, (96, 96, 96), 1), _rand_ct(1, (96, 96, 96), 2)
    for variant in (None, "trilinear", "guided"):
        kw = dict(m["kwargs"])
        if variant:
            kw.update(interface_upsampler=variant, interface_upsampler_ckpt=ckpt)
        torch.manual_seed(0)
        bb = build_backbone(m["name"], weights=None, **kw)
        head = build_head(cfg["head"]["name"], num_classes=118, **bb.head_kwargs())
        model = SegModel(bb, head, freeze_backbone=True).train()
        counts[variant] = model.num_trainable_params()
        if variant is None:
            continue
        assert bb.output_strides() == [16, 2], bb.output_strides()
        out_a = model(torch.cat([x, hu_a], 1))
        out_a = out_a[0] if isinstance(out_a, list) else out_a
        assert out_a.shape == (1, 118, 96, 96, 96), out_a.shape
        out_a.float().pow(2).mean().backward()
        g = bb.adapter.proj["s16"].conv.weight.grad if hasattr(bb.adapter.proj["s16"], "conv") else \
            next(bb.adapter.proj["s16"].parameters()).grad
        assert g is not None and g.abs().sum() > 0
        if variant == "guided":
            assert all(p.grad is None and not p.requires_grad for p in bb.upsampler.parameters())
        with torch.no_grad():
            model.eval()
            o1 = model(torch.cat([x, hu_a], 1)); o2 = model(torch.cat([x, hu_b], 1))
        diff = float((o1 - o2).abs().max())
        if variant == "guided":
            assert diff > 1e-4, "guided skip ignores the CT"
        else:
            assert diff == 0.0, "trilinear control must ignore the HU channel"
        print(f"  ok  interface I1 + {variant} skip: out {tuple(out_a.shape)}, "
              f"HU sensitivity {diff:.2e}", flush=True)
    assert counts[None] == counts["trilinear"] == counts["guided"], counts
    print(f"  ok  trainable params identical: {counts[None]:,} (I1) = trilinear = guided",
          flush=True)
    PASSED.append("interface upsampled skip")


# --------------------------------------------------------------------- bench
def bench(device="cuda"):
    """Time and peak memory at a 96^3 patch, stride-2 output, 1024 channels."""
    torch.backends.cudnn.benchmark = True
    m = GuidedUpsampler3D(feature_keys=True).to(device)
    ct = torch.randn(2, 1, 96, 96, 96, device=device) * 400
    for C, grid in ((1024, (6, 6, 6)), (384, (8, 8, 8)), (512, (24, 24, 10))):
        f = torch.randn(2, C, *grid, device=device, dtype=torch.bfloat16)
        for mode in ("train", "infer"):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            ts = []
            for it in range(6):
                t0 = time.time()
                if mode == "train":
                    out = m(ct, f)
                    out.float().pow(2).mean().backward()
                else:
                    with torch.no_grad():
                        out = m(ct, f)
                torch.cuda.synchronize()
                ts.append(time.time() - t0)
                del out
            peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
            print(f"  bench C={C:4d} grid={grid} {mode:5s}: {1000 * sorted(ts)[2]:.0f} ms/batch-of-2, "
                  f"peak +{peak:.2f} GiB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--encoders", nargs="*", default=None,
                    help=f"also build these encoders (random init, CPU); none given = all "
                         f"of {list(ENCODERS)}")
    ap.add_argument("--store", default=None,
                    help="volume store root: check intensity normalization against MONAI")
    ap.add_argument("--interface", default=None, metavar="CKPT",
                    help="check the upsampled skip in the I1 head with this upsampler")
    a = ap.parse_args()
    torch.manual_seed(0)
    for t in (t_gradcheck, t_naive, t_unity, t_channels, t_linear, t_constant_ct, t_grids,
              t_grads, t_orientation):
        t()
    if a.encoders is not None:
        t_encoders(a.encoders or list(ENCODERS))
    if a.store:
        t_intensity(a.store)
    if a.interface:
        t_interface(a.interface)
    print(f"{len(PASSED)} checks passed")
    if a.bench:
        bench()


if __name__ == "__main__":
    main()
