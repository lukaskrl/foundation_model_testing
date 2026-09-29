"""Phase 0 of the I1/I2 interface: measure what each encoder NATIVELY produces.

Arms N/S/B/W all force every encoder onto the 5-level contract of
``docs/HEAD_DESIGN.md`` §1, so the levels an encoder does not natively have are
manufactured — by a fresh stem, a ``DownsampleNeck``, or resampling. The I1/I2
interface drops the fixed level count instead of filling it, which means we first
need the ground truth this repo has never written down: for every backbone, the
tensors ``encoder_forward`` actually returns, their per-axis stride relative to
the input patch, and their channel widths.

Deliberately reads ONLY ``encoder_forward`` — the pretrained encoder's own output,
before any adapter runs. Where a backbone smuggles non-encoder tensors into that
list (``voco.py`` prepends the raw input volume for its stem), they are detected
and flagged rather than silently counted as a native grid.

Strides are PER AXIS. ``biomedparse`` runs a 2D backbone per axial slice (depth
stride 1, in-plane stride 4+) and ``merlin``'s I3D inflation halves the depth
stride at every level, so a scalar stride cannot describe this suite.

Runs on CPU with random init (``weights=None``): shapes and channel counts are
properties of the architecture, not of the checkpoint.

Usage:
    python -m scripts.probe_native_grids                    # all headline encoders
    python -m scripts.probe_native_grids --models ctfm,dino3d_layerwise
    python -m scripts.probe_native_grids --patch 96,96,96 --threads 16

Outputs: results/native_grids.json  +  docs/NATIVE_GRIDS.md
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]


def _fmt_stride(inp: int, out: int):
    """Exact integer stride where possible, else the float ratio + a flag."""
    if out == 0:
        return {"stride": None, "exact": False}
    if inp % out == 0:
        return {"stride": inp // out, "exact": True}
    return {"stride": round(inp / out, 4), "exact": False}


def probe_one(stem: str, patch, device: str = "cpu") -> dict:
    from unified.models.registry import build_backbone

    mcfg = yaml.safe_load((REPO / "configs" / "models" / f"{stem}.yaml").read_text())["model"]
    name = mcfg["name"]
    kwargs = dict(mcfg.get("kwargs", {}))

    t0 = time.time()
    backbone = build_backbone(name, weights=None, **kwargs)
    backbone.eval().to(device)
    build_s = time.time() - t0

    adapter = getattr(backbone, "adapter", None)
    rec = {
        "config_stem": stem,
        "backbone": name,
        "kwargs": kwargs,
        "patch": list(patch),
        "build_seconds": round(build_s, 1),
        "total_params": sum(p.numel() for p in backbone.parameters()),
        "adapter_params": (sum(p.numel() for p in adapter.parameters())
                           if adapter is not None else None),
        "has_encoder_forward": hasattr(backbone, "encoder_forward"),
        "taps": [],
        "errors": [],
    }

    if not rec["has_encoder_forward"]:
        rec["errors"].append(
            "no encoder_forward/adapter_forward split — cannot be read natively; "
            "SegModel._forward_frozen falls back to freezing the whole backbone"
        )
        del backbone
        return rec

    x = torch.zeros(1, 1, *patch, device=device)
    t0 = time.time()
    with torch.no_grad():
        native = backbone.encoder_forward(x)
    rec["forward_seconds"] = round(time.time() - t0, 1)

    D, H, W = patch
    for i, t in enumerate(native):
        if not torch.is_tensor(t):
            rec["taps"].append({"index": i, "note": f"not a tensor: {type(t).__name__}"})
            continue
        shape = list(t.shape)
        tap = {"index": i, "shape": shape}
        if len(shape) == 5:
            _, c, d, h, w = shape
            sd, sh, sw = (_fmt_stride(D, d), _fmt_stride(H, h), _fmt_stride(W, w))
            tap.update(
                channels=c,
                stride=[sd["stride"], sh["stride"], sw["stride"]],
                stride_exact=bool(sd["exact"] and sh["exact"] and sw["exact"]),
                anisotropic=len({sd["stride"], sh["stride"], sw["stride"]}) > 1,
            )
            # A tap that is bit-identical to the input is the raw volume being
            # passed along for a stem, not an encoder feature.
            if c == 1 and (d, h, w) == (D, H, W) and torch.equal(t, x):
                tap["is_raw_input"] = True
        else:
            tap["note"] = f"non-5D tensor (rank {len(shape)})"
        rec["taps"].append(tap)

    encoder_taps = [t for t in rec["taps"] if not t.get("is_raw_input") and "channels" in t]
    grids = {}
    for t in encoder_taps:
        grids.setdefault(tuple(t["stride"]), []).append(t)
    rec["n_taps_returned"] = len(rec["taps"])
    rec["n_encoder_taps"] = len(encoder_taps)
    rec["n_distinct_grids"] = len(grids)
    def _scale(g):
        """Coarseness of a grid: its largest per-axis stride (None-safe)."""
        vals = [v for v in g if isinstance(v, (int, float)) and v]
        return max(vals) if vals else 0

    rec["distinct_grids"] = [
        {
            "stride": list(g),
            "n_taps": len(grids[g]),
            "channels": sorted({t["channels"] for t in grids[g]}),
            "shareable_projection": len({t["channels"] for t in grids[g]}) == 1,
        }
        for g in sorted(grids, key=lambda s: -_scale(s))
    ]
    if grids:
        coarsest = max(_scale(g) for g in grids)
        finest = min(_scale(g) for g in grids)
        rec["coarsest_stride"] = coarsest
        rec["finest_stride"] = finest
        rec["k_upblocks_to_stride1"] = (
            int(round(math.log2(coarsest)))
            if coarsest and coarsest > 0 and abs(math.log2(coarsest) - round(math.log2(coarsest))) < 1e-6
            else None
        )
        rec["has_native_stride1"] = finest == 1
    rec["any_nonlattice_tap"] = any(
        not t.get("stride_exact", True) for t in encoder_taps
    )
    rec["raw_input_smuggled"] = any(t.get("is_raw_input") for t in rec["taps"])

    del backbone, native
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma-separated config stems (default: the headline set)")
    ap.add_argument("--patch", default=None, help="D,H,W (default: base.yaml data.patch_size)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16,
                    help="torch CPU threads; kept low so running trainings keep their workers")
    ap.add_argument("--json-out", default=str(REPO / "results" / "native_grids.json"))
    ap.add_argument("--md-out", default=str(REPO / "docs" / "NATIVE_GRIDS.md"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    if args.patch:
        patch = tuple(int(v) for v in args.patch.split(","))
    else:
        base = yaml.safe_load((REPO / "configs" / "base.yaml").read_text())
        patch = tuple(base["data"]["patch_size"])

    if args.models:
        stems = [s.strip() for s in args.models.split(",") if s.strip()]
    else:
        from scripts.gen_lowshot_configs import ABLATIONS, BACKBONES
        stems = list(BACKBONES) + list(ABLATIONS)

    out = {
        "what": "native encoder_forward output per backbone — Phase 0 of the I1/I2 interface",
        "patch": list(patch),
        "device": args.device,
        "note": "random init (weights=None); shapes/channels are architecture properties",
        "models": {},
    }
    for stem in stems:
        print(f"[probe] {stem} ...", flush=True)
        try:
            rec = probe_one(stem, patch, args.device)
        except Exception as exc:  # a backbone that cannot be built is itself a finding
            rec = {"config_stem": stem, "errors": [f"{type(exc).__name__}: {exc}"],
                   "traceback": traceback.format_exc()[-2000:]}
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
        else:
            g = rec.get("n_distinct_grids")
            print(f"    taps={rec.get('n_taps_returned')} grids={g} "
                  f"coarsest={rec.get('coarsest_stride')} "
                  f"stride1={rec.get('has_native_stride1')} "
                  f"({rec.get('forward_seconds')}s)", flush=True)
        out["models"][stem] = rec
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2))

    Path(args.md_out).write_text(render_md(out))
    print(f"\nwrote {args.json_out}\nwrote {args.md_out}")


def render_md(out: dict) -> str:
    L = [
        "# Native encoder grids",
        "",
        "AUTO-GENERATED by `python -m scripts.probe_native_grids` — do not edit by hand.",
        "",
        f"Input patch `{tuple(out['patch'])}`, random init, CPU. Every row describes what",
        "`encoder_forward` returns **before any adapter runs** — the pretrained encoder's own",
        "output. This is the input the I1/I2 interface consumes (`docs/HEAD_DESIGN.md` §7).",
        "",
        "`k` is how many times the weight-shared upsample block must be applied to climb from",
        "the coarsest native grid to stride 1.",
        "",
        "| Backbone | taps | distinct grids | strides (D,H,W) | coarsest | native stride 1 | k | aniso | non-lattice | raw input in list |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for stem, r in out["models"].items():
        if r.get("errors") and not r.get("taps"):
            L.append(f"| `{stem}` | — | — | **{r['errors'][0][:70]}** | | | | | | |")
            continue
        grids = r.get("distinct_grids", [])
        strides = " ".join(
            "(" + ",".join(str(v) for v in g["stride"]) + ")" for g in grids
        )
        aniso = any(
            t.get("anisotropic") for t in r.get("taps", []) if "channels" in t
        )
        L.append(
            f"| `{stem}` | {r.get('n_encoder_taps')} | {r.get('n_distinct_grids')} | {strides} | "
            f"{r.get('coarsest_stride')} | {'yes' if r.get('has_native_stride1') else 'no'} | "
            f"{r.get('k_upblocks_to_stride1')} | {'yes' if aniso else 'no'} | "
            f"{'YES' if r.get('any_nonlattice_tap') else 'no'} | "
            f"{'YES' if r.get('raw_input_smuggled') else 'no'} |"
        )
    L += ["", "## Per-tap detail", ""]
    for stem, r in out["models"].items():
        L.append(f"### `{stem}`")
        if r.get("errors"):
            for e in r["errors"]:
                L.append(f"- **error:** {e}")
        if r.get("taps"):
            L.append("")
            L.append("| tap | shape | channels | stride (D,H,W) | note |")
            L.append("|---|---|---|---|---|")
            for t in r["taps"]:
                note = []
                if t.get("is_raw_input"):
                    note.append("**raw input volume, not an encoder feature**")
                if t.get("anisotropic"):
                    note.append("anisotropic")
                if t.get("stride_exact") is False:
                    note.append("does not divide the patch")
                if t.get("note"):
                    note.append(t["note"])
                L.append(
                    f"| {t['index']} | `{t.get('shape')}` | {t.get('channels','—')} | "
                    f"{t.get('stride','—')} | {'; '.join(note)} |"
                )
        gl = r.get("distinct_grids")
        if gl:
            shareable = [g for g in gl if not g["shareable_projection"]]
            if shareable:
                L.append("")
                L.append("- **taps sharing a grid have differing widths** "
                         f"({shareable}); a single shared 1×1 projection is not possible "
                         "for those without a per-width conv.")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
