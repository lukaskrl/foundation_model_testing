"""Build the compact 1.5 mm RAS volume store for the upsampler (unified/upsampler/data.py).

Reads the model-independent 1.5 mm training cache (``data.use_source_affine: true``,
fingerprint 3e6024a06c45) where it exists, otherwise runs the same deterministic
prefix (EnsureTyped -> Spacingd -> CropForegroundd) on the raw NIfTI, which is the
case for the test split. Then reorients to RAS and writes int16 HU and uint8 labels.

CPU only. Usage:
    python -m scripts.upsampler_prepare --splits train val test --n-train 300 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

DEFAULT_OUT = "/home/lukas/data/cache/upsampler/vol15"
_CFG = None


def _cfg():
    global _CFG
    if _CFG is None:
        from unified.utils import load_config
        c = load_config(str(REPO / "configs/models/ctfm.yaml"))
        c["data"]["use_source_affine"] = True
        _CFG = c
    return _CFG


def convert(job):
    sid, split, out = job
    import torch
    torch.set_num_threads(1)
    from monai.transforms import Orientationd
    from unified.data.cache import preprocessing_fingerprint
    out = Path(out)
    if (out / f"{sid}.lab.npy").exists() and (out / f"{sid}.meta.json").exists():
        return sid, json.loads((out / f"{sid}.meta.json").read_text()), "cached"
    cfg = _cfg()
    d = cfg["data"]
    cache = Path(d["cache"]["dir"]) / preprocessing_fingerprint(cfg) / f"{sid}.pt"
    if cache.exists():
        data = torch.load(cache, weights_only=False, map_location="cpu")
        src = "train-cache"
    else:
        from unified.data.totalsegmentator import TotalSegmentatorDataset, load_classes
        from unified.data.transforms import build_cache_det_transforms
        ds = TotalSegmentatorDataset(d["dataset_root"], [sid],
                                     classes=load_classes(REPO / d["classes_file"]),
                                     use_source_affine=True)
        data = build_cache_det_transforms(cfg)(ds[0])
        src = "nifti"
    pix = np.linalg.norm(np.asarray(data["image"].affine)[:3, :3], axis=0)
    if not np.allclose(pix, 1.5, atol=1e-3):
        raise RuntimeError(f"{sid}: voxel size {pix}, expected 1.5 mm")
    data = Orientationd(keys=("image", "label"), axcodes="RAS")(
        {"image": data["image"], "label": data["label"]})
    img = data["image"].as_tensor()[0].numpy()
    lab = data["label"].as_tensor()[0].numpy()
    np.save(out / f"{sid}.img.npy", np.clip(np.rint(img), -32768, 32767).astype(np.int16))
    np.save(out / f"{sid}.lab.npy", lab.astype(np.uint8))
    meta = {"shape": list(img.shape), "split": split, "source": src,
            "affine": np.asarray(data["image"].affine).tolist()}
    (out / f"{sid}.meta.json").write_text(json.dumps(meta))
    return sid, meta, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--n-train", type=int, default=300,
                    help="seeded random subset of the train split (label-free training "
                         "and linear-probe fitting draw from it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split in a.splits:
        ids = [l.strip() for l in open(REPO / "unified/data/splits" / f"{split}.txt") if l.strip()]
        if split == "train" and a.n_train < len(ids):
            ids = sorted(np.random.default_rng(a.seed).choice(ids, a.n_train, replace=False).tolist())
        jobs += [(sid, split, str(out)) for sid in ids]
    print(f"{len(jobs)} volumes -> {out}", flush=True)
    t0 = time.time()
    index = json.loads((out / "index.json").read_text()) if (out / "index.json").exists() else {}
    with Pool(a.workers) as pool:
        for i, (sid, meta, src) in enumerate(pool.imap_unordered(convert, jobs), 1):
            index[sid] = meta
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  last={sid} {meta['shape']} ({src})  "
                      f"{time.time() - t0:.0f}s", flush=True)
                (out / "index.json").write_text(json.dumps(index))
    (out / "index.json").write_text(json.dumps(index))
    print("done", flush=True)


if __name__ == "__main__":
    main()
