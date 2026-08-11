"""CT-RATE chest-CT classification data module (18-way multi-label).

CT-RATE (Hamamci et al.) is a gated HuggingFace dataset: ~25k non-contrast
chest CT volumes + radiology reports + 18 binary abnormality labels. The full
raw release is tens of TB (multiple reconstructions per study at native
resolution), so we DO NOT mirror it. Instead we stream one file at a time from
the Hub, preprocess to a small canonical tensor, cache that, and delete the raw
NIfTI — the on-disk footprint is the *preprocessed cache*, not the archive.

Preprocessing follows CT-RATE's own recipe for the parts that are dataset
intrinsic (rescale slope/intercept and voxel spacing come from the metadata
CSV, not the NIfTI header — matching CT-CLIP's ``data_inference_nii.py``), then
lands on the *unified framework's* shared grid so the classification benchmark
uses the same "swap the encoder, keep the input identical" premise as the
segmentation track:

  MODEL-INDEPENDENT (cached):
    raw stored values
      -> HU = slope * x + intercept
      -> resample (CSV spacing) to framework spacing (1.5mm isotropic)
      -> orient to canonical RAS
      -> clip HU to a broad range, cache as int16

  PER-MODEL (on the fly at load):
      -> reorient RAS -> the encoder's pretrain axcodes (e.g. CT-FM SPL)
      -> intensity window (a_min/a_max -> [0,1]; CT-FM broad vs SuPreM -175/250)
      -> resize/crop to the probe input grid

Caching HU (not a normalized window) and orientation-agnostic RAS means one
cache serves every backbone, exactly like the seg pipeline's model-independent
preprocessing prefix (spacing/crop cached; orientation + intensity on the fly).

TOKEN-GATED / VERIFY-ON-ACCESS: the exact Hub file layout (paths below) and the
NIfTI orientation convention are asserted the first time we can authenticate
against the gated repo (see ``verify_access``). Treat the path templates as the
documented CT-RATE convention until confirmed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# --- CT-RATE constants --------------------------------------------------------

CTRATE_REPO_ID = "ibrahimhamamci/CT-RATE"
CTRATE_REPO_TYPE = "dataset"

# The 18 abnormality labels, in the exact order CT-CLIP uses (zero_shot.py:124).
PATHOLOGIES: tuple[str, ...] = (
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
)

# Hub file layout (CT-RATE convention; verify on first authenticated access).
# The split "valid" is CT-RATE's held-out validation set.
_CSV_PATHS = {
    "valid": {
        "labels": "dataset/multi_abnormality_labels/valid_predicted_labels.csv",
        "reports": "dataset/radiology_text_reports/validation_reports.csv",
        "meta": "dataset/metadata/validation_metadata.csv",
    },
    "train": {
        "labels": "dataset/multi_abnormality_labels/train_predicted_labels.csv",
        "reports": "dataset/radiology_text_reports/train_reports.csv",
        "meta": "dataset/metadata/train_metadata.csv",
    },
}


def _volume_hub_path(split: str, volume_name: str) -> str:
    """Map a VolumeName to its NIfTI path in the Hub repo.

    CT-RATE names encode the nesting: ``train_1_a_1.nii.gz`` lives at
    ``dataset/train/train_1/train_1_a/train_1_a_1.nii.gz``. The validation
    split uses the ``valid_*`` prefix under ``dataset/valid/``.
    """
    stem = volume_name[:-7] if volume_name.endswith(".nii.gz") else volume_name
    parts = stem.split("_")
    patient = "_".join(parts[:2])          # e.g. valid_1
    accession = "_".join(parts[:3])        # e.g. valid_1_a
    folder = "valid" if split == "valid" else "train"
    return f"dataset/{folder}/{patient}/{accession}/{stem}.nii.gz"


# --- preprocessing ------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Model-INDEPENDENT preprocessing target for the cache.

    Caches resampled HU as int16 (like the seg framework's cacheable prefix:
    spacing/crop cached, orientation+intensity applied on the fly). The
    per-model intensity WINDOW is NOT baked in here — it is applied at load
    time via ``window_hu`` so one cache serves encoders with different HU
    windows (e.g. CT-FM [-1024, 2048] vs SuPreM [-175, 250]).
    """

    spacing: tuple[float, float, float] = (1.5, 1.5, 1.5)   # mm, isotropic (D,H,W)
    orient_ras: bool = True
    hu_clip_min: int = -1024
    hu_clip_max: int = 3071        # 12-bit CT superset; covers every model window


def _parse_xy_spacing(raw) -> float:
    """CT-RATE stores XYSpacing as a stringified list, e.g. "[0.7, 0.7]"."""
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().lstrip("[").rstrip("]")
    return float(s.split(",")[0])


def preprocess_volume(
    nii_path: str | Path,
    meta_row,
    cfg: PreprocessConfig = PreprocessConfig(),
) -> np.ndarray:
    """Load one CT-RATE NIfTI and return a float16 (D, H, W) array on the
    framework grid, normalized to [b_min, b_max].

    ``meta_row`` is the metadata CSV row for this VolumeName (needs
    RescaleSlope, RescaleIntercept, XYSpacing, ZSpacing).
    """
    import nibabel as nib
    import torch
    import torch.nn.functional as F

    img = nib.load(str(nii_path))
    if cfg.orient_ras:
        img = nib.as_closest_canonical(img)
    # Request float32 directly so nibabel doesn't first allocate a float64 copy
    # (halves peak RAM/worker — matters on the shared, OOM-prone node).
    data = np.asarray(img.get_fdata(dtype=np.float32))   # (X, Y, Z)

    slope = float(meta_row["RescaleSlope"])
    intercept = float(meta_row["RescaleIntercept"])
    data = slope * data + intercept             # -> HU

    xy = _parse_xy_spacing(meta_row["XYSpacing"])
    z = float(meta_row["ZSpacing"])
    current = (xy, xy, z)                        # (X, Y, Z) mm
    target = tuple(cfg.spacing)                  # isotropic

    # Resample by spacing ratio (trilinear), operating on (X, Y, Z).
    t = torch.from_numpy(data)[None, None]       # (1,1,X,Y,Z)
    scale = [current[i] / target[i] for i in range(3)]
    new_shape = [max(1, int(round(data.shape[i] * scale[i]))) for i in range(3)]
    t = F.interpolate(t, size=new_shape, mode="trilinear", align_corners=False)
    data = t[0, 0].numpy()

    # Clip HU to a broad range (superset of every model window) and store as
    # int16 — the per-model window is applied on the fly by window_hu, so one
    # cache is window-agnostic.
    data = np.clip(data, cfg.hu_clip_min, cfg.hu_clip_max)

    # Store in canonical RAS index order (R, A, S) = (X, Y, Z) as int16. The
    # per-model orientation (e.g. CT-FM's SPL) is applied on the fly at load via
    # reorient_from_ras, so one cache serves every encoder's pretrain frame
    # (mirrors the seg pipeline: orientation is NOT baked into the cache).
    return np.round(data).astype(np.int16)


def reorient_from_ras(arr: np.ndarray, axcodes: str = "RAS") -> np.ndarray:
    """Reorient a canonical-RAS array to ``axcodes`` (e.g. "SPL" for CT-FM).

    Uses nibabel's orientation algebra so flips/permutations are exact. The
    result's (axis0, axis1, axis2) follow ``axcodes`` and are used directly as
    (D, H, W). ``axcodes="RAS"`` is a no-op.
    """
    if tuple(axcodes) == ("R", "A", "S"):
        return arr
    from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation
    src = axcodes2ornt(("R", "A", "S"))
    dst = axcodes2ornt(tuple(axcodes))
    return apply_orientation(arr, ornt_transform(src, dst))


def window_hu(arr, a_min=-1024.0, a_max=2048.0, b_min=0.0, b_max=1.0,
              clip=True) -> np.ndarray:
    """Apply an intensity window to a cached HU array -> float32 [b_min, b_max].

    Mirrors MONAI ScaleIntensityRanged; run per-model at load time so a single
    HU cache serves encoders with different windows (CT-FM's broad window vs
    SuPreM's soft-tissue [-175, 250]).
    """
    x = arr.astype(np.float32)
    if clip:
        x = np.clip(x, a_min, a_max)
    x = (x - a_min) / (a_max - a_min)
    return x * (b_max - b_min) + b_min


# --- labels -------------------------------------------------------------------

def load_label_table(labels_csv: str | Path):
    """Return {VolumeName -> np.float32[18]} in PATHOLOGIES order.

    Asserts the CSV's abnormality columns match PATHOLOGIES so a schema drift
    fails loudly rather than silently mislabeling. Uses stdlib csv (the
    framework does not depend on pandas — see totalsegmentator.py).
    """
    import csv

    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        name_col = fields[0]
        missing = [p for p in PATHOLOGIES if p not in fields]
        if missing:
            raise ValueError(
                f"{labels_csv}: label columns missing expected pathologies "
                f"{missing}; got {fields[1:]}"
            )
        out = {}
        for row in reader:
            out[row[name_col]] = np.array(
                [float(row[p]) for p in PATHOLOGIES], dtype=np.float32
            )
    return out


def load_metadata(meta_csv: str | Path) -> dict:
    """Return {VolumeName -> row dict} from a CT-RATE metadata CSV (stdlib csv).

    Rows keep string values; ``preprocess_volume`` casts the fields it needs
    (RescaleSlope/Intercept, XYSpacing, ZSpacing).
    """
    import csv

    with open(meta_csv, newline="") as f:
        reader = csv.DictReader(f)
        return {row["VolumeName"]: row for row in reader}


# --- streaming + cache --------------------------------------------------------

@dataclass
class CacheStats:
    requested: int = 0
    cached: int = 0
    skipped_existing: int = 0
    failed: list[str] = field(default_factory=list)
    bytes_written: int = 0


def _process_one(name, split, cache_dir, raw_dir, meta_by_name, cfg, token,
                 delete_raw):
    """Fetch->preprocess->cache->delete-raw for one volume. Returns
    (status, payload, nbytes) where status in {ok, skip, fail}.

    Downloads into ``raw_dir`` via ``local_dir=`` so the file is a real copy
    there and NOT duplicated as a blob in the global HF cache (which
    ``os.remove`` of a cache symlink would leave behind). Deleting the local
    copy therefore fully frees the space.
    """
    from huggingface_hub import hf_hub_download

    out_path = Path(cache_dir) / (name.replace(".nii.gz", "") + ".npy")
    if out_path.exists():
        return ("skip", name, 0)
    if name not in meta_by_name:
        return ("fail", f"{name} (no metadata row)", 0)
    raw_path = None
    try:
        raw_path = hf_hub_download(
            repo_id=CTRATE_REPO_ID,
            repo_type=CTRATE_REPO_TYPE,
            filename=_volume_hub_path(split, name),
            token=token,
            local_dir=str(raw_dir),
        )
        arr = preprocess_volume(raw_path, meta_by_name[name], cfg)
        tmp = out_path.with_name(out_path.stem + ".tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, out_path)                       # atomic publish
        return ("ok", name, out_path.stat().st_size)
    except Exception as e:  # noqa: BLE001 - record and continue the stream
        return ("fail", f"{name} ({type(e).__name__}: {str(e)[:120]})", 0)
    finally:
        if delete_raw and raw_path and os.path.exists(raw_path):
            try:
                os.remove(raw_path)
            except OSError:
                pass


def stream_and_cache(
    volume_names: Sequence[str],
    split: str,
    cache_dir: str | Path,
    meta_by_name: dict,
    cfg: PreprocessConfig = PreprocessConfig(),
    token: Optional[str] = None,
    delete_raw: bool = True,
    limit: Optional[int] = None,
    workers: int = 8,
    progress_every: int = 50,
    min_free_gb: float = 50.0,
) -> CacheStats:
    """Fetch, preprocess, cache, and delete raw for each volume.

    ``meta_by_name`` maps VolumeName -> metadata row dict (see load_metadata).
    Idempotent: skips volumes whose cache file already exists, so the job is
    resumable (safe to re-run / run in the background). ``workers`` sets the
    download+preprocess thread pool (downloads are network-bound, ~20s each);
    ``limit`` caps the number processed.

    DISK SAFETY: raw volumes (~370 MB each) are downloaded into a temp
    ``local_dir`` and deleted immediately after preprocessing, so transient use
    is bounded by ~``workers`` raw files (~3 GB at workers=8). ``min_free_gb``
    is a hard floor: once free space on the cache filesystem drops below it,
    NO new downloads are launched (in-flight ones drain) and the run stops
    gracefully — it can never fill the disk. Re-run later to resume.
    Returns CacheStats.
    """
    import shutil
    from functools import partial

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = cache_dir / ".raw_tmp"
    raw_dir.mkdir(parents=True, exist_ok=True)

    names = list(volume_names)[:limit] if limit else list(volume_names)
    stats = CacheStats()
    stats.requested = len(names)

    fn = partial(_process_one, split=split, cache_dir=cache_dir, raw_dir=raw_dir,
                 meta_by_name=meta_by_name, cfg=cfg, token=token,
                 delete_raw=delete_raw)

    def _record(res, i):
        status, payload, nbytes = res
        if status == "ok":
            stats.cached += 1
            stats.bytes_written += nbytes
        elif status == "skip":
            stats.skipped_existing += 1
        else:
            stats.failed.append(payload)
        if progress_every and i % progress_every == 0:
            print(f"[ctrate cache] {i}/{len(names)}  cached={stats.cached} "
                  f"skip={stats.skipped_existing} fail={len(stats.failed)} "
                  f"({stats.bytes_written/1e9:.1f} GB)", flush=True)

    def _free_gb() -> float:
        return shutil.disk_usage(cache_dir).free / 1e9

    if _free_gb() < min_free_gb:
        print(f"[ctrate cache] REFUSING to start: free {_free_gb():.0f}GB "
              f"< floor {min_free_gb}GB", flush=True)
        return stats

    aborted = False
    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
        it = iter(names)
        i = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            # Lazily submit so at most ~workers raw files exist at once, and so
            # the free-space floor can halt NEW work immediately.
            inflight = set()
            for _ in range(workers):
                n = next(it, None)
                if n is not None:
                    inflight.add(ex.submit(fn, n))
            while inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    i += 1
                    _record(fut.result(), i)
                if _free_gb() < min_free_gb:
                    if not aborted:
                        print(f"[ctrate cache] ABORT (disk floor): free "
                              f"{_free_gb():.0f}GB < {min_free_gb}GB; draining "
                              f"in-flight, no new downloads", flush=True)
                        aborted = True
                    continue  # drain in-flight, submit nothing new
                for _ in range(len(done)):
                    n = next(it, None)
                    if n is not None:
                        inflight.add(ex.submit(fn, n))
    else:
        for i, n in enumerate(names, 1):
            if _free_gb() < min_free_gb:
                print(f"[ctrate cache] ABORT (disk floor) at {i}", flush=True)
                break
            _record(fn(n), i)

    # Remove the temp download tree (leftover partials + local_dir metadata).
    try:
        shutil.rmtree(raw_dir, ignore_errors=True)
    except OSError:
        pass
    if aborted:
        stats.failed.append("ABORTED_DISK_FLOOR")
    return stats


def verify_access(token: Optional[str] = None) -> dict:
    """Confirm gated access + that the documented CSV paths resolve.

    Call once a valid token is present. Returns the resolved local CSV paths
    for the valid split.
    """
    from huggingface_hub import hf_hub_download

    resolved = {}
    for key, path in _CSV_PATHS["valid"].items():
        resolved[key] = hf_hub_download(
            repo_id=CTRATE_REPO_ID, repo_type=CTRATE_REPO_TYPE,
            filename=path, token=token,
        )
    return resolved


# --- dataset ------------------------------------------------------------------

class CTRateCachedDataset:
    """Torch Dataset over the preprocessed cache.

    Loads a cached float16 (D,H,W) volume, resizes/crops to ``roi`` (fixed grid
    for batching), adds a channel dim, and pairs it with the 18-way label
    vector. ``roi=None`` returns the native cached grid (batch_size=1 only).
    """

    def __init__(self, volume_names, cache_dir, label_map,
                 roi: Optional[tuple[int, int, int]] = (192, 192, 192),
                 intensity: Optional[dict] = None, axcodes: str = "RAS"):
        import torch  # noqa: F401 (kept import local to match module style)

        self.cache_dir = Path(cache_dir)
        self.roi = roi
        # Per-model intensity window applied on the fly (defaults = base.yaml /
        # CT-FM broad window). Pass e.g. {"a_min": -175, "a_max": 250} for SuPreM.
        self.intensity = intensity or {}
        # Per-model pretrain orientation applied on the fly (RAS cache -> axcodes).
        self.axcodes = axcodes
        self.items = []
        for name in volume_names:
            stem = name.replace(".nii.gz", "")
            p = self.cache_dir / (stem + ".npy")
            if p.exists() and name in label_map:
                self.items.append((p, label_map[name]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import torch
        import torch.nn.functional as F

        path, label = self.items[idx]
        arr = np.load(path)                                # int16 HU, RAS
        arr = reorient_from_ras(arr, self.axcodes)         # -> model pretrain frame
        arr = window_hu(np.ascontiguousarray(arr), **self.intensity)  # -> f32 [0,1]
        vol = torch.from_numpy(arr)[None, None]            # (1,1,D,H,W)
        if self.roi is not None:
            vol = F.interpolate(vol, size=self.roi, mode="trilinear",
                                align_corners=False)
        return vol[0], torch.from_numpy(label)             # (1,D,H,W), (18,)
