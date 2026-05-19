"""TotalSegmentator dataset.

Each subject is a directory ``s<id>/`` containing:
  - ``ct.nii.gz``                          single-channel CT volume
  - ``segmentations/<organ>.nii.gz``       117 per-organ binary masks
  - ``label.nii.gz`` (optional)            merged uint8 label map, written by
                                           ``scripts/prepare_data.py``

Masks are merged into a single (D, H, W) int label map. The merge order is the
alphabetical class order from ``classes.txt`` (background = 0, organ_i = i+1).

If ``label.nii.gz`` is present in the subject directory, ``__getitem__`` reads
that single file instead of opening and gunzipping all 117 per-organ masks
(~50× faster per sample). Otherwise the per-organ merge runs at load time.
Run ``scripts/prepare_data.py`` once to populate the merged labels.

Splits come from ``meta.csv``'s ``split`` column (train / val / test).
"""
from __future__ import annotations
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


CLASSES_FILE = Path(__file__).with_name("classes.txt")

# Filename of the merged uint8 label produced by scripts/prepare_data.py. Lives
# inside each subject's directory next to ct.nii.gz.
LABEL_FILENAME = "label.nii.gz"


def load_classes(path: Optional[os.PathLike] = None) -> List[str]:
    p = Path(path) if path else CLASSES_FILE
    with p.open() as f:
        names = [line.strip() for line in f if line.strip()]
    if not names:
        raise RuntimeError(f"empty classes file {p}")
    return names


def class_index_map(classes: Sequence[str]) -> Dict[str, int]:
    """organ_name -> label index (1..N). Background is 0."""
    return {name: i + 1 for i, name in enumerate(classes)}


def build_subject_lists(meta_csv: os.PathLike) -> Dict[str, List[str]]:
    """Parse meta.csv and group subject IDs by split."""
    splits = {"train": [], "val": [], "test": []}
    with open(meta_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            sid = row.get("image_id", "").strip()
            split = row.get("split", "").strip().lower()
            if not sid:
                continue
            if split not in splits:
                continue
            splits[split].append(sid)
    return splits


@dataclass
class TSItem:
    image_path: Path
    seg_dir: Path
    subject_id: str
    merged_label_path: Optional[Path] = None  # set if subject_dir/label.nii.gz exists


class TotalSegmentatorDataset(Dataset):
    """Yields dicts ``{"image": (1,D,H,W) float32, "label": (1,D,H,W) int64, "id": str}``.

    The dataset itself does NO preprocessing — pass it through a MONAI
    transform pipeline (see ``unified.data.transforms``) for resampling,
    intensity normalization, and crops.
    """

    def __init__(
        self,
        root: os.PathLike,
        subject_ids: Sequence[str],
        classes: Optional[Sequence[str]] = None,
        skip_missing_masks: bool = True,
        use_merged_label: bool = True,
    ):
        """
        Args:
            use_merged_label: if True (default), use subject_dir/label.nii.gz
                when present (~50× faster than the 117-file merge). Set False
                to force the per-organ merge — useful for benchmarking or to
                bypass a stale merged file.
        """
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.classes = list(classes) if classes else load_classes()
        self.class_to_idx = class_index_map(self.classes)
        self.skip_missing_masks = skip_missing_masks
        self.use_merged_label = use_merged_label

        self.items: List[TSItem] = []
        n_merged = 0
        for sid in subject_ids:
            sdir = self.root / sid
            ct = sdir / "ct.nii.gz"
            seg = sdir / "segmentations"
            if not ct.exists() or not seg.is_dir():
                continue
            merged = sdir / LABEL_FILENAME
            merged_path = merged if (use_merged_label and merged.exists()) else None
            if merged_path is not None:
                n_merged += 1
            self.items.append(TSItem(ct, seg, sid, merged_path))

        if not self.items:
            raise RuntimeError(
                f"no valid subjects under {self.root} from {len(subject_ids)} requested"
            )
        self.num_merged_available = n_merged

    def __len__(self) -> int:
        return len(self.items)

    def _load_label_map(self, seg_dir: Path, ref_shape) -> np.ndarray:
        """Merge 117 binary masks into a single int label map. Returns (D, H, W)."""
        import nibabel as nib  # lazy
        label = np.zeros(ref_shape, dtype=np.int16)
        for name, idx in self.class_to_idx.items():
            f = seg_dir / f"{name}.nii.gz"
            if not f.exists():
                if self.skip_missing_masks:
                    continue
                raise FileNotFoundError(f)
            mask = nib.load(str(f)).get_fdata(dtype=np.float32)
            if mask.shape != ref_shape:
                raise RuntimeError(
                    f"mask {f} shape {mask.shape} != CT shape {ref_shape}"
                )
            # Later organ in the list wins where masks overlap (rare in TS).
            label[mask > 0.5] = idx
        return label

    def __getitem__(self, i: int):
        import nibabel as nib  # lazy
        item = self.items[i]
        nii = nib.load(str(item.image_path))
        image = nii.get_fdata(dtype=np.float32)               # (D, H, W) but NIfTI is (X,Y,Z)

        if item.merged_label_path is not None:
            # Single uint8 NIfTI, same shape/affine as the CT.
            # asanyarray(dataobj) avoids the float64 cast that get_fdata does.
            lab_nii = nib.load(str(item.merged_label_path))
            label = np.asanyarray(lab_nii.dataobj).astype(np.int16, copy=False)
            if label.shape != image.shape:
                raise RuntimeError(
                    f"merged label {item.merged_label_path} shape {label.shape} "
                    f"!= CT {image.shape}"
                )
        else:
            label = self._load_label_map(item.seg_dir, image.shape)

        # MONAI conventions: 4D channel-first tensor (C, D, H, W).
        image_t = torch.from_numpy(image)[None, ...].float()
        label_t = torch.from_numpy(label)[None, ...].long()

        return {
            "image": image_t,
            "label": label_t,
            "image_meta_dict": {"affine": nii.affine.astype(np.float32), "filename_or_obj": str(item.image_path)},
            "label_meta_dict": {"affine": nii.affine.astype(np.float32)},
            "id": item.subject_id,
        }
