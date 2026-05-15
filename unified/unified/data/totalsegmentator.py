"""TotalSegmentator dataset.

Each subject is a directory ``s<id>/`` containing:
  - ``ct.nii.gz``                          single-channel CT volume
  - ``segmentations/<organ>.nii.gz``       117 per-organ binary masks

The dataset is READ-ONLY — no caching/conversion is performed on disk. Masks are
merged into a single (D, H, W) int label map at load time, using the alphabetical
class order from ``classes.txt`` (background = 0, organ_i = i+1).

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
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.classes = list(classes) if classes else load_classes()
        self.class_to_idx = class_index_map(self.classes)
        self.skip_missing_masks = skip_missing_masks

        self.items: List[TSItem] = []
        for sid in subject_ids:
            sdir = self.root / sid
            ct = sdir / "ct.nii.gz"
            seg = sdir / "segmentations"
            if not ct.exists() or not seg.is_dir():
                continue
            self.items.append(TSItem(ct, seg, sid))

        if not self.items:
            raise RuntimeError(
                f"no valid subjects under {self.root} from {len(subject_ids)} requested"
            )

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
