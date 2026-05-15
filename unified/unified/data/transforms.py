"""Shared preprocessing pipelines.

Identical for every model. Keep this here, not in a per-model config — the
whole point of the framework is that preprocessing is held constant.
"""
from __future__ import annotations
from typing import Sequence


def build_train_transforms(cfg):
    """cfg is the merged config dict."""
    from monai.transforms import (
        Compose,
        EnsureTyped,
        Orientationd,
        Spacingd,
        ScaleIntensityRanged,
        CropForegroundd,
        SpatialPadd,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandRotate90d,
        RandShiftIntensityd,
        RandScaleIntensityd,
    )

    d = cfg["data"]
    intens = d["intensity"]
    keys = ("image", "label")

    t = [
        EnsureTyped(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=tuple(d["spacing"]),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys="image",
            a_min=intens["a_min"], a_max=intens["a_max"],
            b_min=intens["b_min"], b_max=intens["b_max"],
            clip=intens.get("clip", True),
        ),
        CropForegroundd(keys=keys, source_key="image"),
        SpatialPadd(keys=keys, spatial_size=tuple(d["patch_size"]),
                    mode=("constant", "constant")),
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=tuple(d["patch_size"]),
            num_samples=d["num_samples_per_volume"],
            pos=d["pos_neg_ratio"][0],
            neg=d["pos_neg_ratio"][1],
            image_key="image",
            image_threshold=0.0,
        ),
        RandFlipd(keys=keys, prob=d["augment"]["flip_prob"], spatial_axis=0),
        RandFlipd(keys=keys, prob=d["augment"]["flip_prob"], spatial_axis=1),
        RandFlipd(keys=keys, prob=d["augment"]["flip_prob"], spatial_axis=2),
        RandRotate90d(keys=keys, prob=d["augment"]["rot_prob"], max_k=3),
        RandShiftIntensityd(
            keys="image", offsets=0.10,
            prob=d["augment"]["shift_intensity_prob"],
        ),
        RandScaleIntensityd(
            keys="image", factors=0.10,
            prob=d["augment"]["scale_intensity_prob"],
        ),
    ]
    return Compose(t)


def build_val_transforms(cfg):
    from monai.transforms import (
        Compose,
        EnsureTyped,
        Orientationd,
        Spacingd,
        ScaleIntensityRanged,
        CropForegroundd,
    )

    d = cfg["data"]
    intens = d["intensity"]
    keys = ("image", "label")

    return Compose([
        EnsureTyped(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=tuple(d["spacing"]),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys="image",
            a_min=intens["a_min"], a_max=intens["a_max"],
            b_min=intens["b_min"], b_max=intens["b_max"],
            clip=intens.get("clip", True),
        ),
        CropForegroundd(keys=keys, source_key="image"),
    ])
