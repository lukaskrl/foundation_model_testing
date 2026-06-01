"""Shared preprocessing pipelines.

Identical for every model EXCEPT for two narrow per-encoder overrides that are
part of each encoder's pretraining interface, not its hyperparameter recipe:

  * ``model.preprocessing.axcodes``   — orientation the encoder was pretrained on
  * ``model.preprocessing.intensity`` — HU window used at pretrain time

Everything else (spacing, patch size, augmentation, sampler, label space) is
shared across encoders and locked by base.yaml.
"""
from __future__ import annotations
from typing import Sequence


def _resolved_preprocessing(cfg):
    """Merge model-level preprocessing overrides on top of base data.* defaults.

    Returns (axcodes, intensity_dict).
    """
    d = cfg["data"]
    pre = cfg.get("model", {}).get("preprocessing", {}) or {}
    axcodes = pre.get("axcodes", "RAS")
    intensity = {**d["intensity"], **(pre.get("intensity") or {})}
    return axcodes, intensity


def _det_transform_list(cfg):
    """Model-independent preprocessing prefix — cacheable, shared across all models.

    Only the transforms whose output does NOT depend on per-model overrides:
    EnsureTyped → Spacingd → CropForegroundd → ClassesToIndicesd. Orientation
    and intensity windowing are model-specific (axcodes, HU window) and live in
    ``_orient_intensity_list`` so a single cache on disk serves every model.

    ClassesToIndicesd pre-computes the per-class voxel index lists used by
    RandCropByLabelClassesd. Without this, the cropper scans the whole label
    volume on every call (~4 s/sample); with cached indices it's O(1) per
    crop. We cap each class at 10 k indices (`max_samples_per_class=10000`) to
    bound the cache file size — sampling quality is unaffected since we only
    pick one center per crop.
    """
    from monai.transforms import (
        EnsureTyped,
        Spacingd,
        CropForegroundd,
        ClassesToIndicesd,
    )
    d = cfg["data"]
    keys = ("image", "label")
    return [
        EnsureTyped(keys=keys),
        Spacingd(keys=keys, pixdim=tuple(d["spacing"]),
                 mode=("bilinear", "nearest")),
        CropForegroundd(keys=keys, source_key="image",
                        margin=int(d.get("crop_foreground_margin", 0))),
        ClassesToIndicesd(
            keys="label",
            num_classes=int(d["num_classes"]),
            max_samples_per_class=10000,
        ),
    ]


def _orient_intensity_list(cfg):
    """Model-specific deterministic transforms applied on the fly after the cache.

    These are deterministic but per-model (axcodes / HU window), so excluding
    them from the cache lets one cache on disk serve every model. They run on
    the GPU-bound data path each epoch, but they're cheap: Orientationd is an
    in-place axis permute on the already-resampled volume, and
    ScaleIntensityRanged is a single fused multiply-clamp.
    """
    from monai.transforms import Orientationd, ScaleIntensityRanged
    axcodes, intens = _resolved_preprocessing(cfg)
    keys = ("image", "label")
    return [
        Orientationd(keys=keys, axcodes=axcodes),
        ScaleIntensityRanged(
            keys="image",
            a_min=intens["a_min"], a_max=intens["a_max"],
            b_min=intens["b_min"], b_max=intens["b_max"],
            clip=intens.get("clip", True),
        ),
    ]


def _rand_transform_list(cfg):
    """Stochastic augmentation suffix — must run per-epoch (not cached)."""
    from monai.transforms import (
        SpatialPadd,
        RandCropByLabelClassesd,
        RandAffined,
        RandFlipd,
        RandRotate90d,
        RandShiftIntensityd,
        RandScaleIntensityd,
        RandGaussianSmoothd,
        RandGaussianNoised,
    )
    d = cfg["data"]
    keys = ("image", "label")
    num_classes = int(d["num_classes"])

    # Class-balanced sampler: equal weight to every organ class, zero weight on
    # background. This boosts rare-class recall and is the single biggest free
    # win on mean Dice for TotalSegmentator's 117-class label space.
    sampler_cfg = d.get("sampler", {}) or {}
    ratios = sampler_cfg.get("ratios")
    if ratios is None:
        ratios = [0] + [1] * (num_classes - 1)
    if len(ratios) != num_classes:
        raise ValueError(
            f"data.sampler.ratios has length {len(ratios)}, expected {num_classes}"
        )

    aug = d["augment"]
    t = [
        SpatialPadd(keys=keys, spatial_size=tuple(d["patch_size"]),
                    mode=("constant", "constant")),
        RandCropByLabelClassesd(
            keys=keys,
            label_key="label",
            spatial_size=tuple(d["patch_size"]),
            ratios=list(ratios),
            num_classes=num_classes,
            num_samples=d["num_samples_per_volume"],
            # Use precomputed indices baked into the cache by
            # ClassesToIndicesd. The cropper ignores image_key/image_threshold
            # when indices_key is set, so we drop them here.
            indices_key="label_cls_indices",
            warn=False,
        ),
        RandFlipd(keys=keys, prob=aug["flip_prob"], spatial_axis=0),
        RandFlipd(keys=keys, prob=aug["flip_prob"], spatial_axis=1),
        RandFlipd(keys=keys, prob=aug["flip_prob"], spatial_axis=2),
        RandRotate90d(keys=keys, prob=aug["rot_prob"], max_k=3),
        RandShiftIntensityd(
            keys="image", offsets=aug.get("shift_intensity_factor", 0.10),
            prob=aug["shift_intensity_prob"],
        ),
    ]
    # Continuous affine: realistic patient pose variation. CT-FM uses this in
    # place of RandFlip/RandRotate90; we keep both because they're orthogonal.
    if aug.get("affine_prob", 0.0) > 0:
        rot = float(aug.get("affine_rotate_rad", 0.26))
        scale = float(aug.get("affine_scale_factor", 0.20))
        # spatial_size must be set for cache_grid=True; matches the patch size
        # produced by RandCropByLabelClassesd above.
        t.append(RandAffined(
            keys=keys,
            spatial_size=tuple(d["patch_size"]),
            mode=("bilinear", "nearest"),
            prob=aug["affine_prob"],
            rotate_range=(rot, rot, rot),
            scale_range=(scale, scale, scale),
            padding_mode="zeros",
            cache_grid=True,
        ))
    t += [
        RandScaleIntensityd(
            keys="image", factors=aug.get("scale_intensity_factor", 0.10),
            prob=aug["scale_intensity_prob"],
        ),
    ]  # close intensity-augmentation list
    if aug.get("gauss_smooth_prob", 0.0) > 0:
        t.append(RandGaussianSmoothd(
            keys="image", prob=aug["gauss_smooth_prob"],
            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0),
        ))
    if aug.get("gauss_noise_prob", 0.0) > 0:
        t.append(RandGaussianNoised(
            keys="image", prob=aug["gauss_noise_prob"],
            std=aug.get("gauss_noise_std", 0.1),
        ))
    return t


def build_cache_det_transforms(cfg):
    """Cacheable prefix (model-independent): EnsureTyped → Spacingd → CropForegroundd.

    Shared across every model — one disk cache serves the whole sweep.
    """
    from monai.transforms import Compose
    return Compose(_det_transform_list(cfg))


def build_train_post_transforms(cfg):
    """Per-epoch suffix for training: model-specific orientation + intensity
    windowing followed by the stochastic augmentations."""
    from monai.transforms import Compose
    return Compose(_orient_intensity_list(cfg) + _rand_transform_list(cfg))


def build_val_post_transforms(cfg):
    """Per-epoch suffix for validation: orientation + intensity only (deterministic)."""
    from monai.transforms import Compose
    return Compose(_orient_intensity_list(cfg))


def build_train_transforms(cfg):
    """Full train pipeline (no caching). cfg is the merged config dict."""
    from monai.transforms import Compose
    return Compose(_det_transform_list(cfg)
                   + _orient_intensity_list(cfg)
                   + _rand_transform_list(cfg))


def build_val_transforms(cfg):
    """Full val pipeline (no caching)."""
    from monai.transforms import Compose
    return Compose(_det_transform_list(cfg) + _orient_intensity_list(cfg))
