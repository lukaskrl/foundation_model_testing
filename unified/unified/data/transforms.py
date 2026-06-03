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


def _reindex_class_indices_transform(axcodes,
                                     label_key="label",
                                     indices_key="label_cls_indices"):
    """Build a transform that reorients the cached per-class index lists.

    ``ClassesToIndicesd`` is baked into the shared on-disk cache, which is built
    in the dataset's native (RAS) orientation — i.e. *before* any per-model
    ``Orientationd``. The resulting ``label_cls_indices`` are flat voxel indices
    into the RAS-layout label volume. For encoders fed in RAS (every model
    except CT-FM) the later ``Orientationd(axcodes="RAS")`` is a no-op, so those
    indices stay valid. CT-FM is fed in SPL: ``Orientationd`` permutes and flips
    the label axes, but the separately-keyed index list is not spatial array
    data, so it is left untouched — ``RandCropByLabelClassesd`` then unravels the
    RAS indices against the *reoriented* spatial shape and picks crop centers in
    the wrong voxels.

    This transform rewrites the cached flat indices into the target orientation
    so the cropper sees centers consistent with the reoriented volume, without
    re-scanning the label (the whole point of caching the indices). It mirrors
    exactly what MONAI's orientation op does to the array — flip on the source
    axes, then permute by ``argsort`` of the orientation transform — so the two
    stay in lockstep. It must run *before* ``Orientationd`` (it reads the source
    affine off the still-RAS label) and is a no-op whenever the source
    orientation already equals ``axcodes``, so it is safe to place
    unconditionally in front of the orientation step.
    """
    import numpy as np
    import torch
    import nibabel as nib
    from monai.data import MetaTensor
    from monai.transforms import MapTransform

    class ReindexClassIndicesd(MapTransform):
        def __init__(self):
            super().__init__(keys=label_key, allow_missing_keys=True)

        def __call__(self, data):
            d = dict(data)
            if indices_key not in d or label_key not in d:
                return d
            label = d[label_key]
            sr = label.ndim - 1  # channel-first (C, *spatial)
            if isinstance(label, MetaTensor):
                affine = label.affine.detach().cpu().numpy().astype(np.float64)
            else:  # no affine -> assume identity (RAS), nothing to reorient
                affine = np.eye(sr + 1, dtype=np.float64)

            src = nib.io_orientation(affine)
            dst = nib.orientations.axcodes2ornt(axcodes[:sr])
            ornt = nib.orientations.ornt_transform(src, dst)
            perm = np.argsort(ornt[:, 0].astype(int))
            flip_axes = [ax for ax in range(sr) if ornt[ax, 1] < 0]

            # Source orientation already matches the target: indices stay valid.
            if not flip_axes and list(perm) == list(range(sr)):
                return d

            shape = tuple(int(s) for s in label.shape[1:])      # source spatial shape
            new_shape = tuple(shape[perm[i]] for i in range(sr))  # post-orientation shape

            remapped = []
            for arr in d[indices_key]:
                if arr is None or len(arr) == 0:
                    remapped.append(arr)
                    continue
                is_tensor = isinstance(arr, torch.Tensor)
                flat = arr.detach().cpu().numpy() if is_tensor else np.asarray(arr)
                coords = list(np.unravel_index(flat.astype(np.int64), shape))
                # Flip on the source axes first (matches torch.flip before permute)...
                for ax in flip_axes:
                    coords[ax] = shape[ax] - 1 - coords[ax]
                # ...then permute: output axis i takes source axis perm[i].
                new_coords = [coords[perm[i]] for i in range(sr)]
                new_flat = np.ravel_multi_index(new_coords, new_shape)
                if is_tensor:
                    remapped.append(torch.as_tensor(new_flat, dtype=arr.dtype))
                else:
                    remapped.append(new_flat.astype(flat.dtype, copy=False))
            d[indices_key] = remapped
            return d

    return ReindexClassIndicesd()


def _orient_intensity_list(cfg, *, reindex_class_indices=False):
    """Model-specific deterministic transforms applied on the fly after the cache.

    These are deterministic but per-model (axcodes / HU window), so excluding
    them from the cache lets one cache on disk serve every model. They run on
    the GPU-bound data path each epoch, but they're cheap: Orientationd is an
    in-place axis permute on the already-resampled volume, and
    ScaleIntensityRanged is a single fused multiply-clamp.

    ``reindex_class_indices`` prepends a transform that reorients the cached
    ``label_cls_indices`` to ``axcodes`` (see
    ``_reindex_class_indices_transform``). Set it on the training path, where
    ``RandCropByLabelClassesd`` consumes those indices; it is a no-op for RAS
    encoders and unnecessary on the validation path (no class-balanced crop).
    """
    from monai.transforms import Orientationd, ScaleIntensityRanged
    axcodes, intens = _resolved_preprocessing(cfg)
    keys = ("image", "label")
    t = []
    if reindex_class_indices:
        # Must precede Orientationd: it reads the source affine off the still-RAS
        # label to compute the same permute/flip Orientationd will apply.
        t.append(_reindex_class_indices_transform(axcodes))
    t += [
        Orientationd(keys=keys, axcodes=axcodes),
        ScaleIntensityRanged(
            keys="image",
            a_min=intens["a_min"], a_max=intens["a_max"],
            b_min=intens["b_min"], b_max=intens["b_max"],
            clip=intens.get("clip", True),
        ),
    ]
    return t


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
    ]
    # Discrete flips / 90° rotations are DISABLED by default (flip_prob =
    # rot_prob = 0) to match the CT-FM TotalSegmentatorV2 recipe, whose only
    # spatial augmentation is the small continuous RandAffine below. Two
    # reasons: (1) a left-right flip destroys organ laterality, so the model
    # can no longer tell *_left from *_right; (2) flip (3 axes @ p) + rot90
    # together leave only a few percent of training patches in the upright
    # orientation that validation/inference actually use. Both depress Dice
    # across every class. Kept behind a prob switch so an ablation can
    # re-enable them from config without touching this file.
    if aug.get("flip_prob", 0.0) > 0:
        t += [
            RandFlipd(keys=keys, prob=aug["flip_prob"], spatial_axis=0),
            RandFlipd(keys=keys, prob=aug["flip_prob"], spatial_axis=1),
            RandFlipd(keys=keys, prob=aug["flip_prob"], spatial_axis=2),
        ]
    if aug.get("rot_prob", 0.0) > 0:
        t.append(RandRotate90d(keys=keys, prob=aug["rot_prob"], max_k=3))
    t.append(
        RandShiftIntensityd(
            keys="image", offsets=aug.get("shift_intensity_factor", 0.10),
            prob=aug["shift_intensity_prob"],
        )
    )
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
    return Compose(_orient_intensity_list(cfg, reindex_class_indices=True)
                   + _rand_transform_list(cfg))


def build_val_post_transforms(cfg):
    """Per-epoch suffix for validation: orientation + intensity only (deterministic)."""
    from monai.transforms import Compose
    return Compose(_orient_intensity_list(cfg))


def build_train_transforms(cfg):
    """Full train pipeline (no caching). cfg is the merged config dict."""
    from monai.transforms import Compose
    return Compose(_det_transform_list(cfg)
                   + _orient_intensity_list(cfg, reindex_class_indices=True)
                   + _rand_transform_list(cfg))


def build_val_transforms(cfg):
    """Full val pipeline (no caching)."""
    from monai.transforms import Compose
    return Compose(_det_transform_list(cfg) + _orient_intensity_list(cfg))
