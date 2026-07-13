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


def _lr_axis(axcodes):
    """Spatial axis index (0-based) that runs left<->right, for an orientation.

    RAS -> 0 (the R axis); SPL (CT-FM) -> 2 (the L axis). Used to know which axis
    a laterality-safe mirror must flip.
    """
    for i, c in enumerate(axcodes):
        if c in ("L", "R"):
            return i
    raise ValueError(f"axcodes {axcodes!r} has no L/R axis")


def _lr_label_remap(num_classes):
    """LUT (length ``num_classes``) mapping each label to its left<->right swap.

    Background (0) and non-lateralized classes map to themselves. Class order is
    the alphabetical ``classes.txt`` order (label index = position + 1), the same
    map the dataset uses to merge masks — so the LUT indexes the label volume
    directly. ``vertebrae_L3`` etc. are untouched ('L3' is not the token 'left').
    Returns ``(remap_list, n_pairs)``.
    """
    from .totalsegmentator import load_classes
    classes = load_classes()
    name_to_idx = {n: i + 1 for i, n in enumerate(classes)}

    def _swap(name):
        parts = ["right" if p == "left" else "left" if p == "right" else p
                 for p in name.split("_")]
        return "_".join(parts)

    remap = list(range(num_classes))
    n_pairs = 0
    for n, i in name_to_idx.items():
        sw = _swap(n)
        if sw != n and sw in name_to_idx:
            remap[i] = name_to_idx[sw]
            if i < name_to_idx[sw]:
                n_pairs += 1
    return remap, n_pairs


def _sigmoid_intensity_transform(center, scale):
    """Nonlinear soft-tissue window: ``out = sigmoid((x - center) / scale)``.

    A smooth alternative to ``ScaleIntensityRanged``'s hard window: it expands
    contrast around ``center`` (soft-tissue WL, ~40 HU) while bone/lung saturate
    SMOOTHLY toward 1/0 instead of hard-clipping, so no intensity information is
    destroyed (monotonic, invertible). Output range (0, 1) matches the linear
    window, so the intensity-augmentation magnitudes still apply unchanged.
    """
    import torch
    from monai.transforms import MapTransform

    class SigmoidIntensityd(MapTransform):
        def __init__(self):
            super().__init__(keys="image")
            self.center = float(center)
            self.scale = float(scale)

        def __call__(self, data):
            d = dict(data)
            for k in self.key_iterator(d):
                d[k] = torch.sigmoid((d[k] - self.center) / self.scale)
            return d

    return SigmoidIntensityd()


def _lateral_flip_transform(lr_axis, remap, prob, label_key="label",
                            keys=("image", "label")):
    """Laterality-safe left<->right mirror (flip + label swap).

    Flips image+label along the L-R spatial axis with probability ``prob``, then
    remaps ``*_left`` <-> ``*_right`` label indices via ``remap`` so organ
    laterality is preserved. A plain ``RandFlipd`` would make left/right organs
    indistinguishable — which is why discrete flips were disabled. Only the L-R
    axis is mirrored (an A-P or S-I flip would create an unrealistic pose).
    """
    import torch
    from monai.transforms import MapTransform, RandomizableTransform
    from monai.data import MetaTensor

    class RandLateralFlipd(MapTransform, RandomizableTransform):
        def __init__(self):
            MapTransform.__init__(self, keys)
            RandomizableTransform.__init__(self, prob)
            self.flip_dim = int(lr_axis) + 1  # channel-first: spatial axis -> +1
            self.remap_lut = torch.as_tensor(remap, dtype=torch.long)

        def __call__(self, data):
            d = dict(data)
            self.randomize(None)
            if not self._do_transform:
                return d
            for k in self.key_iterator(d):
                d[k] = torch.flip(d[k], dims=[self.flip_dim])
            lab = d.get(label_key)
            if lab is not None:
                remapped = self.remap_lut.to(lab.device)[lab.long()]
                if isinstance(lab, MetaTensor):
                    lab.copy_(remapped)          # in-place: preserve MetaTensor
                    d[label_key] = lab
                else:
                    d[label_key] = remapped.to(lab.dtype)
            return d

    return RandLateralFlipd()


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
    t.append(Orientationd(keys=keys, axcodes=axcodes))
    # Intensity normalization. mode "range" (default) = the linear HU window
    # (ScaleIntensityRanged); mode "sigmoid" = the nonlinear soft-tissue window.
    # Both are per-encoder-overridable via model.preprocessing.intensity.
    mode = str(intens.get("mode", "range")).lower()
    if mode == "range":
        t.append(ScaleIntensityRanged(
            keys="image",
            a_min=intens["a_min"], a_max=intens["a_max"],
            b_min=intens["b_min"], b_max=intens["b_max"],
            clip=intens.get("clip", True),
        ))
    elif mode == "sigmoid":
        t.append(_sigmoid_intensity_transform(
            center=intens.get("center", 40.0),
            scale=intens.get("scale", 80.0),
        ))
    else:
        raise ValueError(
            f"unknown data.intensity.mode {mode!r}; expected 'range' or 'sigmoid'")
    return t


def _rand_transform_list(cfg):
    """Stochastic augmentation suffix — must run per-epoch (not cached)."""
    from monai.transforms import (
        SpatialPadd,
        RandCropByLabelClassesd,
        RandAffined,
        RandRotate90d,
        RandShiftIntensityd,
        RandScaleIntensityd,
        RandGaussianSmoothd,
        RandGaussianNoised,
        RandAdjustContrastd,
        RandSimulateLowResolutiond,
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
    # flip_prob now drives a LATERALITY-SAFE left<->right mirror: it flips along
    # the L-R axis (orientation-dependent) AND swaps *_left<->*_right label
    # indices, so left/right organs stay distinguishable. This replaces the old
    # 3-axis RandFlipd, which broke laterality (why flips were disabled). Only
    # L-R is mirrored (A-P / S-I flips make unrealistic poses). Default is 0 in
    # base.yaml (fair-sweep untouched); set >0 per-run to enable the mirror.
    if aug.get("flip_prob", 0.0) > 0:
        axcodes, _ = _resolved_preprocessing(cfg)
        remap, n_pairs = _lr_label_remap(num_classes)
        t.append(_lateral_flip_transform(
            lr_axis=_lr_axis(axcodes), remap=remap, prob=float(aug["flip_prob"])))
    # rot90 stays gated + off: a 180° axial rotation ALSO swaps left/right and is
    # not made safe here, so leave rot_prob=0 unless doing a dedicated ablation.
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
    # Tier-1 recipe upgrade: two nnU-Net quality/appearance augmentations. Both
    # are image-only and laterality-safe (unlike flips), so they lift the shared
    # ceiling without breaking left/right classes. Guarded by prob>0 like the
    # spatial augs above, so prob=0 in config drops them for an ablation.
    #   * simulate-low-resolution: downsample to zoom_range then upsample back,
    #     teaching robustness to resolution/scanner variation (val is
    #     generalization-bound, so this hits the ceiling directly).
    #   * gamma: nonlinear intensity remap; retain_stats keeps the per-patch
    #     mean/std so it perturbs contrast without shifting the distribution.
    if aug.get("lowres_prob", 0.0) > 0:
        t.append(RandSimulateLowResolutiond(
            keys="image", prob=aug["lowres_prob"],
            zoom_range=tuple(aug.get("lowres_zoom_range", (0.5, 1.0))),
        ))
    if aug.get("gamma_prob", 0.0) > 0:
        t.append(RandAdjustContrastd(
            keys="image", prob=aug["gamma_prob"],
            gamma=tuple(aug.get("gamma_range", (0.7, 1.5))),
            retain_stats=True,
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
