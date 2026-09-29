from typing import Callable, Dict, Type

_BACKBONES: Dict[str, Type] = {}


def register_backbone(name: str) -> Callable[[Type], Type]:
    def deco(cls: Type) -> Type:
        if name in _BACKBONES:
            raise ValueError(f"backbone {name!r} already registered")
        _BACKBONES[name] = cls
        return cls
    return deco


def build_backbone(name: str, **kwargs):
    """Construct a backbone, optionally wrapped for Arm S.

    ``stem_fusion`` / ``stem_inplanes`` are consumed here rather than by any
    individual backbone: Arm S must be one shared code path so every encoder gets
    a bit-identical stem and the added capacity is a constant. Popping them here
    means no backbone module needs to know the arm exists, and Arm N vs Arm S
    differs by exactly this wrapper.
    """
    if name not in _BACKBONES:
        raise KeyError(
            f"unknown backbone {name!r}. Registered: {sorted(_BACKBONES)}"
        )
    stem_fusion = bool(kwargs.pop("stem_fusion", False))
    stem_inplanes = int(kwargs.pop("stem_inplanes", 16))
    # Arms I1/I2 (docs/HEAD_DESIGN.md §7). Consumed here for the same reason as
    # stem_fusion: one shared code path, so no backbone module knows the arm
    # exists and I1 vs I2 differs by exactly one flag.
    interface = kwargs.pop("interface", None)
    interface_width = int(kwargs.pop("interface_width", 256))
    interface_patch_size = tuple(kwargs.pop("interface_patch_size", (96, 96, 96)))
    # Optional upsampled skip on I1/I2 (interface.py): "trilinear" | "guided".
    interface_upsampler = kwargs.pop("interface_upsampler", None)
    interface_upsampler_ckpt = kwargs.pop("interface_upsampler_ckpt", None)
    interface_up_stride = int(kwargs.pop("interface_up_stride", 2))

    backbone = _BACKBONES[name](**kwargs)

    if interface is not None:
        if stem_fusion:
            raise ValueError(
                "stem_fusion (Arm S) and interface (Arms I1/I2) both change where the "
                "fine path comes from; crossing them identifies nothing"
            )
        key = str(interface).lower()
        if key not in ("i1", "i2"):
            raise ValueError(f"interface must be 'i1' or 'i2', got {interface!r}")
        from .interface import InterfaceBackbone
        return InterfaceBackbone(backbone, patch_size=interface_patch_size,
                                 width=interface_width, stem=(key == "i2"),
                                 up_skip=interface_upsampler,
                                 up_ckpt=interface_upsampler_ckpt,
                                 up_stride=interface_up_stride)
    if interface_upsampler is not None:
        raise ValueError("interface_upsampler only applies to the I1/I2 interface")

    if not stem_fusion:
        return backbone
    from .stem_fusion import StemFusionBackbone
    return StemFusionBackbone(backbone, stem_inplanes=stem_inplanes)


def list_backbones():
    return sorted(_BACKBONES)
