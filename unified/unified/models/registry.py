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
    backbone = _BACKBONES[name](**kwargs)
    if not stem_fusion:
        return backbone
    from .stem_fusion import StemFusionBackbone
    return StemFusionBackbone(backbone, stem_inplanes=stem_inplanes)


def list_backbones():
    return sorted(_BACKBONES)
