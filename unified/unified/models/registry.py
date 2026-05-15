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
    if name not in _BACKBONES:
        raise KeyError(
            f"unknown backbone {name!r}. Registered: {sorted(_BACKBONES)}"
        )
    return _BACKBONES[name](**kwargs)


def list_backbones():
    return sorted(_BACKBONES)
