"""Model registry, base interface, shared head, and per-backbone adapters."""
from . import backbones  # noqa: F401  -- triggers @register_backbone decorators
from .registry import build_backbone, list_backbones
from .seg_model import BackboneInterface, SegModel
from .head import UnifiedSegHead, HEAD_REGISTRY, build_head, register_head

__all__ = [
    "BackboneInterface",
    "SegModel",
    "UnifiedSegHead",
    "HEAD_REGISTRY",
    "build_head",
    "register_head",
    "build_backbone",
    "list_backbones",
]
