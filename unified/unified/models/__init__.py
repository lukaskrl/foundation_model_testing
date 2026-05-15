"""Model registry, base interface, shared head, and per-backbone adapters."""
from . import backbones  # noqa: F401  -- triggers @register_backbone decorators
from .registry import build_backbone, list_backbones
from .seg_model import BackboneInterface, SegModel
from .head import UnifiedSegHead

__all__ = [
    "BackboneInterface",
    "SegModel",
    "UnifiedSegHead",
    "build_backbone",
    "list_backbones",
]
