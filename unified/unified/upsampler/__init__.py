"""Encoder-agnostic 3D feature upsampling for frozen CT foundation models.

See docs/UPSAMPLER_PLAN.md. ``module`` holds the upsampler and the non-learned
baselines, ``geometry`` the grid layouts they share.
"""
from .module import (GuidedUpsampler3D, TrilinearUpsampler, BilateralUpsampler,  # noqa: F401
                     aggregate)
