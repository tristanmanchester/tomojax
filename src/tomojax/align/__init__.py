"""Small production facade for alignment.

The package root is intentionally limited to the stable product entrypoints:
``AlignConfig``, ``coupled_pose_config``, ``align``, and ``align_multires``.
Advanced schedule, loss, profile, geometry-state, and solver helpers remain available from
``tomojax.align.api`` while that developer facade is split into deeper owners.
"""

from __future__ import annotations

from ._config import coupled_pose_config
from .pipeline import AlignConfig, align, align_multires

__all__ = [
    "AlignConfig",
    "align",
    "align_multires",
    "coupled_pose_config",
]
