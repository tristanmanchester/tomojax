"""Public alignment API.

Only ``AlignConfig``, ``align``, and ``align_multires`` are the public
alignment extension surface. Production code that needs schedule, loss,
profile, or geometry-update helpers should import them from ``tomojax.align.api``
so nested implementation packages can keep moving toward private ownership.
"""

from __future__ import annotations

from .pipeline import AlignConfig, align, align_multires

__all__ = [
    "AlignConfig",
    "align",
    "align_multires",
]
