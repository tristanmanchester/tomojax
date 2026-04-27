"""Public alignment API and legacy loss-module compatibility alias.

Only ``AlignConfig``, ``align``, and ``align_multires`` are the public
alignment extension surface. New code that needs internals should import owner
modules under ``align.model``, ``align.geometry``, ``align.objectives``, or
``align.io`` directly.

``tomojax.align.losses`` remains as a documented compatibility alias for older
callers that imported the loss facade through the alignment package.
"""

from __future__ import annotations

import sys

from .pipeline import AlignConfig, align, align_multires
from .objectives import losses as _losses

sys.modules.setdefault(f"{__name__}.losses", _losses)

__all__ = [
    "AlignConfig",
    "align",
    "align_multires",
]
