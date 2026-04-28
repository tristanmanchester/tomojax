"""Public alignment API and bounded legacy compatibility aliases.

Only ``AlignConfig``, ``align``, and ``align_multires`` are the public
alignment extension surface. New code that needs internals should import owner
modules under ``align.model``, ``align.geometry``, ``align.objectives``, or
``align.io`` directly.

``tomojax.align.losses`` remains as a documented compatibility alias for older
callers that imported the loss facade through the alignment package.

The remaining compatibility aliases are deliberately limited to modules with
existing downstream/test imports from the pre-decomposition layout:
``checkpoint``, ``diagnostics``, ``motion_models``, and ``params_export``.
These aliases are import compatibility only; they are not part of ``__all__``
and new callers should use the owner-module paths.
"""

from __future__ import annotations

import sys

from .pipeline import AlignConfig, align, align_multires
from .io import checkpoint as _checkpoint
from .io import params_export as _params_export
from .model import diagnostics as _diagnostics
from .model import motion_models as _motion_models
from .objectives import losses as _losses

_LEGACY_COMPAT_MODULE_ALIASES = {
    "checkpoint": _checkpoint,
    "diagnostics": _diagnostics,
    "motion_models": _motion_models,
    "params_export": _params_export,
}

sys.modules.setdefault(f"{__name__}.losses", _losses)
for _legacy_name, _module in _LEGACY_COMPAT_MODULE_ALIASES.items():
    sys.modules.setdefault(f"{__name__}.{_legacy_name}", _module)

__all__ = [
    "AlignConfig",
    "align",
    "align_multires",
]
