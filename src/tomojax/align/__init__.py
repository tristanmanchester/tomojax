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
from .geometry import detector_center as _detector_center
from .geometry import geometry_applier as _geometry_applier
from .geometry import geometry_blocks as _geometry_blocks
from .geometry import initializers as _initializers
from .geometry import parametrizations as _parametrizations
from .io import checkpoint as _checkpoint
from .io import params_export as _params_export
from .model import diagnostics as _diagnostics
from .model import dof_specs as _dof_specs
from .model import dofs as _dofs
from .model import gauge as _gauge
from .model import motion_models as _motion_models
from .model import schedules as _schedules
from .model import state as _state
from .objectives import losses as _losses

_LEGACY_MODULE_ALIASES = {
    "checkpoint": _checkpoint,
    "detector_center": _detector_center,
    "diagnostics": _diagnostics,
    "dof_specs": _dof_specs,
    "dofs": _dofs,
    "gauge": _gauge,
    "geometry_applier": _geometry_applier,
    "geometry_blocks": _geometry_blocks,
    "initializers": _initializers,
    "motion_models": _motion_models,
    "parametrizations": _parametrizations,
    "params_export": _params_export,
    "schedules": _schedules,
    "state": _state,
}

sys.modules.setdefault(f"{__name__}.losses", _losses)
for _legacy_name, _module in _LEGACY_MODULE_ALIASES.items():
    sys.modules.setdefault(f"{__name__}.{_legacy_name}", _module)

__all__ = [
    "AlignConfig",
    "align",
    "align_multires",
]
