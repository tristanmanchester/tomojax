"""Public alignment API and legacy module aliases.

Only ``AlignConfig``, ``align``, and ``align_multires`` are the public alignment
extension surface. The registered submodule aliases below are temporary
compatibility shims for older imports; new production code should import owner
modules under ``align.model``, ``align.geometry``, ``align.objectives``, or
``align.io`` directly when it needs non-public internals.
"""

from __future__ import annotations

import sys
from types import ModuleType

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
from .objectives import fold_recon as _fold_recon
from .objectives import folds as _folds
from .objectives import losses as _losses
from .objectives import recon_layer as _recon_layer
from .objectives import validation_residuals as _validation_residuals


def _register_legacy_module(name: str, module: ModuleType) -> None:
    sys.modules.setdefault(f"{__name__}.{name}", module)
    globals()[name] = module


_LEGACY_MODULES = {
    "checkpoint": _checkpoint,
    "detector_center": _detector_center,
    "diagnostics": _diagnostics,
    "dof_specs": _dof_specs,
    "dofs": _dofs,
    "fold_recon": _fold_recon,
    "folds": _folds,
    "gauge": _gauge,
    "geometry_applier": _geometry_applier,
    "geometry_blocks": _geometry_blocks,
    "initializers": _initializers,
    "losses": _losses,
    "motion_models": _motion_models,
    "parametrizations": _parametrizations,
    "params_export": _params_export,
    "recon_layer": _recon_layer,
    "schedules": _schedules,
    "state": _state,
    "validation_residuals": _validation_residuals,
}

for _name, _module in _LEGACY_MODULES.items():
    _register_legacy_module(_name, _module)

del _name, _module

__all__ = [
    "AlignConfig",
    "align",
    "align_multires",
]
