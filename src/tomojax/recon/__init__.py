"""Public reconstruction API."""
# pyright: reportAny=false

from __future__ import annotations

from types import ModuleType
from typing import Any

from . import fbp as _fbp_module, fista_tv as _fista_tv_module, spdhg_tv as _spdhg_tv_module
from ._backprojection_accumulation import sum_backproject_views_chunked
from ._fista_diagnostics import (
    ReferenceFISTADiagnosticArtifacts,
    reference_fista_diagnostic_artifacts,
    write_fista_trace_recomputed_csv,
)
from ._fista_reference import (
    ReferenceFISTAConfig,
    ReferenceFISTAQuality,
    ReferenceFISTAResult,
    ReferenceFISTATraceRow,
    fista_reconstruct_reference,
    reference_fista_returned_quality,
    write_fista_trace_csv,
)
from ._gauge_modes import (
    DetUGaugeMode,
    DetUGaugeProjectionReport,
    build_det_u_gauge_mode,
    project_det_u_gauge_component,
)
from ._reference import reconstruct_average_reference, reconstruct_backprojection_reference
from ._schedule_reference import (
    ReferenceFISTASchedule,
    ReferenceFISTAScheduleEntry,
    ReferenceReconstructionScheduleName,
    reference_fista_schedule,
)
from ._scout_support import ScoutSupportResult, build_scout_support
from ._support import VolumeSupportKind, centered_volume_support
from .types import Regulariser


class _CallableReconModule(ModuleType):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        function_name = self.__name__.rsplit(".", 1)[-1]
        return getattr(self, function_name)(*args, **kwargs)


def _make_callable_module(module: ModuleType) -> ModuleType:
    module.__class__ = _CallableReconModule
    return module


fbp = _make_callable_module(_fbp_module)
fista_tv = _make_callable_module(_fista_tv_module)
spdhg_tv = _make_callable_module(_spdhg_tv_module)

FBPConfig = _fbp_module.FBPConfig
default_fbp_scale = _fbp_module.default_fbp_scale
run_parallel_fbp_direct_pallas = _fbp_module.run_parallel_fbp_direct_pallas
supports_parallel_fbp_z_integer = _fbp_module.supports_parallel_fbp_z_integer
FistaConfig = _fista_tv_module.FistaConfig
SPDHGConfig = _spdhg_tv_module.SPDHGConfig

__all__ = [
    "DetUGaugeMode",
    "DetUGaugeProjectionReport",
    "FBPConfig",
    "FistaConfig",
    "ReferenceFISTAConfig",
    "ReferenceFISTADiagnosticArtifacts",
    "ReferenceFISTAQuality",
    "ReferenceFISTAResult",
    "ReferenceFISTASchedule",
    "ReferenceFISTAScheduleEntry",
    "ReferenceFISTATraceRow",
    "ReferenceReconstructionScheduleName",
    "Regulariser",
    "SPDHGConfig",
    "ScoutSupportResult",
    "VolumeSupportKind",
    "build_det_u_gauge_mode",
    "build_scout_support",
    "centered_volume_support",
    "default_fbp_scale",
    "fbp",
    "fista_reconstruct_reference",
    "fista_tv",
    "project_det_u_gauge_component",
    "reconstruct_average_reference",
    "reconstruct_backprojection_reference",
    "reference_fista_diagnostic_artifacts",
    "reference_fista_returned_quality",
    "reference_fista_schedule",
    "run_parallel_fbp_direct_pallas",
    "spdhg_tv",
    "sum_backproject_views_chunked",
    "supports_parallel_fbp_z_integer",
    "write_fista_trace_csv",
    "write_fista_trace_recomputed_csv",
]
