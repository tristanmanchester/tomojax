"""Public API for reconstruction routines."""

from __future__ import annotations

from tomojax.recon._backprojection_accumulation import sum_backproject_views_chunked
from tomojax.recon._fista_diagnostics import (
    ReferenceFISTADiagnosticArtifacts,
    reference_fista_diagnostic_artifacts,
    write_fista_trace_recomputed_csv,
)
from tomojax.recon._fista_reference import (
    ReferenceFISTAConfig,
    ReferenceFISTAQuality,
    ReferenceFISTAResult,
    ReferenceFISTATraceRow,
    fista_reconstruct_reference,
    reference_fista_returned_quality,
    write_fista_trace_csv,
)
from tomojax.recon._gauge_modes import (
    DetUGaugeMode,
    DetUGaugeProjectionReport,
    build_det_u_gauge_mode,
    project_det_u_gauge_component,
)
from tomojax.recon._reference import (
    reconstruct_average_reference,
    reconstruct_backprojection_reference,
)
from tomojax.recon._schedule_reference import (
    ReferenceFISTASchedule,
    ReferenceFISTAScheduleEntry,
    ReferenceReconstructionScheduleName,
    reference_fista_schedule,
)
from tomojax.recon._scout_support import ScoutSupportResult, build_scout_support
from tomojax.recon._support import VolumeSupportKind, centered_volume_support
from tomojax.recon.fbp import (
    FBPConfig,
    default_fbp_scale,
    fbp,
    run_parallel_fbp_direct_pallas,
    supports_parallel_fbp_z_integer,
)
from tomojax.recon.fista_tv import FistaConfig, fista_tv
from tomojax.recon.spdhg_tv import SPDHGConfig, spdhg_tv
from tomojax.recon.types import Regulariser

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
