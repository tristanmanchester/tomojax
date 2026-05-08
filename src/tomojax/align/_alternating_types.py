"""Alternating smoke configuration and result containers."""
# pyright: reportAny=false, reportUnusedClass=false

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import jax

    from tomojax.align._continuation import ContinuationSchedule
    from tomojax.align._joint_schur_lm import JointSchurDiagnostics
    from tomojax.geometry import GeometryState

GeometryUpdateVolumeSource = Literal["fixed_synthetic_truth", "stopped_reconstruction"]
GeometryUpdateSolver = Literal["joint_schur", "setup_only_lm"]
PreviewInitialization = Literal["average_projection", "backprojection", "constant", "zero"]
PreviewReconstructionMaskSource = Literal["all_views", "train_views"]
PreviewResidualFilterMode = Literal["continuation", "raw"]
PreviewVolumeSupport = Literal["cylindrical", "none", "spherical"]
StoppedPreviewPolicy = Literal[
    "standard",
    "constant_cylindrical_first_level",
    "constant_cylindrical_first_level_no_fista",
]


@dataclass(frozen=True)
class AlternatingSmokeConfig:
    """Configuration for the deterministic 32^3 alternating smoke run."""

    seed: int = 17
    size: int = 32
    n_views: int = 4
    schedule: ContinuationSchedule | None = None
    verification_loss_tolerance: float = 1.0e-5
    gauge_stability_tolerance: float = 1.0e-10
    parameter_update_tolerance: float = 2.0
    heldout_residual_tolerance: float = 1.0e-5
    heldout_view_index: int | None = -1
    geometry_update_volume_source: GeometryUpdateVolumeSource = "stopped_reconstruction"
    geometry_update_solver: GeometryUpdateSolver = "joint_schur"
    geometry_update_setup_prior_strength: float | None = None
    geometry_update_pose_prior_strength: float | None = None
    geometry_update_pose_trust_radius: float | None = None
    geometry_update_pose_frozen: bool = False
    geometry_update_pose_activate_at_level_factor: int | None = None
    geometry_update_alpha_beta_activate_at_level_factor: int | None = None
    geometry_update_theta_activate_at_level_factor: int | None = None
    geometry_update_phi_polish_updates: int = 0
    geometry_update_active_setup_parameters: tuple[str, ...] = ("theta_offset_rad", "det_u_px")
    geometry_update_active_pose_dofs: tuple[str, ...] = (
        "phi_residual_rad",
        "dx_px",
        "dz_px",
    )
    preview_volume_support: PreviewVolumeSupport = "none"
    preview_initialization: PreviewInitialization = "backprojection"
    preview_reconstruction_mask_source: PreviewReconstructionMaskSource = "all_views"
    preview_tv_scale: float = 1.0
    preview_residual_filter_mode: PreviewResidualFilterMode = "continuation"
    preview_center_l2_weight: float = 0.0
    stopped_preview_policy: StoppedPreviewPolicy = "standard"
    fit_gain_offset_nuisance: bool = False
    fit_background_nuisance: bool = False
    synthetic_dataset_name: str | None = None
    synthetic_dataset_artifact_dir: Path | None = None
    synthetic_dataset_nuisance_applied: bool = False
    synthetic_dataset_sidecar_readback: Mapping[str, object] | None = None


@dataclass(frozen=True)
class AlternatingLevelSummary:
    """Per-level alternating smoke summary."""

    level_factor: int
    role: str
    reconstruction_iterations: int
    geometry_updates: int
    executed_geometry_updates: int
    residual_filter_kinds: tuple[str, ...]
    loss_before: float
    loss_after: float
    loss_nonincreasing: bool
    finite_loss: bool
    residual_sigma_estimated: float
    residual_sigma_effective: float
    prior_strength: float
    heldout_residual_before: float | None
    heldout_residual_after: float | None
    heldout_residual_passed: bool | None
    gauge_stable: bool
    parameter_update_norm: float
    parameter_update_small: bool
    verified: bool
    skipped_geometry: bool
    skipped_level: bool
    early_exit_reason: str | None
    schur_diagnostics: JointSchurDiagnostics | None = None


@dataclass(frozen=True)
class _LevelVerificationChecks:
    loss_nonincreasing: bool
    finite_loss: bool
    gauge_stable: bool
    parameter_update_norm: float
    parameter_update_small: bool
    verified: bool


@dataclass(frozen=True)
class AlternatingSmokeResult:
    """Result payload for the deterministic alternating smoke run."""

    final_volume: jax.Array
    initial_geometry: GeometryState
    final_geometry: GeometryState
    levels: tuple[AlternatingLevelSummary, ...]
    verification: Mapping[str, object]
    artifacts: Mapping[str, Path]
