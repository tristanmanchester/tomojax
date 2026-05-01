from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, TypeAlias, cast

from ..core.geometry.base import Geometry
from ..recon.types import Regulariser
from .early_stop import EarlyStopProfile, normalize_early_stop_profile
from .objectives.loss_specs import AlignmentLossConfig, L2OtsuLossSpec
from .model.diagnostics import GaugePolicy
from .model.dofs import (
    DofBounds,
    ScopedAlignmentDofs,
    bounds_vectors,
    normalize_alignment_dofs,
    normalize_bounds,
)
from .model.gauge import GaugeFixMode, normalize_gauge_fix, validate_alignment_gauge_feasible
from .geometry.geometry_blocks import normalize_geometry_dofs
from .model.schedules import AlignmentSchedule, ResolvedAlignmentSchedule, resolve_alignment_schedule

ReconAlgo: TypeAlias = Literal["fista", "spdhg"]
ReconAlgoInput: TypeAlias = Literal[
    "fista",
    "spdhg",
    "fista_tv",
    "spdhg_tv",
    "fista-tv",
    "spdhg-tv",
]
GaugePolicyInput: TypeAlias = GaugePolicy | Literal[
    "anchor-mean",
    "prior-required",
    "diagnose-only",
]
PoseModelInput: TypeAlias = Literal[
    "per_view",
    "per-view",
    "polynomial",
    "spline",
]


def _active_dof_mask_for_cfg(cfg: "AlignConfig") -> tuple[bool, bool, bool, bool, bool]:
    return _scoped_dofs_for_cfg(cfg).pose_mask


def _active_dofs_for_cfg(cfg: "AlignConfig") -> tuple[str, ...]:
    return _scoped_dofs_for_cfg(cfg).active_pose_dofs


def _active_geometry_dofs_for_cfg(
    cfg: "AlignConfig",
    geometry: Geometry | None = None,
) -> tuple[str, ...]:
    return _scoped_dofs_for_cfg(cfg, geometry=geometry).active_geometry_dofs


def _scoped_dofs_for_cfg(
    cfg: "AlignConfig",
    *,
    geometry: Geometry | None = None,
) -> ScopedAlignmentDofs:
    resolved = _resolved_schedule_for_cfg(cfg, geometry=geometry)
    return ScopedAlignmentDofs(
        active_pose_dofs=resolved.active_pose_dofs,
        active_geometry_dofs=resolved.active_geometry_dofs,
        frozen_pose_dofs=tuple(name for name in cfg.freeze_dofs if name in {"alpha", "beta", "phi", "dx", "dz"}),
        frozen_geometry_dofs=tuple(
            name
            for name in cfg.freeze_dofs
            if name
            in {
                "det_u_px",
                "det_v_px",
                "detector_roll_deg",
                "axis_rot_x_deg",
                "axis_rot_y_deg",
                "tilt_deg",
            }
        ),
    )


def _resolved_schedule_for_cfg(
    cfg: "AlignConfig",
    *,
    geometry: Geometry | None = None,
) -> ResolvedAlignmentSchedule:
    return resolve_alignment_schedule(
        schedule=cfg.schedule,
        optimise_dofs=cfg.optimise_dofs,
        freeze_dofs=cfg.freeze_dofs,
        geometry_dofs=cfg.geometry_dofs,
        geometry=geometry,
        gauge_policy=cfg.gauge_policy,
        gauge_priors=cfg.gauge_priors,
        opt_method=cfg.opt_method,
        outer_iters=int(cfg.outer_iters),
        early_stop=bool(cfg.early_stop),
    )


@dataclass
class AlignConfig:
    outer_iters: int = 5
    recon_iters: int = 10
    lambda_tv: float = 0.005
    regulariser: Regulariser = "tv"
    huber_delta: float = 1e-2
    tv_prox_iters: int = 10
    recon_algo: ReconAlgoInput = "fista"
    recon_positivity: bool = True
    spdhg_seed: int = 0
    # Reconstruction stopping criteria
    recon_rel_tol: float | None = None
    recon_patience: int = 2
    # Alignment step sizes
    lr_rot: float = 1e-3  # radians
    lr_trans: float = 1e-1  # world units
    # Memory/throughput knobs
    views_per_batch: int = 1  # stream one view at a time
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"
    # Solver and regularization
    opt_method: str = "gn"
    gn_damping: float = 1e-6
    lbfgs_maxiter: int = 20
    lbfgs_ftol: float = 1e-6
    lbfgs_gtol: float = 1e-5
    lbfgs_maxls: int = 20
    lbfgs_memory_size: int = 10
    w_rot: float = 0.0
    w_trans: float = 0.0
    schedule: str | AlignmentSchedule | None = None
    optimise_dofs: tuple[str, ...] | None = None
    freeze_dofs: tuple[str, ...] = field(default_factory=tuple)
    geometry_dofs: tuple[str, ...] = field(default_factory=tuple)
    bounds: DofBounds | str | Mapping[str, object] = field(default_factory=tuple)
    gauge_policy: GaugePolicyInput = "reject"
    gauge_priors: Mapping[str, object] | None = None
    pose_model: PoseModelInput = "per_view"
    knot_spacing: int = 8
    degree: int = 3
    gauge_fix: GaugeFixMode = "mean_translation"
    seed_translations: bool = False
    # Volume masking before forward projection (modeling for ROI/truncation)
    # Options: "off" (default), "cyl" (cylindrical mask in x–y broadcast along z)
    mask_vol: str = "off"
    # Logging
    log_summary: bool = False
    log_compact: bool = True  # print one compact line per outer when log_summary is enabled
    # Reconstruction Lipschitz (optional override to skip power-method)
    recon_L: float | None = None
    # Early stopping across outers (alignment phase)
    early_stop: bool = True
    early_stop_profile: EarlyStopProfile = "compute_saving"
    early_stop_rel_impr: float = 1e-3  # stop if (before-after)/before < this
    early_stop_patience: int = 2
    # Accept GN steps only when they improve the loss, up to gn_accept_tol.
    gn_accept_only_improving: bool = True
    gn_accept_tol: float = 0.0  # allow tiny increases if >0 (as fraction of before)
    # Data term / similarity
    loss: AlignmentLossConfig = field(default_factory=L2OtsuLossSpec)

    def __post_init__(self) -> None:
        recon_algo = str(self.recon_algo).strip().lower().replace("-", "_")
        if recon_algo in {"fista_tv"}:
            recon_algo = "fista"
        elif recon_algo in {"spdhg_tv"}:
            recon_algo = "spdhg"
        self.recon_algo = cast(ReconAlgoInput, recon_algo)
        if self.recon_algo not in {"fista", "spdhg"}:
            raise ValueError("recon_algo must be one of 'fista' or 'spdhg'")
        opt_method = str(self.opt_method).strip().lower().replace("-", "_")
        if opt_method in {"lbfgsb", "l_bfgs", "l_bfgs_b"}:
            opt_method = "lbfgs"
        self.opt_method = opt_method
        if self.opt_method not in {"gd", "gn", "lbfgs"}:
            raise ValueError("opt_method must be one of 'gd', 'gn', or 'lbfgs'")
        if int(self.lbfgs_maxiter) < 1:
            raise ValueError("lbfgs_maxiter must be >= 1")
        if int(self.lbfgs_maxls) < 1:
            raise ValueError("lbfgs_maxls must be >= 1")
        if int(self.lbfgs_memory_size) < 1:
            raise ValueError("lbfgs_memory_size must be >= 1")
        if float(self.lbfgs_ftol) < 0.0:
            raise ValueError("lbfgs_ftol must be >= 0")
        if float(self.lbfgs_gtol) < 0.0:
            raise ValueError("lbfgs_gtol must be >= 0")
        if self.schedule is not None and self.optimise_dofs is not None:
            raise ValueError("schedule and optimise_dofs are mutually exclusive")
        if isinstance(self.schedule, str):
            self.schedule = self.schedule.strip().lower().replace("-", "_")
            if not self.schedule:
                self.schedule = None
        if self.optimise_dofs is not None:
            self.optimise_dofs = normalize_alignment_dofs(
                self.optimise_dofs,
                option_name="optimise_dofs",
            )
        self.freeze_dofs = normalize_alignment_dofs(self.freeze_dofs, option_name="freeze_dofs")
        self.geometry_dofs = normalize_geometry_dofs(
            self.geometry_dofs,
            geometry=None,
        )
        self.gauge_policy = cast(
            GaugePolicyInput,
            str(self.gauge_policy).strip().lower().replace("-", "_"),
        )
        if self.gauge_policy not in {"reject", "anchor_mean", "prior_required", "diagnose_only"}:
            raise ValueError(
                "gauge_policy must be one of 'reject', 'anchor_mean', "
                "'prior_required', or 'diagnose_only'"
            )
        _active_dof_mask_for_cfg(self)
        self.bounds = normalize_bounds(self.bounds, option_name="bounds")
        pose_model = str(self.pose_model).strip().lower().replace("-", "_")
        self.pose_model = cast(PoseModelInput, pose_model)
        if self.pose_model not in {"per_view", "polynomial", "spline"}:
            raise ValueError("pose_model must be one of 'per_view', 'polynomial', or 'spline'")
        if self.pose_model == "polynomial" and int(self.degree) < 0:
            raise ValueError("degree must be >= 0 for polynomial pose_model")
        if self.pose_model == "spline":
            if int(self.knot_spacing) < 1:
                raise ValueError("knot_spacing must be >= 1 for spline pose_model")
            if int(self.degree) not in (1, 2, 3):
                raise ValueError("degree must be one of 1, 2, or 3 for spline pose_model")
        self.gauge_fix = normalize_gauge_fix(self.gauge_fix)
        if self.gauge_fix == "mean_translation":
            bounds_lower, bounds_upper = bounds_vectors(self.bounds)
            active_mask_for_gauge = _active_dof_mask_for_cfg(self)
            validate_alignment_gauge_feasible(
                mode=self.gauge_fix,
                active_mask=active_mask_for_gauge,
                bounds_lower=bounds_lower,
                bounds_upper=bounds_upper,
            )
        self.early_stop_profile = normalize_early_stop_profile(self.early_stop_profile)
        if float(self.early_stop_rel_impr) < 0.0:
            raise ValueError("early_stop_rel_impr must be >= 0")
        if int(self.early_stop_patience) < 1:
            raise ValueError("early_stop_patience must be >= 1")




__all__ = [
    "AlignConfig",
    "_active_dof_mask_for_cfg",
    "_active_dofs_for_cfg",
    "_active_geometry_dofs_for_cfg",
    "_resolved_schedule_for_cfg",
    "_scoped_dofs_for_cfg",
]
