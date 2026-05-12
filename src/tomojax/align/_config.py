from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from tomojax.core.backend_policy import normalize_projector_backend

from ._profiles import (
    AlignmentProfileInput,
    FallbackPolicy,
    QualityTier,
    alignment_profile_policy,
    normalize_alignment_profile,
)
from .geometry.geometry_blocks import normalize_geometry_dofs
from .model.diagnostics import GaugePolicy
from .model.dofs import (
    ScopedAlignmentDofs,
    bounds_vectors,
    normalize_alignment_dofs,
    normalize_bounds,
)
from .model.gauge import GaugeFixMode, normalize_gauge_fix, validate_alignment_gauge_feasible
from .model.schedules import (
    AlignmentSchedule,
    ResolvedAlignmentSchedule,
    resolve_alignment_schedule,
)
from .objectives.loss_specs import L2OtsuLossSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax.core.backend_policy import ProjectorBackendInput
    from tomojax.core.geometry.base import Geometry
    from tomojax.recon.types import Regulariser

    from .model.dofs import DofBounds
    from .objectives.loss_specs import AlignmentLossConfig

type ReconAlgo = Literal["fista", "spdhg"]
type ReconAlgoInput = Literal[
    "fista",
    "spdhg",
    "fista_tv",
    "spdhg_tv",
    "fista-tv",
    "spdhg-tv",
]
type GaugePolicyInput = (
    GaugePolicy
    | Literal[
        "anchor-mean",
        "prior-required",
        "diagnose-only",
    ]
)
type PoseModelInput = Literal[
    "per_view",
    "per-view",
    "polynomial",
    "spline",
]


def _active_dof_mask_for_cfg(cfg: AlignConfig) -> tuple[bool, bool, bool, bool, bool]:
    return _scoped_dofs_for_cfg(cfg).pose_mask


def _active_dofs_for_cfg(cfg: AlignConfig) -> tuple[str, ...]:
    return _scoped_dofs_for_cfg(cfg).active_pose_dofs


def _active_geometry_dofs_for_cfg(
    cfg: AlignConfig,
    geometry: Geometry | None = None,
) -> tuple[str, ...]:
    return _scoped_dofs_for_cfg(cfg, geometry=geometry).active_geometry_dofs


def _scoped_dofs_for_cfg(
    cfg: AlignConfig,
    *,
    geometry: Geometry | None = None,
) -> ScopedAlignmentDofs:
    resolved = _resolved_schedule_for_cfg(cfg, geometry=geometry)
    return ScopedAlignmentDofs(
        active_pose_dofs=resolved.active_pose_dofs,
        active_geometry_dofs=resolved.active_geometry_dofs,
        frozen_pose_dofs=tuple(
            name for name in cfg.freeze_dofs if name in {"alpha", "beta", "phi", "dx", "dz"}
        ),
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
    cfg: AlignConfig,
    *,
    geometry: Geometry | None = None,
) -> ResolvedAlignmentSchedule:
    return resolve_alignment_schedule(
        schedule=cfg.schedule,
        optimise_dofs=cfg.optimise_dofs,
        freeze_dofs=cfg.freeze_dofs,
        geometry_dofs=cfg.geometry_dofs,
        geometry=geometry,
        gauge_policy=cast("GaugePolicy", cfg.gauge_policy),
        gauge_priors=cfg.gauge_priors,
        opt_method=cfg.opt_method,
        outer_iters=int(cfg.outer_iters),
        early_stop=bool(cfg.early_stop),
    )


@dataclass
class AlignConfig:
    align_profile: AlignmentProfileInput = "lightning"
    outer_iters: int = 5
    recon_iters: int = 10
    lambda_tv: float = 0.005
    regulariser: Regulariser = "huber_tv"
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
    views_per_batch: int = 0  # 0 means use the whole view stack when memory allows
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "auto"
    projector_backend: ProjectorBackendInput = "pallas"
    quality_tier: QualityTier = "fast"
    fallback_policy: FallbackPolicy = "fallback"
    fold_rigid_detector_grid: bool = True
    # Solver and regularization
    opt_method: str = "gn"
    gn_damping: float = 1e-3
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
    pose_model: PoseModelInput = "spline"
    knot_spacing: int = 8
    degree: int = 3
    gauge_fix: GaugeFixMode = "mean_translation"
    seed_translations: bool = False
    # Volume masking before forward projection (modeling for ROI/truncation)
    # Options: "off" (default), "cyl" (cylindrical mask in x-y broadcast along z)
    mask_vol: str = "off"
    # Logging
    log_summary: bool = False
    log_compact: bool = True  # print one compact line per outer when log_summary is enabled
    # Reconstruction Lipschitz (optional override to skip power-method)
    recon_L: float | None = None
    # Early stopping across outers (alignment phase)
    early_stop: bool = True
    early_stop_rel_impr: float = 1e-3  # stop if (before-after)/before < this
    early_stop_patience: int = 2
    # Accept GN steps only when they improve the loss, up to gn_accept_tol.
    gn_accept_only_improving: bool = True
    gn_accept_tol: float = 0.0  # allow tiny increases if >0 (as fraction of before)
    # Data term / similarity
    loss: AlignmentLossConfig = field(default_factory=L2OtsuLossSpec)

    def __post_init__(self) -> None:
        self._apply_profile_policy()
        self._normalize_reconstruction_options()
        self._normalize_backend_options()
        self._normalize_optimizer_options()
        self._normalize_schedule_options()
        self._normalize_dof_options()
        self._normalize_gauge_options()
        self._normalize_pose_model_options()
        self._normalize_gauge_fix_options()

    def _apply_profile_policy(self) -> None:
        self.align_profile = normalize_alignment_profile(self.align_profile)
        profile_policy = alignment_profile_policy(self.align_profile)
        if self.align_profile == "tortoise":
            self.projector_backend = profile_policy.projector_backend
            self.gather_dtype = profile_policy.gather_dtype
            self.regulariser = profile_policy.regulariser
            self.recon_algo = profile_policy.recon_algo  # type: ignore[assignment]
            self.views_per_batch = int(profile_policy.views_per_batch)
            self.checkpoint_projector = bool(profile_policy.checkpoint_projector)
            self.pose_model = profile_policy.pose_model  # type: ignore[assignment]
            self.quality_tier = profile_policy.quality_tier
            self.fallback_policy = profile_policy.fallback_policy
        else:
            self.quality_tier = profile_policy.quality_tier
            self.fallback_policy = profile_policy.fallback_policy

    def _normalize_reconstruction_options(self) -> None:
        recon_algo = str(self.recon_algo).strip().lower().replace("-", "_")
        if recon_algo in {"fista_tv"}:
            recon_algo = "fista"
        elif recon_algo in {"spdhg_tv"}:
            recon_algo = "spdhg"
        self.recon_algo = cast("ReconAlgoInput", recon_algo)
        if self.recon_algo not in {"fista", "spdhg"}:
            raise ValueError("recon_algo must be one of 'fista' or 'spdhg'")

    def _normalize_backend_options(self) -> None:
        self.projector_backend = normalize_projector_backend(self.projector_backend)
        self.gather_dtype = str(self.gather_dtype).strip().lower()
        self.quality_tier = cast("QualityTier", str(self.quality_tier).strip().lower())
        if self.quality_tier not in {"fast", "reference"}:
            raise ValueError("quality_tier must be one of 'fast' or 'reference'")
        self.fallback_policy = cast("FallbackPolicy", str(self.fallback_policy).strip().lower())
        if self.fallback_policy not in {"fallback", "strict"}:
            raise ValueError("fallback_policy must be one of 'fallback' or 'strict'")

    def _normalize_optimizer_options(self) -> None:
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

    def _normalize_schedule_options(self) -> None:
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

    def _normalize_dof_options(self) -> None:
        self.freeze_dofs = normalize_alignment_dofs(self.freeze_dofs, option_name="freeze_dofs")
        self.geometry_dofs = normalize_geometry_dofs(
            self.geometry_dofs,
            geometry=None,
        )

    def _normalize_gauge_options(self) -> None:
        self.gauge_policy = cast(
            "GaugePolicyInput",
            str(self.gauge_policy).strip().lower().replace("-", "_"),
        )
        if self.gauge_policy not in {"reject", "anchor_mean", "prior_required", "diagnose_only"}:
            raise ValueError(
                "gauge_policy must be one of 'reject', 'anchor_mean', "
                "'prior_required', or 'diagnose_only'"
            )
        _ = _active_dof_mask_for_cfg(self)
        self.bounds = normalize_bounds(self.bounds, option_name="bounds")

    def _normalize_pose_model_options(self) -> None:
        pose_model = str(self.pose_model).strip().lower().replace("-", "_")
        self.pose_model = cast("PoseModelInput", pose_model)
        if self.pose_model not in {"per_view", "polynomial", "spline"}:
            raise ValueError("pose_model must be one of 'per_view', 'polynomial', or 'spline'")
        if self.pose_model == "polynomial" and int(self.degree) < 0:
            raise ValueError("degree must be >= 0 for polynomial pose_model")
        if self.pose_model == "spline":
            if int(self.knot_spacing) < 1:
                raise ValueError("knot_spacing must be >= 1 for spline pose_model")
            if int(self.degree) not in (1, 2, 3):
                raise ValueError("degree must be one of 1, 2, or 3 for spline pose_model")

    def _normalize_gauge_fix_options(self) -> None:
        self.gauge_fix = normalize_gauge_fix(self.gauge_fix)
        if self.gauge_fix == "mean_translation":
            bounds = cast("DofBounds", self.bounds)
            bounds_lower, bounds_upper = bounds_vectors(bounds)
            active_mask_for_gauge = _active_dof_mask_for_cfg(self)
            validate_alignment_gauge_feasible(
                mode=self.gauge_fix,
                active_mask=active_mask_for_gauge,
                bounds_lower=bounds_lower,
                bounds_upper=bounds_upper,
            )


__all__ = [
    "AlignConfig",
    "_active_dof_mask_for_cfg",
    "_active_dofs_for_cfg",
    "_active_geometry_dofs_for_cfg",
    "_resolved_schedule_for_cfg",
    "_scoped_dofs_for_cfg",
]
