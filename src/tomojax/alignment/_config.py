from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import TYPE_CHECKING, Literal, cast

from tomojax.core.backend_policy import normalize_projector_backend
from tomojax.core.projector import RAY_INTEGRATORS

from ._geometry.parametrizations import PoseTranslationFrame
from ._model.diagnostics import GaugePolicy
from ._model.dofs import (
    DOF_NAMES,
    ScopedAlignmentDofs,
    normalize_alignment_dofs,
    normalize_bounds,
)
from ._model.schedules import (
    AlignmentSchedule,
    ResolvedAlignmentSchedule,
    StageQualityTier,
    resolve_alignment_schedule,
)
from ._objectives.loss_specs import L2LossSpec, L2OtsuLossSpec
from ._profiles import QualityTier, alignment_profile_policy, normalize_quality

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax.core.backend_policy import ProjectorBackendInput
    from tomojax.recon.types import Regulariser

    from ._model.dofs import DofBounds
    from ._objectives.loss_specs import AlignmentLossConfig

type Reconstruction = Literal["fista", "spdhg"]
type PoseModel = Literal["per_view", "polynomial", "spline"]


def _active_dof_mask_for_cfg(cfg: AlignConfig) -> tuple[bool, ...]:
    return _scoped_dofs_for_cfg(cfg).pose_mask


def _active_dofs_for_cfg(cfg: AlignConfig) -> tuple[str, ...]:
    return _scoped_dofs_for_cfg(cfg).active_pose_dofs


def _active_geometry_dofs_for_cfg(
    cfg: AlignConfig,
) -> tuple[str, ...]:
    return _scoped_dofs_for_cfg(cfg).active_geometry_dofs


def _scoped_dofs_for_cfg(cfg: AlignConfig) -> ScopedAlignmentDofs:
    resolved = _resolved_schedule_for_cfg(cfg)
    return ScopedAlignmentDofs(
        active_pose_dofs=resolved.active_pose_dofs,
        active_geometry_dofs=resolved.active_geometry_dofs,
        frozen_pose_dofs=tuple(name for name in cfg.freeze if name in set(DOF_NAMES)),
        frozen_geometry_dofs=tuple(
            name
            for name in cfg.freeze
            if name
            in {
                "det_u_px",
                "det_v_px",
                "detector_roll_deg",
                "axis_rot_x_deg",
                "axis_rot_y_deg",
            }
        ),
    )


def resolved_schedule_for_config(cfg: AlignConfig) -> ResolvedAlignmentSchedule:
    """Return the schedule ``cfg`` runs, including coupled pose-stage objectives."""
    return _resolved_schedule_for_cfg(cfg)


def _resolved_schedule_for_cfg(cfg: AlignConfig) -> ResolvedAlignmentSchedule:
    resolved = resolve_alignment_schedule(
        schedule=cfg.schedule,
        optimise_dofs=cfg.optimise_dofs,
        freeze=cfg.freeze,
        gauge_policy=cfg.gauge_policy,
        gauge_priors=cfg.gauge_priors,
        opt_method=cfg.opt_method,
        outer_iterations=int(cfg.outer_iterations),
        early_stop=bool(cfg.early_stop),
    )
    if cfg.gn_coupling == "joint":
        # Pose-only stages solve volume and poses together; setup stages keep theirs.
        resolved = replace(
            resolved,
            stages=tuple(
                replace(stage, objective_kind="joint_volume_pose")
                if stage.active_pose_dofs and not stage.active_geometry_dofs
                else stage
                for stage in resolved.stages
            ),
        )
    return resolved


@dataclass(kw_only=True)
class AlignConfig:
    quality: QualityTier = "fast"
    outer_iterations: int = 5
    iterations: int = 10
    tv_weight: float = 0.005
    regulariser: Regulariser = "huber_tv"
    huber_delta: float = 1e-2
    tv_prox_iterations: int = 10
    reconstruction: Reconstruction = "fista"
    nonnegative: bool = True
    seed: int = 0
    # Reconstruction stopping criteria
    recon_rel_tol: float | None = None
    recon_patience: int = 2
    # Alignment step sizes
    lr_rot: float = 1e-3  # radians
    lr_trans: float = 1e-1  # world units
    # Memory/throughput knobs. views_per_batch=0 sizes reconstruction batches
    # from free device memory when alignment starts.
    views_per_batch: int = 0
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "auto"
    projector_backend: ProjectorBackendInput = "pallas"
    ray_integrator: Literal["sampled", "exact", "joseph", "joseph_cubic"] = field(
        default="sampled", kw_only=True
    )
    # Solver and regularization
    opt_method: str = "gn"
    gn_damping: float = 1e-3
    gn_jacobian: Literal["autodiff", "central"] = field(default="autodiff", kw_only=True)
    # Central-difference displacement as a fraction of the smallest voxel pitch.
    gn_difference_step: float = field(default=1e-3, kw_only=True)
    gn_coupling: Literal["fixed_volume", "joint"] = field(default="fixed_volume", kw_only=True)
    gn_joint_solver: Literal["stacked", "pose_eliminated"] = field(default="stacked", kw_only=True)
    gn_joint_iterations: int = field(default=40, kw_only=True)
    gn_joint_rtol: float = field(default=1e-4, kw_only=True)
    gn_volume_damping: float = field(default=1e-3, kw_only=True)
    lbfgs_maxiter: int = 20
    lbfgs_ftol: float = 1e-6
    lbfgs_gtol: float = 1e-5
    lbfgs_maxls: int = 20
    lbfgs_memory_size: int = 10
    w_rot: float = 0.0
    w_trans: float = 0.0
    schedule: str | AlignmentSchedule | None = None
    optimise_dofs: tuple[str, ...] | None = None
    freeze: tuple[str, ...] = field(default_factory=tuple)
    bounds: DofBounds | str | Mapping[str, object] = field(default_factory=tuple)
    gauge_policy: GaugePolicy = "reject"
    gauge_priors: Mapping[str, object] | None = None
    pose_model: PoseModel = "per_view"
    pose_translation_frame: PoseTranslationFrame = field(default="object", kw_only=True)
    knot_spacing: int = 8
    degree: int = 3
    seed_translations: bool = False
    # Volume masking before forward projection (modeling for ROI/truncation)
    # Options: "off" (default), "cyl" (cylindrical mask in x-y broadcast along z)
    mask_vol: str = "off"
    # Logging
    log_summary: bool = False
    log_compact: bool = True  # print one compact line per outer when log_summary is enabled
    # Fixed Lipschitz constant for the inner FISTA; None runs the power method.
    lipschitz: float | None = None
    # Early stopping across outers (alignment phase)
    early_stop: bool = True
    early_stop_rel_impr: float = 1e-3  # stop if (before-after)/before < this
    early_stop_patience: int = 2
    # Accept GN steps only when they improve the loss, up to gn_accept_tol.
    gn_accept_only_improving: bool = True
    gn_accept_tol: float = 0.0  # allow tiny increases if >0 (as fraction of before)
    # Data term / similarity
    loss: AlignmentLossConfig = field(default_factory=L2OtsuLossSpec)
    # The reconstruction tier of the stage being run: ``quality`` until a
    # schedule's stage runner sets that stage's own tier. Not a constructor option.
    stage_quality_tier: StageQualityTier = field(init=False, default="fast", repr=False)

    def __post_init__(self) -> None:
        if self.ray_integrator not in RAY_INTEGRATORS:
            raise ValueError(f"ray_integrator must be one of {RAY_INTEGRATORS}")
        self._apply_profile_policy()
        self._normalize_reconstruction_options()
        self._normalize_backend_options()
        self._normalize_optimizer_options()
        self._normalize_schedule_options()
        self._normalize_dof_options()
        self._normalize_gauge_options()
        self._normalize_pose_model_options()
        if self.gn_coupling == "joint":
            if self.pose_model != "per_view" or self.opt_method != "gn":
                raise ValueError("joint GN requires opt_method='gn' and pose_model='per_view'")
            if self.tv_weight != 0 and self.regulariser != "huber_tv":
                raise ValueError("joint GN supports Huber-TV or zero volume regularisation")

    def _apply_profile_policy(self) -> None:
        self.quality = normalize_quality(self.quality)
        self.stage_quality_tier = self.quality
        if self.quality == "reference":
            policy = alignment_profile_policy(self.quality)
            self.projector_backend = policy.projector_backend
            self.gather_dtype = policy.gather_dtype
            self.regulariser = policy.regulariser
            self.reconstruction = cast("Reconstruction", policy.reconstruction)
            self.views_per_batch = int(policy.views_per_batch)
            self.checkpoint_projector = bool(policy.checkpoint_projector)
            self.pose_model = cast("PoseModel", policy.pose_model)

    def _normalize_reconstruction_options(self) -> None:
        if self.reconstruction not in {"fista", "spdhg"}:
            raise ValueError(
                f"reconstruction must be 'fista' or 'spdhg', not {self.reconstruction!r}"
            )

    def _normalize_backend_options(self) -> None:
        self.projector_backend = normalize_projector_backend(self.projector_backend)
        self.gather_dtype = str(self.gather_dtype).strip().lower()

    def _normalize_optimizer_options(self) -> None:
        if self.opt_method not in {"gd", "gn", "lbfgs"}:
            raise ValueError("opt_method must be one of 'gd', 'gn', or 'lbfgs'")
        if self.gn_jacobian not in {"autodiff", "central"}:
            raise ValueError("gn_jacobian must be 'autodiff' or 'central'")
        if not math.isfinite(self.gn_difference_step) or self.gn_difference_step <= 0:
            raise ValueError("gn_difference_step must be finite and > 0")
        if self.gn_coupling not in {"fixed_volume", "joint"}:
            raise ValueError("gn_coupling must be fixed_volume or joint")
        if self.gn_joint_solver not in {"stacked", "pose_eliminated"}:
            raise ValueError("gn_joint_solver must be stacked or pose_eliminated")
        if (
            self.gn_joint_iterations < 1
            or int(self.gn_joint_iterations) != self.gn_joint_iterations
        ):
            raise ValueError("gn_joint_iterations must be a positive integer")
        if not math.isfinite(self.gn_joint_rtol) or not 0 < self.gn_joint_rtol < 1:
            raise ValueError("gn_joint_rtol must be finite and between zero and one")
        if not math.isfinite(self.gn_volume_damping) or self.gn_volume_damping <= 0:
            raise ValueError("gn_volume_damping must be finite and positive")
        if self.gn_coupling == "joint" and (
            not math.isfinite(self.gn_damping) or self.gn_damping <= 0
        ):
            raise ValueError("joint GN requires finite positive gn_damping")
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
        if self.schedule == "":
            self.schedule = None
        if self.schedule == "cor_then_pose" and self.pose_translation_frame != "detector":
            raise ValueError(
                "schedule 'cor_then_pose' needs pose_translation_frame='detector': only "
                "detector-frame translations express a constant detector shift at every view"
            )
        if self.optimise_dofs is not None:
            self.optimise_dofs = normalize_alignment_dofs(
                self.optimise_dofs,
                option_name="optimise_dofs",
            )

    def _normalize_dof_options(self) -> None:
        self.freeze = normalize_alignment_dofs(self.freeze, option_name="freeze")

    def _normalize_gauge_options(self) -> None:
        if self.gauge_policy not in {"reject", "anchor_mean", "prior_required", "diagnose_only"}:
            raise ValueError(
                "gauge_policy must be one of 'reject', 'anchor_mean', "
                "'prior_required', or 'diagnose_only'"
            )
        _ = _active_dof_mask_for_cfg(self)
        self.bounds = normalize_bounds(self.bounds, option_name="bounds")

    def _normalize_pose_model_options(self) -> None:
        if self.pose_translation_frame not in {"object", "detector"}:
            raise ValueError("pose_translation_frame must be 'object' or 'detector'")
        if self.pose_model not in {"per_view", "polynomial", "spline"}:
            raise ValueError("pose_model must be one of 'per_view', 'polynomial', or 'spline'")
        if self.pose_model == "polynomial" and int(self.degree) < 0:
            raise ValueError("degree must be >= 0 for polynomial pose_model")
        if self.pose_model == "spline":
            if int(self.knot_spacing) < 1:
                raise ValueError("knot_spacing must be >= 1 for spline pose_model")
            if int(self.degree) not in (1, 2, 3):
                raise ValueError("degree must be one of 1, 2, or 3 for spline pose_model")


def coupled_pose_config(**overrides: object) -> AlignConfig:
    """Return the recommended configuration for per-view pose alignment.

    This is what ``tomojax align --mode pose`` runs: each Gauss-Newton step
    solves the free voxels and every view's pose together, with Joseph
    plane sampling, an unregularised least-squares fit, fp32 gathers and up to
    30 early-stopped outer iterations, after a global search for each view's
    detector shift. Pass keyword overrides for any
    :class:`AlignConfig` field. ``AlignConfig()`` itself keeps the older
    alternating defaults for compatibility.
    """
    settings: dict[str, object] = {
        "gn_coupling": "joint",
        "gn_joint_solver": "pose_eliminated",
        "ray_integrator": "joseph",
        "gather_dtype": "fp32",
        "loss": L2LossSpec(),
        "tv_weight": 0.0,
        "outer_iterations": 30,
        "seed_translations": True,
    }
    settings.update(overrides)
    return AlignConfig(**settings)  # type: ignore[arg-type]


__all__ = [
    "AlignConfig",
    "_active_dof_mask_for_cfg",
    "_active_dofs_for_cfg",
    "_active_geometry_dofs_for_cfg",
    "_resolved_schedule_for_cfg",
    "_scoped_dofs_for_cfg",
    "coupled_pose_config",
    "resolved_schedule_for_config",
]
