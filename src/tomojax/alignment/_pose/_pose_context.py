from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from tomojax.alignment._config import AlignConfig
from tomojax.alignment._model.dofs import DOF_INDEX
from tomojax.alignment._model.gauge import (
    GaugeFixMode,
    apply_alignment_gauge,
    gauge_stats_to_python,
)
from tomojax.alignment._model.motion_models import (
    build_pose_motion_model,
    expand_motion_coefficients,
    fit_motion_coefficients,
    scan_coordinate_from_geometry,
)
from tomojax.alignment._objectives.loss_adapters import LossAdapter
from tomojax.alignment._observer import ObserverCallback
from tomojax.alignment._results import AlignResumeState
from tomojax.core.cone import ConeModel
from tomojax.core.geometry.base import Detector, Geometry, Grid
from tomojax.core.geometry.cone import is_cone_beam

from ._pose_block import feasible_pose_scale


def with_beam_translation(
    active: tuple[bool, ...], geometry: object, cfg: AlignConfig
) -> tuple[bool, ...]:
    """Add dy (along the beam) to cone-beam alignment that estimates dx and dz.

    A cone beam's magnification makes dy observable; parallel beams leave it
    inactive. ``freeze=("dy",)`` keeps it fixed.
    """
    dx, dz, dy = (DOF_INDEX[name] for name in ("dx", "dz", "dy"))
    if not is_cone_beam(geometry) or "dy" in cfg.freeze or not (active[dx] and active[dz]):
        return active
    return tuple(True if i == dy else value for i, value in enumerate(active))


def validate_resume_config(resume_state: AlignResumeState, cfg: AlignConfig) -> None:
    """Reject a resumed state whose iteration or physical convention changed."""
    if resume_state.start_outer_iter < 0 or resume_state.start_outer_iter > int(
        cfg.outer_iterations
    ):
        raise ValueError(
            "align resume_state start_outer_iter must be between 0 and cfg.outer_iterations; "
            f"got {resume_state.start_outer_iter} "
            f"for outer_iterations={int(cfg.outer_iterations)}"
        )
    if resume_state.pose_translation_frame != cfg.pose_translation_frame:
        raise ValueError("align resume_state pose_translation_frame differs from config")
    if resume_state.ray_integrator != cfg.ray_integrator:
        raise ValueError("align resume_state ray_integrator differs from config")


@dataclass(frozen=True)
class _AlignSetupState:
    cfg: AlignConfig
    observer_fn: ObserverCallback | None
    n_views: int
    x: jnp.ndarray
    pose_params: jnp.ndarray
    frozen_pose_params: jnp.ndarray
    active_mask_tuple: tuple[bool, ...]
    active_mask_bool: jnp.ndarray
    active_names: tuple[str, ...]
    active_mask: jnp.ndarray
    bounds_lower: jnp.ndarray
    bounds_upper: jnp.ndarray
    gauge_fix: GaugeFixMode
    gauge_dofs: tuple[str, ...]


@dataclass(frozen=True)
class PoseConstraintContext:
    active_mask_tuple: tuple[bool, ...]
    active_mask_bool: jnp.ndarray
    frozen_pose_params: jnp.ndarray
    bounds_lower: jnp.ndarray
    bounds_upper: jnp.ndarray
    gauge_fix: GaugeFixMode
    gauge_dofs: tuple[str, ...]

    @classmethod
    def from_setup(cls, setup: _AlignSetupState) -> PoseConstraintContext:
        return cls(
            active_mask_tuple=setup.active_mask_tuple,
            active_mask_bool=setup.active_mask_bool,
            frozen_pose_params=setup.frozen_pose_params,
            bounds_lower=setup.bounds_lower,
            bounds_upper=setup.bounds_upper,
            gauge_fix=setup.gauge_fix,
            gauge_dofs=setup.gauge_dofs,
        )

    def apply_param_constraints(self, candidate: jnp.ndarray) -> jnp.ndarray:
        clipped = jnp.clip(candidate, self.bounds_lower, self.bounds_upper)
        return jnp.where(self.active_mask_bool, clipped, self.frozen_pose_params)

    def apply_full_constraints(self, candidate: jnp.ndarray) -> jnp.ndarray:
        constrained = self.apply_param_constraints(candidate)
        gauged, _ = apply_alignment_gauge(
            constrained,
            mode=self.gauge_fix,
            active_mask=self.active_mask_tuple,
            bounds_lower=self.bounds_lower,
            bounds_upper=self.bounds_upper,
        )
        return self.apply_param_constraints(gauged)

    def apply_full_constraints_with_stats(
        self,
        candidate: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, float | str | list[str]]]:
        constrained = self.apply_param_constraints(candidate)
        gauged, stats = apply_alignment_gauge(
            constrained,
            mode=self.gauge_fix,
            active_mask=self.active_mask_tuple,
            bounds_lower=self.bounds_lower,
            bounds_upper=self.bounds_upper,
        )
        gauged = self.apply_param_constraints(gauged)
        final_gauged, final_stats = apply_alignment_gauge(
            gauged,
            mode=self.gauge_fix,
            active_mask=self.active_mask_tuple,
            bounds_lower=self.bounds_lower,
            bounds_upper=self.bounds_upper,
        )
        final_gauged = self.apply_param_constraints(final_gauged)
        stats_py = gauge_stats_to_python(stats)
        final_py = gauge_stats_to_python(final_stats)
        stats_py["dx_mean_after"] = final_py["dx_mean_after"]
        stats_py["dz_mean_after"] = final_py["dz_mean_after"]
        return final_gauged, stats_py

    def constrain_trajectory(self, candidate: jnp.ndarray) -> jnp.ndarray:
        """Keep an in-basis path in-basis while imposing gauge and box bounds.

        Constants lie in every supported trajectory model. Move from a feasible
        constant path toward the requested one, rather than clipping its views.
        Frozen columns retain their original, possibly nonconstant, values.
        """
        gauge_dofs = (
            (*self.gauge_dofs, "dy")
            if self.active_mask_bool[5] and "dy" not in self.gauge_dofs
            else self.gauge_dofs
        )
        for name in gauge_dofs:
            column = DOF_INDEX[name]
            candidate = candidate.at[:, column].add(-jnp.mean(candidate[:, column]))
        anchor = jnp.broadcast_to(jnp.mean(candidate, axis=0), candidate.shape)
        anchor = self.apply_full_constraints(anchor)
        direction = jnp.where(self.active_mask_bool, candidate - anchor, 0)
        scale = feasible_pose_scale(anchor, direction, self.bounds_lower, self.bounds_upper)
        return anchor + scale * direction

    def description(self) -> str:
        if self.gauge_fix == "none":
            return "none"
        gauge_dofs_label = ",".join(self.gauge_dofs) if self.gauge_dofs else "no translation DOFs"
        return f"{self.gauge_fix} over active {gauge_dofs_label}"


@dataclass(frozen=True)
class PoseMotionContext:
    motion_model: Any
    use_smooth_pose_model: bool
    pose_params: jnp.ndarray
    motion_coeffs: jnp.ndarray | None

    @classmethod
    def build(
        cls,
        *,
        geometry: Geometry,
        cfg: AlignConfig,
        n_views: int,
        active_names: tuple[str, ...],
        pose_params: jnp.ndarray,
        resume_state: AlignResumeState | None,
        constraint_ctx: PoseConstraintContext,
    ) -> PoseMotionContext:
        scan_coordinate = scan_coordinate_from_geometry(geometry, n_views)
        motion_model = build_pose_motion_model(
            pose_model=str(cfg.pose_model),
            n_views=n_views,
            active_dofs=active_names,
            frozen_pose_params=constraint_ctx.frozen_pose_params,
            scan_coordinate=scan_coordinate,
            knot_spacing=int(cfg.knot_spacing),
            degree=int(cfg.degree),
        )
        use_smooth_pose_model = motion_model.name != "per_view"
        motion_coeffs = None
        constrained_params = pose_params
        if use_smooth_pose_model:
            motion_coeffs = fit_motion_coefficients(motion_model, constrained_params)
            constrained_params = constraint_ctx.constrain_trajectory(
                expand_motion_coefficients(motion_model, motion_coeffs)
            )
            motion_coeffs = fit_motion_coefficients(motion_model, constrained_params)
            if resume_state is not None and resume_state.motion_coeffs is not None:
                resume_coeffs = jnp.asarray(resume_state.motion_coeffs, dtype=jnp.float32)
                if tuple(resume_coeffs.shape) != tuple(motion_coeffs.shape):
                    raise ValueError(
                        "align resume_state motion_coeffs shape mismatch: "
                        f"expected {tuple(motion_coeffs.shape)}, got {tuple(resume_coeffs.shape)}"
                    )
                motion_coeffs = resume_coeffs
                constrained_params = constraint_ctx.constrain_trajectory(
                    expand_motion_coefficients(motion_model, motion_coeffs)
                )

        return cls(
            motion_model=motion_model,
            use_smooth_pose_model=use_smooth_pose_model,
            pose_params=constrained_params,
            motion_coeffs=motion_coeffs,
        )


@dataclass(frozen=True)
class AlignmentRuntimeContext:
    pose_stack: jnp.ndarray
    det_grid: tuple[jnp.ndarray, jnp.ndarray]
    smoothness_weights: jnp.ndarray
    active_loss_name: str
    loss_adapter: LossAdapter
    loss_mask: jnp.ndarray | None
    has_loss_mask: bool
    objective_provenance: dict[str, str]
    nv: int
    nu: int
    # The views' cone arrangements and lab frames; None for a parallel beam.
    cone: ConeModel | None = None
    frames: jnp.ndarray | None = None


@dataclass(frozen=True)
class _PoseObjectiveContext:
    grid: Grid
    detector: Detector
    projections: jnp.ndarray
    cfg: AlignConfig
    n_views: int
    active_mask: jnp.ndarray
    pose_stack: jnp.ndarray
    det_grid: tuple[jnp.ndarray, jnp.ndarray]
    smoothness_weights: jnp.ndarray
    loss_adapter: LossAdapter
    nv: int
    nu: int
    loss_mask: jnp.ndarray | None
    has_loss_mask: bool
    # The views' cone arrangements and lab frames; None for a parallel beam.
    cone: ConeModel | None = None
    frames: jnp.ndarray | None = None


def _pose_objective_context(
    *,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    cfg: AlignConfig,
    n_views: int,
    active_mask: jnp.ndarray,
    runtime: AlignmentRuntimeContext,
) -> _PoseObjectiveContext:
    return _PoseObjectiveContext(
        grid=grid,
        detector=detector,
        projections=projections,
        cfg=cfg,
        n_views=n_views,
        active_mask=active_mask,
        pose_stack=runtime.pose_stack,
        det_grid=runtime.det_grid,
        smoothness_weights=runtime.smoothness_weights,
        loss_adapter=runtime.loss_adapter,
        nv=runtime.nv,
        nu=runtime.nu,
        loss_mask=runtime.loss_mask,
        has_loss_mask=runtime.has_loss_mask,
        cone=runtime.cone,
        frames=runtime.frames,
    )
