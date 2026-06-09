from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align._config import AlignConfig
from tomojax.align._model.gauge import GaugeFixMode, apply_alignment_gauge, gauge_stats_to_python
from tomojax.align._model.motion_models import (
    build_pose_motion_model,
    expand_motion_coefficients,
    fit_motion_coefficients,
    scan_coordinate_from_geometry,
)
from tomojax.align._objectives.loss_adapters import LossAdapter
from tomojax.align._observer import ObserverCallback
from tomojax.align._results import AlignResumeState
from tomojax.core.geometry.base import Detector, Geometry, Grid
from tomojax.geometry import cylindrical_mask_xy


def _build_alignment_volume_mask(
    grid: Grid,
    detector: Detector,
    *,
    mask_vol: str,
) -> jnp.ndarray | None:
    mask_mode = str(mask_vol).lower()
    if mask_mode in ("off", "none", ""):
        return None
    if mask_mode not in ("cyl", "cylindrical"):
        raise ValueError("align mask_vol must be one of 'off' or 'cyl'")
    try:
        m_xy = cylindrical_mask_xy(grid, detector)
        return jnp.asarray(m_xy, dtype=jnp.float32)[:, :, None]
    except Exception as exc:
        raise ValueError(f"Failed to apply requested mask_vol={mask_mode!r}") from exc


@dataclass(frozen=True)
class _AlignSetupState:
    cfg: AlignConfig
    observer_fn: ObserverCallback | None
    n_views: int
    x: jnp.ndarray
    params5: jnp.ndarray
    frozen_params5: jnp.ndarray
    active_mask_tuple: tuple[bool, bool, bool, bool, bool]
    active_mask_bool: jnp.ndarray
    active_col_indices_np: np.ndarray
    active_names: tuple[str, ...]
    active_mask: jnp.ndarray
    bounds_lower: jnp.ndarray
    bounds_upper: jnp.ndarray
    gauge_fix: GaugeFixMode
    gauge_dofs: tuple[str, ...]


@dataclass(frozen=True)
class PoseConstraintContext:
    active_mask_tuple: tuple[bool, bool, bool, bool, bool]
    active_mask_bool: jnp.ndarray
    frozen_params5: jnp.ndarray
    bounds_lower: jnp.ndarray
    bounds_upper: jnp.ndarray
    gauge_fix: GaugeFixMode
    gauge_dofs: tuple[str, ...]

    @classmethod
    def from_setup(cls, setup: _AlignSetupState) -> PoseConstraintContext:
        return cls(
            active_mask_tuple=setup.active_mask_tuple,
            active_mask_bool=setup.active_mask_bool,
            frozen_params5=setup.frozen_params5,
            bounds_lower=setup.bounds_lower,
            bounds_upper=setup.bounds_upper,
            gauge_fix=setup.gauge_fix,
            gauge_dofs=setup.gauge_dofs,
        )

    def apply_param_constraints(self, candidate: jnp.ndarray) -> jnp.ndarray:
        clipped = jnp.clip(candidate, self.bounds_lower, self.bounds_upper)
        return jnp.where(self.active_mask_bool, clipped, self.frozen_params5)

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

    def description(self) -> str:
        if self.gauge_fix == "none":
            return "none"
        gauge_dofs_label = ",".join(self.gauge_dofs) if self.gauge_dofs else "no translation DOFs"
        return f"{self.gauge_fix} over active {gauge_dofs_label}"


@dataclass(frozen=True)
class PoseMotionContext:
    motion_model: Any
    use_smooth_pose_model: bool
    active_coeff_indices: jnp.ndarray
    params5: jnp.ndarray
    motion_coeffs: jnp.ndarray | None
    constraint_ctx: PoseConstraintContext

    @classmethod
    def build(
        cls,
        *,
        geometry: Geometry,
        cfg: AlignConfig,
        n_views: int,
        active_names: tuple[str, ...],
        params5: jnp.ndarray,
        resume_state: AlignResumeState | None,
        constraint_ctx: PoseConstraintContext,
    ) -> PoseMotionContext:
        scan_coordinate = scan_coordinate_from_geometry(geometry, n_views)
        motion_model = build_pose_motion_model(
            pose_model=str(cfg.pose_model),
            n_views=n_views,
            active_dofs=active_names,
            frozen_params5=constraint_ctx.frozen_params5,
            scan_coordinate=scan_coordinate,
            knot_spacing=int(cfg.knot_spacing),
            degree=int(cfg.degree),
        )
        use_smooth_pose_model = motion_model.name != "per_view"
        active_coeff_indices = jnp.asarray(motion_model.active_indices, dtype=jnp.int32)
        motion_coeffs = None
        constrained_params = params5
        if use_smooth_pose_model:
            motion_coeffs = fit_motion_coefficients(motion_model, constrained_params)
            constrained_params = constraint_ctx.apply_full_constraints(
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
                constrained_params = constraint_ctx.apply_full_constraints(
                    expand_motion_coefficients(motion_model, motion_coeffs)
                )

        return cls(
            motion_model=motion_model,
            use_smooth_pose_model=use_smooth_pose_model,
            active_coeff_indices=active_coeff_indices,
            params5=constrained_params,
            motion_coeffs=motion_coeffs,
            constraint_ctx=constraint_ctx,
        )

    def coeffs_to_constrained_params(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        return self.constraint_ctx.apply_full_constraints(
            expand_motion_coefficients(self.motion_model, coeffs)
        )

    def project_params_to_smooth(self, candidate: jnp.ndarray) -> jnp.ndarray:
        constrained = self.constraint_ctx.apply_full_constraints(candidate)
        coeffs = fit_motion_coefficients(self.motion_model, constrained)
        return self.coeffs_to_constrained_params(coeffs)

    def loss_and_grad_for(
        self,
        align_loss: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] | None:
        if not self.use_smooth_pose_model:
            return None

        def motion_align_loss(
            coeffs: jnp.ndarray, vol: jnp.ndarray, loss_rng_key: jnp.ndarray
        ) -> jnp.ndarray:
            return align_loss(self.coeffs_to_constrained_params(coeffs), vol, loss_rng_key)

        return jax.jit(jax.value_and_grad(motion_align_loss))


@dataclass(frozen=True)
class AlignmentRuntimeContext:
    pose_stack: jnp.ndarray
    det_grid: tuple[jnp.ndarray, jnp.ndarray]
    smoothness_weights: jnp.ndarray
    smoothness_gram: jnp.ndarray
    smoothness_weights_sq: jnp.ndarray
    medium_smoothness_weights_sq: jnp.ndarray
    trans_only_smoothness_weights_sq: jnp.ndarray
    light_smoothness_weights_sq: jnp.ndarray
    volume_mask: jnp.ndarray | None
    active_loss_name: str
    loss_adapter: LossAdapter
    loss_mask: jnp.ndarray | None
    has_loss_mask: bool
    supports_gauss_newton: bool
    objective_provenance: dict[str, str]
    nv: int
    nu: int
    chunk_size: int
    num_chunks: int
    empty_loss_mask_chunk: jnp.ndarray


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
    volume_mask: jnp.ndarray | None
    loss_adapter: LossAdapter
    per_view_loss_fn: Callable[..., jnp.ndarray]
    nv: int
    nu: int
    chunk_size: int
    num_chunks: int
    loss_mask: jnp.ndarray | None
    has_loss_mask: bool
    empty_loss_mask_chunk: jnp.ndarray


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
        volume_mask=runtime.volume_mask,
        loss_adapter=runtime.loss_adapter,
        per_view_loss_fn=runtime.loss_adapter.per_view_loss,
        nv=runtime.nv,
        nu=runtime.nu,
        chunk_size=runtime.chunk_size,
        num_chunks=runtime.num_chunks,
        loss_mask=runtime.loss_mask,
        has_loss_mask=runtime.has_loss_mask,
        empty_loss_mask_chunk=runtime.empty_loss_mask_chunk,
    )
