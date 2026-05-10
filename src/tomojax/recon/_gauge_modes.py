"""Volume gauge modes that mimic differentiable geometry tangents."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.forward import core_projection_geometry_from_state, project_parallel_reference
from tomojax.geometry import GeometryState
from tomojax.recon._backprojection_accumulation import sum_backproject_views_chunked

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class DetUGaugeMode:
    """Volume mode that best reproduces detector-u projection tangent."""

    mode: jax.Array
    transfer_ratio_before: float
    mode_norm: float
    mode_metric_norm: float
    cg_iterations: int
    cg_residual_norm: float
    provenance: dict[str, object]


def build_det_u_gauge_mode(
    volume: jax.Array,
    geometry: GeometryState,
    *,
    projection_valid_mask: jax.Array,
    regularisation: float = 1.0e-3,
    finite_difference_step: float = 1.0e-2,
    cg_max_iterations: int = 6,
) -> DetUGaugeMode:
    """Compute the local volume mode that can absorb detector-u motion."""
    vol = jnp.asarray(volume, dtype=jnp.float32)
    mask = jnp.asarray(projection_valid_mask, dtype=jnp.float32)
    tangent = _detu_projection_tangent(vol, geometry, eps=finite_difference_step) * mask
    fixed_curvature = float(jnp.vdot(tangent, tangent).real)
    core = core_projection_geometry_from_state(
        (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])),
        geometry,
        detector_shape=(int(mask.shape[1]), int(mask.shape[2])),
    )
    rhs = sum_backproject_views_chunked(core, tangent)
    mode, iterations, residual = _cg_solve(
        _matvec_for_geometry(
            geometry=geometry,
            mask=mask,
            regularisation=regularisation,
        ),
        rhs,
        max_iterations=cg_max_iterations,
    )
    transferred = project_parallel_reference(mode, geometry) * mask
    transferred_curvature = float(jnp.vdot(tangent, transferred).real)
    transfer_ratio = _safe_ratio(transferred_curvature, fixed_curvature)
    mode_metric_norm = float(jnp.vdot(mode, mode).real)
    return DetUGaugeMode(
        mode=mode.astype(jnp.float32),
        transfer_ratio_before=transfer_ratio,
        mode_norm=float(jnp.sqrt(jnp.maximum(mode_metric_norm, 0.0))),
        mode_metric_norm=mode_metric_norm,
        cg_iterations=iterations,
        cg_residual_norm=residual,
        provenance={
            "schema": "tomojax.det_u_gauge_mode.v1",
            "geometry_source": "current_alignment_geometry",
            "volume_source": "current_alignment_volume",
            "mask_source": "projection_valid_mask",
            "uses_truth": False,
            "regularisation_lambda": float(regularisation),
            "finite_difference_step": float(finite_difference_step),
        },
    )


def _detu_projection_tangent(
    volume: jax.Array,
    geometry: GeometryState,
    *,
    eps: float,
) -> jax.Array:
    current = float(geometry.setup.det_u_px.value)
    plus = project_parallel_reference(volume, _with_det_u(geometry, current + eps))
    minus = project_parallel_reference(volume, _with_det_u(geometry, current - eps))
    return (plus - minus) / jnp.asarray(2.0 * eps, dtype=jnp.float32)


def _matvec_for_geometry(
    *,
    geometry: GeometryState,
    mask: jax.Array,
    regularisation: float,
) -> Callable[[jax.Array], jax.Array]:
    def matvec(candidate: jax.Array) -> jax.Array:
        projected = project_parallel_reference(candidate, geometry) * mask
        core = core_projection_geometry_from_state(
            (int(candidate.shape[0]), int(candidate.shape[1]), int(candidate.shape[2])),
            geometry,
            detector_shape=(int(mask.shape[1]), int(mask.shape[2])),
        )
        return sum_backproject_views_chunked(core, projected) + jnp.asarray(
            regularisation,
            dtype=jnp.float32,
        ) * candidate

    return matvec


def _cg_solve(
    matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    max_iterations: int,
) -> tuple[jax.Array, int, float]:
    x = jnp.zeros_like(rhs)
    residual = rhs - matvec(x)
    direction = residual
    rs_old = jnp.vdot(residual, residual).real
    iterations = 0
    for _ in range(max(1, int(max_iterations))):
        mat_direction = matvec(direction)
        denom = jnp.maximum(
            jnp.vdot(direction, mat_direction).real,
            jnp.asarray(1.0e-12, dtype=jnp.float32),
        )
        alpha = rs_old / denom
        x = x + alpha * direction
        residual = residual - alpha * mat_direction
        rs_new = jnp.vdot(residual, residual).real
        iterations += 1
        if float(jnp.sqrt(rs_new)) < 1.0e-4:
            rs_old = rs_new
            break
        beta = rs_new / jnp.maximum(rs_old, jnp.asarray(1.0e-12, dtype=jnp.float32))
        direction = residual + beta * direction
        rs_old = rs_new
    return x, iterations, float(jnp.sqrt(rs_old))


def _with_det_u(geometry: GeometryState, det_u_px: float) -> GeometryState:
    setup = geometry.setup.replace_parameter(
        "det_u_px",
        geometry.setup.det_u_px.with_value(float(det_u_px)),
    )
    return GeometryState(setup=setup, pose=geometry.pose, acquisition=geometry.acquisition)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1.0e-12:
        return 0.0
    return float(numerator / denominator)
