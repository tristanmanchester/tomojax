"""Deterministic reference-FISTA scalar and gradient diagnostics."""
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.forward import (
    ResidualFilterConfig,
    core_projection_geometry_from_arrays,
    core_projection_geometry_from_state,
    project_parallel_reference,
)
from tomojax.geometry import GeometryState
from tomojax.recon._backprojection_accumulation import sum_backproject_views_chunked
from tomojax.recon._fista_reference import (
    ReferenceFISTAConfig,
    _loss_and_explicit_gradient,
    _project_views,
    fista_reconstruct_reference,
)
from tomojax.recon._support import centered_volume_support

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax.forward import CoreProjectionGeometry


@dataclass(frozen=True)
class ReferenceFISTADiagnosticArtifacts:
    """JSON/CSV-ready diagnostic payloads for the reference FISTA contract."""

    fista_gradient_checks: Mapping[str, object]
    adjoint_checks: Mapping[str, object]
    geometry_jvp_vjp_checks: Mapping[str, object]
    loss_normalisation_report: Mapping[str, object]
    fista_trace_recomputed_rows: tuple[Mapping[str, object], ...]


def reference_fista_diagnostic_artifacts() -> ReferenceFISTADiagnosticArtifacts:
    """Run small deterministic scalar/gradient diagnostics for reference FISTA."""
    geometry = GeometryState.zeros(2)
    volume = _diagnostic_volume()
    observed = project_parallel_reference(volume, geometry)
    core = core_projection_geometry_from_state(
        _shape3(volume),
        geometry,
        detector_shape=(int(observed.shape[1]), int(observed.shape[2])),
    )
    return ReferenceFISTADiagnosticArtifacts(
        fista_gradient_checks=_fista_gradient_checks(volume, observed, core),
        adjoint_checks=_adjoint_checks(volume, core),
        geometry_jvp_vjp_checks=_geometry_jvp_vjp_checks(volume, geometry),
        loss_normalisation_report=_loss_normalisation_report(volume, observed, core),
        fista_trace_recomputed_rows=_fista_trace_recomputed_rows(volume, observed, geometry, core),
    )


def write_fista_trace_recomputed_csv(
    rows: tuple[Mapping[str, object], ...],
    path: str | Path,
) -> Path:
    """Write recomputed FISTA trace diagnostics."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "trace_loss",
        "recomputed_final_loss",
        "absolute_difference",
        "trace_loss_point",
        "recomputed_loss_point",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return output_path


def _fista_gradient_checks(
    volume: jax.Array,
    observed: jax.Array,
    core: CoreProjectionGeometry,
) -> dict[str, object]:
    cases = [
        (
            "raw_valid_mask",
            ReferenceFISTAConfig(
                residual_filters=(ResidualFilterConfig(kind="raw"),),
                residual_loss_mode="l2",
                non_negative=False,
            ),
            jnp.ones_like(observed).at[:, 0, :].set(0.0),
        ),
        (
            "lowpass_boundary_mask",
            ReferenceFISTAConfig(
                residual_filters=(ResidualFilterConfig(kind="lowpass_gaussian", sigma_px=0.8),),
                residual_loss_mode="pseudo_huber",
                non_negative=False,
            ),
            jnp.ones_like(observed).at[:, :, 0].set(0.0),
        ),
        (
            "dog_tv_center",
            ReferenceFISTAConfig(
                residual_filters=(
                    ResidualFilterConfig(
                        kind="bandpass_difference_of_gaussians",
                        sigma_px=0.7,
                        outer_sigma_px=1.4,
                    ),
                ),
                tv_weight=2.0e-3,
                center_l2_weight=0.1,
                non_negative=False,
            ),
            jnp.ones_like(observed).at[:, -1, :].set(0.0),
        ),
    ]
    direction = _unit_direction(volume)
    records = [
        _finite_difference_record(
            name=name,
            volume=volume,
            observed=observed,
            core=core,
            config=config,
            mask=mask,
            direction=direction,
        )
        for name, config, mask in cases
    ]
    support = centered_volume_support(_shape3(volume), kind="cylindrical", radius_fraction=0.4)
    constrained = fista_reconstruct_reference(
        observed,
        GeometryState.zeros(int(observed.shape[0])),
        initial_volume=jnp.ones_like(volume),
        volume_support=support,
        config=ReferenceFISTAConfig(iterations=1, step_size=1.0e-3),
    ).volume
    outside_support = jnp.where(support, 0.0, constrained)
    support_record = {
        "name": "support_projection",
        "passed": float(jnp.max(jnp.abs(outside_support))) == 0.0,
        "max_abs_outside_support": float(jnp.max(jnp.abs(outside_support))),
    }
    return {
        "schema": "tomojax.fista_gradient_checks.v1",
        "status": "passed" if all(bool(record["passed"]) for record in records) else "failed",
        "normalizer": "full_projection_array_size",
        "cases": records,
        "support_check": support_record,
    }


def _finite_difference_record(
    *,
    name: str,
    volume: jax.Array,
    observed: jax.Array,
    core: CoreProjectionGeometry,
    config: ReferenceFISTAConfig,
    mask: jax.Array | None,
    direction: jax.Array,
) -> dict[str, object]:
    loss, _data, _regulariser, gradient = _loss_and_explicit_gradient(
        volume,
        observed,
        core=core,
        mask=mask,
        config=config,
    )
    epsilon = jnp.asarray(1.0e-2, dtype=jnp.float32)
    plus, _plus_data, _plus_reg, _plus_grad = _loss_and_explicit_gradient(
        volume + epsilon * direction,
        observed,
        core=core,
        mask=mask,
        config=config,
    )
    minus, _minus_data, _minus_reg, _minus_grad = _loss_and_explicit_gradient(
        volume - epsilon * direction,
        observed,
        core=core,
        mask=mask,
        config=config,
    )
    finite_difference = (plus - minus) / (jnp.asarray(2.0, dtype=jnp.float32) * epsilon)
    directional_gradient = jnp.sum(gradient * direction)
    abs_error = float(jnp.abs(directional_gradient - finite_difference))
    denom = float(jnp.maximum(jnp.abs(finite_difference), jnp.asarray(1.0e-6)))
    rel_error = abs_error / denom
    passed = bool(rel_error <= 5.0e-2 or abs_error <= 5.0e-3)
    return {
        "name": name,
        "passed": passed,
        "loss": float(loss),
        "finite_difference": float(finite_difference),
        "directional_gradient": float(directional_gradient),
        "absolute_error": abs_error,
        "relative_error": rel_error,
        "mask_valid_fraction": _mask_valid_fraction(mask, observed),
        "residual_filters": [filter_config.kind for filter_config in config.residual_filters],
        "tv_weight": float(config.tv_weight),
        "center_l2_weight": float(config.center_l2_weight),
    }


def _adjoint_checks(volume: jax.Array, core: CoreProjectionGeometry) -> dict[str, object]:
    projection = _project_views(volume, core=core, t_all=core.t_all)
    residual = jnp.sin(jnp.arange(projection.size, dtype=jnp.float32)).reshape(projection.shape)
    backprojected = sum_backproject_views_chunked(core, residual)
    lhs = jnp.vdot(projection, residual).real
    rhs = jnp.vdot(volume, backprojected).real
    abs_error = float(jnp.abs(lhs - rhs))
    denom = float(jnp.maximum(jnp.abs(lhs), jnp.asarray(1.0e-6)))
    rel_error = abs_error / denom
    return {
        "schema": "tomojax.adjoint_checks.v1",
        "status": "passed" if rel_error <= 2.0e-2 or abs_error <= 2.0e-3 else "failed",
        "cases": [
            {
                "name": "core_projector_backprojector",
                "lhs_projection_dot_residual": float(lhs),
                "rhs_volume_dot_backprojection": float(rhs),
                "absolute_error": abs_error,
                "relative_error": rel_error,
            }
        ],
    }


def _geometry_jvp_vjp_checks(volume: jax.Array, geometry: GeometryState) -> dict[str, object]:
    theta = jnp.asarray(geometry.theta_total_rad(), dtype=jnp.float32)
    zeros = jnp.zeros_like(theta)

    def project_at_det_u(det_u_px: jax.Array) -> jax.Array:
        core = core_projection_geometry_from_arrays(
            _shape3(volume),
            theta_rad=theta,
            dx_px=jnp.full_like(theta, det_u_px),
            dz_px=zeros,
            detector_shape=(int(volume.shape[0]), int(volume.shape[2])),
        )
        return _project_views(volume, core=core, t_all=core.t_all)

    det_u = jnp.asarray(0.25, dtype=jnp.float32)
    tangent = jnp.asarray(1.0, dtype=jnp.float32)
    base, jvp_value = jax.jvp(project_at_det_u, (det_u,), (tangent,))
    epsilon = jnp.asarray(1.0e-2, dtype=jnp.float32)
    finite_difference = (project_at_det_u(det_u + epsilon) - project_at_det_u(det_u - epsilon)) / (
        jnp.asarray(2.0, dtype=jnp.float32) * epsilon
    )
    residual = jnp.cos(jnp.arange(base.size, dtype=jnp.float32)).reshape(base.shape)
    def scalar(u: jax.Array) -> jax.Array:
        return jnp.vdot(project_at_det_u(u), residual).real

    vjp_gradient = jax.grad(scalar)(det_u)
    residual_dot_jvp = jnp.vdot(residual, jvp_value).real
    jvp_abs_error = float(jnp.linalg.norm(jvp_value - finite_difference))
    jvp_denom = float(jnp.maximum(jnp.linalg.norm(finite_difference), jnp.asarray(1.0e-6)))
    jvp_rel_error = jvp_abs_error / jvp_denom
    vjp_abs_error = float(jnp.abs(vjp_gradient - residual_dot_jvp))
    vjp_denom = float(jnp.maximum(jnp.abs(residual_dot_jvp), jnp.asarray(1.0e-6)))
    vjp_rel_error = vjp_abs_error / vjp_denom
    return {
        "schema": "tomojax.geometry_jvp_vjp_checks.v1",
        "status": "passed"
        if (jvp_rel_error <= 5.0e-2 or jvp_abs_error <= 5.0e-3)
        and (vjp_rel_error <= 5.0e-2 or vjp_abs_error <= 5.0e-3)
        else "failed",
        "det_u_px": float(det_u),
        "jvp": {
            "finite_difference_norm": float(jnp.linalg.norm(finite_difference)),
            "jvp_norm": float(jnp.linalg.norm(jvp_value)),
            "absolute_error": jvp_abs_error,
            "relative_error": jvp_rel_error,
        },
        "vjp": {
            "residual_dot_jvp": float(residual_dot_jvp),
            "vjp_gradient": float(vjp_gradient),
            "absolute_error": vjp_abs_error,
            "relative_error": vjp_rel_error,
        },
    }


def _loss_normalisation_report(
    volume: jax.Array,
    observed: jax.Array,
    core: CoreProjectionGeometry,
) -> dict[str, object]:
    mask = jnp.ones_like(observed).at[:, 0, :].set(0.0).at[:, :, -1].set(0.0)
    config = ReferenceFISTAConfig(
        residual_filters=(ResidualFilterConfig(kind="raw"),),
        residual_loss_mode="l2",
        non_negative=False,
    )
    loss, _data, _regulariser, _grad = _loss_and_explicit_gradient(
        volume,
        observed,
        core=core,
        mask=mask,
        config=config,
    )
    predicted = _project_views(volume, core=core, t_all=core.t_all)
    residual = (predicted - observed) * mask
    sum_loss = jnp.asarray(0.5, dtype=jnp.float32) * jnp.sum(residual * residual)
    full_count = jnp.asarray(observed.size, dtype=jnp.float32)
    valid_count = jnp.sum(mask)
    per_full = sum_loss / jnp.maximum(full_count, 1.0)
    per_valid = sum_loss / jnp.maximum(valid_count, 1.0)
    return {
        "schema": "tomojax.loss_normalisation_report.v1",
        "status": "reported",
        "current_contract": "full_projection_array_size",
        "transition_note": "valid-residual normalisation is reported but not yet enabled",
        "sum_loss": float(sum_loss),
        "loss_from_reference_fista": float(loss),
        "loss_per_array_pixel": float(per_full),
        "loss_per_valid_residual": float(per_valid),
        "array_pixel_count": int(observed.size),
        "valid_residual_count": int(valid_count),
        "mask_valid_fraction": _mask_valid_fraction(mask, observed),
        "reference_matches_array_pixel_normalisation": bool(
            jnp.abs(loss - per_full) <= jnp.asarray(1.0e-6, dtype=jnp.float32)
        ),
    }


def _fista_trace_recomputed_rows(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    core: CoreProjectionGeometry,
) -> tuple[Mapping[str, object], ...]:
    config = ReferenceFISTAConfig(
        iterations=2,
        step_size=2.0e-3,
        tv_weight=1.0e-3,
        residual_filters=(ResidualFilterConfig(kind="lowpass_gaussian", sigma_px=0.8),),
        non_negative=False,
    )
    result = fista_reconstruct_reference(
        observed,
        geometry,
        initial_volume=volume * jnp.asarray(0.5, dtype=jnp.float32),
        config=config,
    )
    final_loss, _data, _regulariser, _grad = _loss_and_explicit_gradient(
        result.volume,
        observed,
        core=core,
        mask=None,
        config=config,
    )
    return tuple(
        {
            "iteration": row.iteration,
            "trace_loss": row.loss,
            "recomputed_final_loss": float(final_loss),
            "absolute_difference": abs(float(row.loss) - float(final_loss)),
            "trace_loss_point": "momentum_point",
            "recomputed_loss_point": "returned_final_volume",
        }
        for row in result.trace
    )


def _diagnostic_volume() -> jax.Array:
    volume = jnp.zeros((4, 4, 4), dtype=jnp.float32)
    volume = volume.at[1, :, 1].set(0.5)
    volume = volume.at[2, :, 3].set(0.8)
    return volume.at[3, 1:3, 2].set(0.3)


def _unit_direction(volume: jax.Array) -> jax.Array:
    direction = jnp.cos(jnp.arange(volume.size, dtype=jnp.float32)).reshape(volume.shape)
    return direction / jnp.linalg.norm(direction)


def _shape3(volume: jax.Array) -> tuple[int, int, int]:
    return cast("tuple[int, int, int]", tuple(int(dim) for dim in volume.shape))


def _mask_valid_fraction(mask: jax.Array | None, observed: jax.Array) -> float:
    if mask is None:
        return 1.0
    mask_arr = np.asarray(mask, dtype=np.float32)
    return float(np.count_nonzero(mask_arr)) / float(observed.size)
