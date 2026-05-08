from __future__ import annotations

# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
import csv
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from tomojax.forward import project_parallel_reference
from tomojax.geometry import GeometryState
from tomojax.recon import (
    ReferenceFISTAConfig,
    centered_volume_support,
    fista_reconstruct_reference,
    reconstruct_backprojection_reference,
    write_fista_trace_csv,
)

# check-public-imports: allow-private
from tomojax.recon._fista_reference import _center_l2

if TYPE_CHECKING:
    from pathlib import Path


def test_reference_fista_reduces_projection_loss_and_keeps_nonnegative() -> None:
    truth = _tiny_volume()
    geometry = GeometryState.zeros(2)
    projections = project_parallel_reference(truth, geometry)
    initial = jnp.zeros_like(truth)

    result = fista_reconstruct_reference(
        projections,
        geometry,
        initial_volume=initial,
        config=ReferenceFISTAConfig(iterations=6, step_size=5e-3, tv_weight=1e-3),
    )

    assert len(result.trace) == 6
    assert result.trace[-1].loss < result.trace[0].loss
    assert result.trace[-1].backend == "core_trilinear_ray_explicit_adjoint"
    assert float(jnp.min(result.volume)) >= 0.0
    assert float(jnp.sum(result.volume)) > 0.0


def test_reference_fista_warm_start_and_trace_csv(tmp_path: Path) -> None:
    truth = _tiny_volume()
    geometry = GeometryState.zeros(1)
    projections = project_parallel_reference(truth, geometry)
    warm_start = jnp.ones_like(truth) * 0.1

    result = fista_reconstruct_reference(
        projections,
        geometry,
        initial_volume=warm_start,
        config=ReferenceFISTAConfig(iterations=2, step_size=5e-3, non_negative=False),
    )
    trace_path = write_fista_trace_csv(result, tmp_path / "fista_trace.csv")

    assert trace_path.exists()
    assert result.volume.shape == warm_start.shape
    assert not np.allclose(np.asarray(result.volume), np.asarray(warm_start))
    with trace_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["iteration"] for row in rows] == ["0", "1"]
    assert rows[0]["backend"] == "core_trilinear_ray_explicit_adjoint"


def test_reference_backprojection_uses_geometry_and_preserves_shape() -> None:
    truth = _tiny_volume()
    geometry = GeometryState.zeros(2)
    projections = project_parallel_reference(truth, geometry)

    volume = reconstruct_backprojection_reference(projections, geometry, depth=truth.shape[1])

    assert volume.shape == truth.shape
    assert volume.dtype == jnp.float32
    assert float(jnp.max(volume)) > 0.0
    assert float(jnp.mean(volume)) <= float(jnp.max(projections))
    reprojection = project_parallel_reference(volume, geometry)
    assert reprojection.shape == projections.shape


def test_reference_fista_projects_candidate_and_momentum_into_support() -> None:
    truth = jnp.ones((6, 6, 6), dtype=jnp.float32)
    geometry = GeometryState.zeros(2)
    projections = project_parallel_reference(truth, geometry)
    support = centered_volume_support((6, 6, 6), kind="cylindrical", radius_fraction=0.35)
    warm_start = jnp.ones_like(truth)

    result = fista_reconstruct_reference(
        projections,
        geometry,
        initial_volume=warm_start,
        volume_support=support,
        config=ReferenceFISTAConfig(iterations=3, step_size=2e-3),
    )

    outside_support = jnp.where(support > 0, 0.0, result.volume)
    assert float(jnp.max(jnp.abs(outside_support))) == 0.0
    assert float(jnp.min(result.volume)) >= 0.0


def test_reference_fista_center_l2_penalty_enters_regulariser() -> None:
    geometry = GeometryState.zeros(1)
    observed = jnp.zeros((1, 4, 4), dtype=jnp.float32)
    warm_start = jnp.zeros((4, 4, 4), dtype=jnp.float32).at[3, :, :].set(1.0)

    result = fista_reconstruct_reference(
        observed,
        geometry,
        initial_volume=warm_start,
        config=ReferenceFISTAConfig(
            iterations=1,
            step_size=1.0e-3,
            tv_weight=0.0,
            center_l2_weight=2.0,
        ),
    )

    assert result.trace[0].regulariser > 0.0
    assert result.config.center_l2_weight == 2.0


def test_reference_fista_center_l2_uses_core_x_y_axes() -> None:
    axis0_offset = jnp.zeros((4, 4, 4), dtype=jnp.float32).at[3, :, :].set(1.0)
    axis2_offset = jnp.zeros((4, 4, 4), dtype=jnp.float32).at[:, :, 3].set(1.0)

    assert float(_center_l2(axis0_offset)) > 0.0
    assert float(_center_l2(axis2_offset)) == 0.0


def test_centered_volume_support_generates_cylinder_and_sphere() -> None:
    cylinder = centered_volume_support((5, 7, 9), kind="cylindrical", radius_fraction=0.5)
    sphere = centered_volume_support((5, 7, 9), kind="spherical", radius_fraction=0.5)

    assert cylinder.shape == (5, 7, 9)
    assert sphere.shape == (5, 7, 9)
    assert cylinder.dtype == jnp.bool_
    assert sphere.dtype == jnp.bool_
    assert bool(cylinder[2, 3, 4])
    assert bool(sphere[2, 3, 4])
    assert bool(cylinder[2, 3, 0])
    assert bool(cylinder[2, 3, 8])
    assert not bool(cylinder[0, 0, 4])
    assert not bool(sphere[2, 3, 0])
    np.testing.assert_array_equal(np.asarray(cylinder[:, :, 0]), np.asarray(cylinder[:, :, 8]))
    assert int(jnp.sum(sphere)) < int(jnp.sum(cylinder))


def _tiny_volume() -> jnp.ndarray:
    volume = jnp.zeros((4, 4, 4), dtype=jnp.float32)
    volume = volume.at[1, :, 1].set(0.5)
    return volume.at[2, :, 3].set(0.8)
