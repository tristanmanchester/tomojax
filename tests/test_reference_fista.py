from __future__ import annotations

# pyright: reportAny=false, reportUnknownMemberType=false
import csv
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from tomojax.forward import project_parallel_reference
from tomojax.geometry import GeometryState
from tomojax.recon import (
    ReferenceFISTAConfig,
    fista_reconstruct_reference,
    reconstruct_backprojection_reference,
    write_fista_trace_csv,
)

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
    assert result.trace[-1].backend == "jax_reference"
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
    assert rows[0]["backend"] == "jax_reference"


def test_reference_backprojection_uses_geometry_and_preserves_shape() -> None:
    truth = _tiny_volume()
    geometry = GeometryState.zeros(2)
    projections = project_parallel_reference(truth, geometry)

    volume = reconstruct_backprojection_reference(projections, geometry, depth=truth.shape[1])

    assert volume.shape == truth.shape
    assert volume.dtype == jnp.float32
    assert float(jnp.max(volume)) > 0.0
    reprojection = project_parallel_reference(volume, geometry)
    assert reprojection.shape == projections.shape


def _tiny_volume() -> jnp.ndarray:
    volume = jnp.zeros((4, 4, 4), dtype=jnp.float32)
    volume = volume.at[1, :, 1].set(0.5)
    return volume.at[2, :, 3].set(0.8)
