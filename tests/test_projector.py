"""The reusable projector is the matched functional pair, with stable geometry."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj


def _geometry(kind):
    grid = tj.Grid(7, 5, 3, 0.8, 1.1, 1.3, vol_center=(0.2, -0.1, 0.3))
    detector = tj.Detector(9, 5, 0.9, 1.2, (0.17, -0.23))
    args = grid, detector, np.array([0.0, 31.0, 91.0, 179.0])
    if kind == "cone":
        return tj.ConeGeometry(*args, tj.ConeBeam(30.0, 50.0), tilt_deg=3)
    if kind == "lamino":
        return tj.LaminographyGeometry(*args, tilt_deg=31)
    return tj.ParallelGeometry(*args)


@pytest.mark.parametrize("kind", ["parallel", "lamino", "cone"])
def test_operator_matches_functions_and_independent_transpose(kind):
    geometry = _geometry(kind)
    operator = tj.Projector(geometry)
    rng = np.random.default_rng(6)
    x = jnp.asarray(rng.normal(size=operator.volume_shape), dtype=jnp.float32)
    y = jnp.asarray(rng.normal(size=operator.projection_shape), dtype=jnp.float32)
    projected, adjoint = operator(x), operator.adjoint(y)
    np.testing.assert_allclose(projected, tj.project(geometry, x), atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(adjoint, tj.backproject(geometry, y), atol=1e-5, rtol=3e-5)
    expected = jax.vjp(operator, x)[1](y)[0]
    np.testing.assert_allclose(adjoint, expected, atol=1e-5, rtol=3e-5)
    np.testing.assert_allclose(jnp.vdot(projected, y), jnp.vdot(x, adjoint), atol=1e-4, rtol=2e-5)
    compiled = jax.jit(lambda v: operator.adjoint(operator(v)))
    np.testing.assert_allclose(compiled(x), operator.adjoint(projected), atol=1e-5, rtol=3e-5)


def test_snapshot_grid_override_and_shape_checks():
    geometry = _geometry("parallel")
    grid = tj.Grid(5, 6, 2, 0.9, 1.0, 1.2)
    operator = tj.Projector(geometry, grid=grid)
    assert operator.volume_shape == (5, 6, 2)
    x = np.random.default_rng(8).normal(size=operator.volume_shape).astype(np.float32)
    expected = np.asarray(tj.project(geometry, x, grid=grid))
    geometry.angles[:] += 50
    np.testing.assert_allclose(operator(x), expected, atol=3e-6, rtol=3e-6)
    with pytest.raises(ValueError, match="volume shape"):
        operator(x.transpose(2, 1, 0))
    with pytest.raises(ValueError, match="projections shape"):
        operator.adjoint(expected[:, :-1])
