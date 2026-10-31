"""The reusable projector is the matched functional pair, with stable geometry."""

from dataclasses import replace

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
    assert operator.volume_shape == (2, 6, 5)
    x = np.random.default_rng(8).normal(size=operator.volume_shape).astype(np.float32)
    expected = np.asarray(tj.project(geometry, x, grid=grid))
    geometry.angles[:] += 50
    np.testing.assert_allclose(operator(x), expected, atol=3e-6, rtol=3e-6)
    # Calls share a projector only for equal arguments, so they see the change.
    assert not np.allclose(tj.project(geometry, x, grid=grid), expected, atol=1e-3)
    with pytest.raises(ValueError, match="volume shape"):
        operator(x.transpose(2, 1, 0))
    with pytest.raises(ValueError, match="projections shape"):
        operator.adjoint(expected[:, :-1])


def test_equal_calls_share_a_projector_until_a_setting_changes(monkeypatch):
    # check-public-imports: allow-private
    from tomojax.recon._operators import _projector

    monkeypatch.setenv("TOMOJAX_CUDA_KERNELS", "1")
    x = jnp.ones((3, 5, 7))
    first = _projector(_geometry("parallel"), None, None, x)
    assert _projector(_geometry("parallel"), None, None, x) is first
    with jax.default_device(jax.devices("cpu")[0]):
        assert _projector(_geometry("parallel"), None, None, x) is not first
    monkeypatch.setenv("TOMOJAX_CUDA_KERNELS", "0")
    assert _projector(_geometry("parallel"), None, None, x) is not first
    # One built inside a trace stays there.
    traced = jax.jit(lambda v: tj.project(_geometry("lamino"), v))(x)
    np.testing.assert_allclose(tj.project(_geometry("lamino"), x), traced, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kind", ["parallel", "lamino", "cone"])
def test_single_slice_is_only_a_squeeze_of_the_three_dimensional_pair(kind):
    geometry = replace(
        _geometry(kind), grid=tj.Grid(7, 5, 1, 0.8, 1.1, 1.3), detector=tj.Detector(9, 1, 0.9, 1.3)
    )
    operator = tj.Projector(geometry)
    x = jnp.arange(35, dtype=jnp.float32).reshape(5, 7) / 35
    y = jnp.arange(36, dtype=jnp.float32).reshape(4, 9) / 36
    np.testing.assert_allclose(operator(x), operator(x[None])[:, 0], atol=1e-6)
    np.testing.assert_allclose(operator.adjoint(y), operator.adjoint(y[:, None])[0], atol=1e-6)
    np.testing.assert_allclose(jax.jit(operator)(x), tj.project(geometry, x), atol=1e-6)
    np.testing.assert_allclose(operator.adjoint(y), tj.backproject(geometry, y), atol=1e-6)
    np.testing.assert_allclose(
        jax.grad(lambda v: jnp.vdot(operator(v), y))(x), operator.adjoint(y), atol=1e-5, rtol=3e-5
    )
    np.testing.assert_allclose(
        jnp.vdot(operator(x), y), jnp.vdot(x, operator.adjoint(y)), atol=1e-5, rtol=3e-5
    )


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["parallel", "lamino"])
def test_private_storage_calls_give_the_public_bits(kind, monkeypatch):
    # A call takes a z-fast copy when it fits; without room, or inside a trace,
    # it reads the (z, y, x) array. Four planes make the parallel forward band-staged.
    monkeypatch.setenv("TOMOJAX_CUDA_KERNELS", "1")
    geometry = replace(
        _geometry(kind), grid=tj.Grid(7, 5, 4, 1, 1, 1), detector=tj.Detector(9, 4, 1, 1)
    )
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(4, 5, 7)), dtype=jnp.float32)
    y = jnp.asarray(rng.normal(size=(4, 4, 9)), dtype=jnp.float32)
    operator = tj.Projector(geometry)
    if operator._build_private is None:
        pytest.skip("requires the CUDA Joseph projector")
    results = [operator(x), operator.adjoint(y)]
    traced = jax.jit(lambda v, w: (tj.project(geometry, v), tj.backproject(geometry, w)))(x, y)
    monkeypatch.setattr("tomojax.recon._operators.device_free_memory_bytes", lambda: 0)
    for public in ([operator(x), operator.adjoint(y)], traced):
        for private, other in zip(results, public, strict=True):
            np.testing.assert_array_equal(
                np.asarray(private).view(np.uint32), np.asarray(other).view(np.uint32)
            )
    # Axial views are transposed in bands in one call, adding the batches' sums in turn.
    monkeypatch.setattr("tomojax.recon._operators._BATCH", 3)
    monkeypatch.setattr("tomojax.recon._operators._AXIAL_STACK_BYTES", 0)
    batched = tj.Projector(geometry)
    one_call, per_batch = (
        np.asarray(v).view(np.uint32) for v in (batched.adjoint(y), batched._adjoint(y))
    )
    np.testing.assert_array_equal(one_call, per_batch)
