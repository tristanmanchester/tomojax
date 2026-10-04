"""Per-view normal equations against explicit Jacobians and physical tangents."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_joseph_projector import matrix, problem
from tomojax.forward import joseph_pose_normal_equations, project_joseph


def inputs(count=5):
    g, d, poses = problem()
    rng = np.random.default_rng(189)
    x = jnp.asarray(rng.normal(size=(g.nx, g.ny, g.nz)), jnp.float32)
    y = jnp.asarray(rng.normal(size=(len(poses), d.nv, d.nu)), jnp.float32)
    # Arbitrary combinations of physical rigid tangents. Overcomplete bases
    # deliberately produce singular positive-semidefinite normal matrices.
    generators = np.zeros((6, 4, 4), np.float32)
    for i in range(3):
        generators[i, (i + 1) % 3, (i + 2) % 3] = -1
        generators[i, (i + 2) % 3, (i + 1) % 3] = 1
        generators[i + 3, i, 3] = 1
    directions = np.einsum(
        "pij,njk,pa->nika", generators, poses, rng.uniform(-0.5, 0.5, (6, count))
    ).astype(np.float32)
    return g, d, jnp.asarray(poses), x, y, jnp.asarray(directions)


def explicit(poses, x, y, directions, g, d, interpolation):
    def project(parameters):
        changed = poses + jnp.einsum(
            "nijk,k->nij", directions, parameters, precision=jax.lax.Precision.HIGHEST
        )
        return project_joseph(x, changed, g, d, interpolation=interpolation)

    zero = jnp.zeros(directions.shape[-1], jnp.float32)
    prediction, jacobian = jax.jit(lambda p: (project(p), jax.jacfwd(project)(p)))(zero)
    residual = np.asarray(prediction, np.float64) - np.asarray(y)
    flat = residual.reshape(len(poses), -1)
    jacobian = np.asarray(jacobian, np.float64).reshape(len(poses), -1, directions.shape[-1])
    return (
        0.5 * np.sum(flat**2, axis=1),
        np.einsum("nmp,nm->np", jacobian, flat),
        np.einsum("nmp,nmq->npq", jacobian, jacobian),
        residual,
    )


@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("count", [1, 5, 16])
def test_normal_equations_match_explicit_per_view_jacobians(backend, interpolation, count):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    g, d, poses, x, y, directions = inputs(count)
    expected = explicit(poses, x, y, directions, g, d, interpolation)
    actual = jax.jit(
        lambda v, t, ds, y: joseph_pose_normal_equations(
            v, t, ds, y, g, d, backend=backend, interpolation=interpolation
        )
    )(x, poses, directions, y)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=4e-5, atol=4e-5)
    hessian = np.asarray(actual[2])
    np.testing.assert_array_equal(hessian, hessian.transpose(0, 2, 1))
    eig = np.linalg.eigvalsh(hessian.astype(np.float64))
    assert np.min(eig) > -3e-6 * np.max(eig)


@pytest.mark.gpu
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_normal_equations_match_independent_physical_translation_tangent(interpolation):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    g, d, poses, x, y, _ = inputs(1)
    poses = poses.at[:, :3, 3].add(jnp.array([0.043, 0.057, -0.079]))
    direction = (
        jnp.zeros((*poses.shape, 1), jnp.float32)
        .at[:, :3, 3, 0]
        .set(jnp.array([0.17, -0.31, 0.23]))
    )
    epsilon = 1e-6
    plus, minus = np.asarray(poses, np.float64).copy(), np.asarray(poses, np.float64).copy()
    plus += epsilon * np.asarray(direction[..., 0], np.float64)
    minus -= epsilon * np.asarray(direction[..., 0], np.float64)
    jac = (
        (matrix(plus, g, d, interpolation) - matrix(minus, g, d, interpolation))
        @ np.asarray(x).ravel()
        / (2 * epsilon)
    )
    jac = jac.reshape(len(poses), -1)
    residual = (matrix(np.asarray(poses), g, d, interpolation) @ np.asarray(x).ravel()).reshape(
        len(poses), -1
    ) - np.asarray(y).reshape(len(poses), -1)
    actual = jax.jit(
        lambda: joseph_pose_normal_equations(
            x, poses, direction, y, g, d, backend="pallas", interpolation=interpolation
        )
    )()
    np.testing.assert_allclose(
        np.asarray(actual[1])[:, 0], np.sum(jac * residual, axis=1), atol=3e-5, rtol=3e-5
    )
    np.testing.assert_allclose(
        np.asarray(actual[2])[:, 0, 0], np.sum(jac * jac, axis=1), atol=3e-5, rtol=3e-5
    )


@pytest.mark.gpu
def test_batched_normal_equations_and_zero_pose_directions():
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    g, d, poses, x, y, directions = inputs()
    fn = jax.jit(
        lambda v, ds: joseph_pose_normal_equations(
            v, poses, ds, y, g, d, backend="pallas", interpolation="cubic"
        )
    )
    volumes = jnp.stack([x, -0.7 * x])
    actual = jax.jit(jax.vmap(fn, in_axes=(0, None)))(volumes, directions)
    expected = [fn(v, directions) for v in volumes]
    for index, result in enumerate(expected):
        for a, b in zip(actual, result, strict=True):
            np.testing.assert_allclose(a[index], b, rtol=2e-6, atol=2e-6)
    _, gradient, hessian, residual = fn(x, jnp.zeros_like(directions))
    np.testing.assert_array_equal(gradient, 0)
    np.testing.assert_array_equal(hessian, 0)
    assert np.isfinite(np.asarray(residual)).all()


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_reference_normal_equations_support_further_volume_differentiation(interpolation):
    g, d, poses, x, y, directions = inputs(1)

    def fn(scale):
        return joseph_pose_normal_equations(
            scale * x, poses, directions, y, g, d, interpolation=interpolation
        )[2]

    value, tangent = jax.jit(lambda s: jax.jvp(fn, (s,), (jnp.float32(1),)))(jnp.float32(1))
    np.testing.assert_allclose(tangent, 2 * value, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize(
    "bad",
    [
        "rank",
        "view_count",
        "pose_shape",
        "empty",
        "too_many",
        "complex",
        "target_shape",
        "target_complex",
    ],
)
def test_invalid_normal_equation_inputs_are_rejected(bad):
    g, d, poses, x, y, directions = inputs()
    if bad == "rank":
        directions = directions[..., 0]
    elif bad == "view_count":
        directions = directions[:-1]
    elif bad == "pose_shape":
        directions = directions[:, :3]
    elif bad == "empty":
        directions = directions[..., :0]
    elif bad == "too_many":
        directions = jnp.zeros((*poses.shape, 17))
    elif bad == "complex":
        directions = directions.astype(jnp.complex64)
    elif bad == "target_shape":
        y = y[:-1]
    else:
        y = y.astype(jnp.complex64)
    with pytest.raises(ValueError, match="must be real"):
        joseph_pose_normal_equations(x, poses, directions, y, g, d)
