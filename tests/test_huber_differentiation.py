"""TV proximal scales are respected and flat-region Huber derivatives stay finite."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment._geometry.geometry_applier import BaseGeometryArrays, apply_alignment_state

# check-public-imports: allow-private
from tomojax.alignment._model.state import AlignmentState

# check-public-imports: allow-private
from tomojax.alignment._objectives.recon_layer import ReconLayer, ReconLayerConfig

# check-public-imports: allow-private
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry

# check-public-imports: allow-private
from tomojax.recon._tv_ops import huber_tv_grad, huber_tv_value, prox_huber_tv_conj
from tomojax.recon.fista_tv import tv_proximal


def _edges(shape):
    rows = []
    for index in np.ndindex(shape):
        for axis in range(3):
            neighbor = list(index)
            neighbor[axis] += 1
            if neighbor[axis] < shape[axis]:
                row = np.zeros(shape)
                row[index] = -1
                row[tuple(neighbor)] = 1
                rows.append(row.ravel())
    return np.asarray(rows).reshape((-1, np.prod(shape)))


@pytest.mark.numerical
@pytest.mark.parametrize("shape", [(3, 4, 2), (1, 4, 2), (2, 1, 1), (1, 1, 1)])
@pytest.mark.parametrize("level", [0.0, 2.5])
def test_flat_huber_hessian_matches_independent_edge_matrix(shape, level):
    delta = 0.13
    x = jnp.full(shape, level, dtype=jnp.float32)
    direction = jnp.asarray(np.random.default_rng(715).normal(size=shape), dtype=x.dtype)
    edges = _edges(shape)
    expected = (edges.T @ edges @ np.asarray(direction).ravel() / delta).reshape(shape)

    def gradient(z):
        return huber_tv_grad(z, delta)

    jvp = jax.jit(lambda z, v: jax.jvp(gradient, (z,), (v,))[1])(x, direction)
    vjp = jax.jit(lambda z, v: jax.vjp(gradient, z)[1](v)[0])(x, direction)
    np.testing.assert_allclose(jvp, expected, rtol=2e-6, atol=3e-6)
    np.testing.assert_allclose(vjp, expected, rtol=2e-6, atol=3e-6)
    np.testing.assert_array_equal(gradient(x), np.zeros(shape))


@pytest.mark.numerical
def test_piecewise_constant_huber_hessian_matches_value_and_finite_differences():
    volume = np.zeros((4, 3, 2), np.float32)
    volume[2:] = 0.7
    x = jnp.asarray(volume)
    direction = jnp.asarray(np.random.default_rng(52).normal(size=x.shape), dtype=x.dtype)
    delta = 0.2

    def explicit(z):
        return huber_tv_grad(z, delta)

    automatic = jax.grad(lambda z: huber_tv_value(z, delta))
    actual = jax.jvp(explicit, (x,), (direction,))[1]
    expected = jax.jvp(automatic, (x,), (direction,))[1]
    epsilon = 1e-3
    finite = (explicit(x + epsilon * direction) - explicit(x - epsilon * direction)) / (2 * epsilon)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual, finite, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(explicit(x), automatic(x), rtol=2e-6, atol=2e-6)


@pytest.mark.numerical
@pytest.mark.parametrize("lam", [0.0, 0.25])
def test_huber_conjugate_prox_has_correct_derivative_at_zero(lam):
    x = jnp.zeros((3, 3, 2, 2), jnp.float32)
    direction = jnp.asarray(np.random.default_rng(831).normal(size=x.shape), dtype=x.dtype)
    sigma, delta = 0.5, 0.1

    def prox(z):
        return jnp.stack(prox_huber_tv_conj(*z, sigma=sigma, lam=lam, delta=delta))

    expected = lam / (lam + sigma * delta) * np.asarray(direction)
    jvp = jax.jit(lambda z, v: jax.jvp(prox, (z,), (v,))[1])(x, direction)
    vjp = jax.jit(lambda z, v: jax.vjp(prox, z)[1](v)[0])(x, direction)
    np.testing.assert_allclose(jvp, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(vjp, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(prox(x), x)


@pytest.mark.numerical
@pytest.mark.parametrize("scale", [1e-20, 1e-10, 1e-7, 1.0, 1e20])
@pytest.mark.parametrize("ratio", [0.0, 0.001, 0.1, 1.0])
def test_tv_prox_matches_analytic_two_voxel_solution_at_all_scales(scale, ratio):
    # One edge: min_u 0.5 ||u - (0, scale)||^2 + lambda |u1 - u0|.
    # Its solution moves both endpoints by min(lambda, scale/2).
    image = jnp.asarray([0.0, scale], dtype=jnp.float32).reshape((2, 1, 1))
    actual = tv_proximal(image, ratio * scale, iterations=100)
    shift = min(ratio, 0.5)
    expected = np.asarray([shift, 1.0 - shift])
    np.testing.assert_allclose(np.asarray(actual).ravel() / scale, expected, rtol=3e-6, atol=2e-7)


@pytest.mark.numerical
@pytest.mark.parametrize("scale", [1e-20, 1e-10, 1e-7, 1.0, 1e20])
def test_huber_dual_prox_matches_independent_radial_solution_at_all_scales(scale):
    vectors = np.asarray([[0.0, 0.0, 0.0], [0.03, 0.04, 0.0], [3.0, -4.0, 2.0]])
    lam, sigma, delta = 0.1, 0.2, 0.3
    shrunk = vectors * lam / (lam + sigma * delta)
    norms = np.linalg.norm(shrunk, axis=1, keepdims=True)
    expected = shrunk * np.minimum(1.0, lam / np.where(norms == 0, 1.0, norms))
    actual = prox_huber_tv_conj(
        *jnp.asarray(vectors.T * scale, dtype=jnp.float32),
        sigma=sigma,
        lam=lam * scale,
        delta=delta * scale,
    )
    actual = np.asarray(actual).T / scale
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)
    assert np.max(np.linalg.norm(actual, axis=1)) <= lam * (1 + 3e-6)


@pytest.mark.numerical
@pytest.mark.parametrize(
    ("input_scale", "radius"), [(1e20, 1e-20), (1e10, 1e-30), (1e-20, 1e20), (1e-30, 1e10)]
)
def test_huber_dual_projection_handles_independently_scaled_input_and_radius(input_scale, radius):
    vector = np.asarray([3.0, -4.0, 2.0])
    expected_scale = min(input_scale, radius / np.linalg.norm(vector))
    # sigma = 0 leaves just the conjugate domain's ball projection.
    actual = prox_huber_tv_conj(
        *jnp.asarray(vector * input_scale, dtype=jnp.float32), sigma=0.0, lam=radius, delta=1.0
    )
    np.testing.assert_allclose(np.asarray(actual) / expected_scale, vector, rtol=3e-6, atol=2e-7)


@pytest.mark.numerical
@pytest.mark.parametrize("kind", ["parallel", "anisotropic", "lamino"])
@pytest.mark.parametrize("mode", ["unrolled", "implicit"])
def test_regularized_reconstruction_derivatives_at_zero_match_dense_system(kind, mode):
    """Check measured-data and initialization gradients through the real layer."""
    shape = (2, 2, 2)
    spacing = (0.8, 1.2, 1.4) if kind == "anisotropic" else (1.0, 1.0, 1.0)
    grid = Grid(*shape, *spacing)
    detector = Detector(4, 3, 0.7, 0.9, (0.17, 0.23))
    angles = np.array([13.0, 71.0, 127.0], dtype=np.float32)
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt_deg=30)
        if kind == "lamino"
        else ParallelGeometry(grid, detector, angles)
    )
    state = AlignmentState.zeros(n_views=3)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    config = ReconLayerConfig(
        iterations=4,
        tv_weight=0.03,
        huber_delta=0.1,
        lipschitz=30.0,
        differentiation_mode=mode,
        implicit_damping=0.1,
        implicit_cg_iterations=64,
        implicit_cg_tol=1e-7,
    )
    layer = ReconLayer(base, grid, detector, config)
    effective = apply_alignment_state(base, state)

    def project(flat):
        return jax.vmap(
            lambda t: forward_project_view_T(
                t, grid, detector, flat.reshape(shape), det_grid=effective.det_grid
            )
        )(effective.pose_stack).ravel()

    matrix = np.asarray(jax.jacfwd(project)(jnp.zeros(np.prod(shape))), dtype=np.float64)
    edges = _edges(shape)
    hessian = matrix.T @ matrix + (config.tv_weight / config.huber_delta) * (edges.T @ edges)
    cotangent = np.linspace(-0.4, 1.0, np.prod(shape))
    if mode == "implicit":
        expected_data = matrix @ np.linalg.solve(
            hessian + config.implicit_damping * np.eye(np.prod(shape)), cotangent
        )
        expected_init = np.zeros(np.prod(shape))
    else:
        # Differentiate the quadratic-basin FISTA recurrence using dense matrices,
        # independently of JAX's implementation or automatic differentiation.
        jac_x = np.concatenate([np.zeros((matrix.shape[1], matrix.shape[0])), np.eye(8)], axis=1)
        jac_z = jac_x.copy()
        forcing = np.concatenate([matrix.T, np.zeros((8, 8))], axis=1) / config.lipschitz
        transition = np.eye(8) - hessian / config.lipschitz
        t = 1.0
        for _ in range(config.iterations):
            jac_next = transition @ jac_z + forcing
            t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
            jac_z = jac_next + (t - 1) / t_next * (jac_next - jac_x)
            jac_x, t = jac_next, t_next
        expected_data = jac_x[:, : matrix.shape[0]].T @ cotangent
        expected_init = jac_x[:, matrix.shape[0] :].T @ cotangent

    def value(data, initial):
        x = layer.reconstruct(state=state, projections=data, init_x=initial).x
        return jnp.vdot(x.ravel(), jnp.asarray(cotangent, dtype=x.dtype))

    data = jnp.zeros((3, detector.nv, detector.nu), jnp.float32)
    initial = jnp.zeros(shape, jnp.float32)
    grad_data, grad_init = jax.jit(jax.grad(value, argnums=(0, 1)))(data, initial)
    np.testing.assert_allclose(grad_data.ravel(), expected_data, rtol=3e-4, atol=2e-5)
    np.testing.assert_allclose(grad_init.ravel(), expected_init, rtol=3e-4, atol=2e-5)
