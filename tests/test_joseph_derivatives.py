"""First-order CUDA derivatives checked against AD and physical finite differences."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_joseph_projector import matrix, problem

# check-public-imports: allow-private
from tomojax.core.joseph import forward_jax, forward_project_planes, plane_coefficients

# check-public-imports: allow-private
from tomojax.core.pallas._pallas_joseph_derivatives import l2_loss_and_grads

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def cuda_only():
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")


def arrays(voxel=(0.8, 1.1, 1.3), spacing=(0.7, 1.2)):
    grid, detector, poses = problem(voxel, spacing)
    rng = np.random.default_rng(531)
    x = jnp.asarray(rng.normal(size=(grid.nx, grid.ny, grid.nz)), jnp.float32)
    y = jnp.asarray(rng.normal(size=(len(poses), detector.nv, detector.nu)), jnp.float32)
    poses = jnp.asarray(poses)
    return grid, detector, poses, x, y


def assert_close_tree(actual, expected, tolerance=3e-5):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    ("voxel", "spacing"),
    [((0.8, 1.1, 1.3), (0.7, 1.2)), ((2.0, 0.4, 1.3), (0.3, 0.5)), ((0.5, 0.6, 0.7), (1.4, 1.8))],
)
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_reverse_and_fused_loss_match_reference_with_tail_tiles(voxel, spacing, interpolation):
    g, d, poses, x, y = arrays(voxel, spacing)

    def loss(t, v, backend):
        pred = forward_project_planes(
            plane_coefficients(t, g, d), v, g, d, backend=backend, interpolation=interpolation
        )
        return 0.5 * jnp.sum((pred - y) ** 2)

    expected = jax.jit(jax.value_and_grad(lambda t, v: loss(t, v, "jax"), argnums=(0, 1)))(poses, x)
    actual = jax.jit(jax.value_and_grad(lambda t, v: loss(t, v, "pallas"), argnums=(0, 1)))(
        poses, x
    )
    assert_close_tree(actual, expected)

    @jax.jit
    def fused(t, v, target):
        coeff, pull = jax.vjp(lambda t: plane_coefficients(t, g, d), t)
        value, gc, gx = l2_loss_and_grads(coeff, v, target, g, d, interpolation=interpolation)
        return value, (pull(gc)[0], gx)

    result = fused(poses, x, y)
    assert_close_tree(result, expected)
    assert_close_tree(result, actual)


@pytest.mark.parametrize("active", ["volume", "coefficients", "both"])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_jvp_and_linearized_transpose_match_reference(active, interpolation):
    g, d, poses, x, y = arrays()
    cf = plane_coefficients(poses, g, d)
    rng = np.random.default_rng(602)
    dc = jnp.asarray(rng.normal(size=cf.shape), jnp.float32)
    dx = jnp.asarray(rng.normal(size=x.shape), jnp.float32)

    def function(backend):
        def fn(*args):
            c = cf if active == "volume" else args[0]
            v = x if active == "coefficients" else args[-1]
            return forward_project_planes(c, v, g, d, backend=backend, interpolation=interpolation)

        return fn

    args = (x,) if active == "volume" else (cf,) if active == "coefficients" else (cf, x)
    tangents = (dx,) if active == "volume" else (dc,) if active == "coefficients" else (dc, dx)
    expected = jax.jvp(function("jax"), args, tangents)
    fn = function("pallas")
    assert_close_tree(jax.jit(lambda *ts: jax.jvp(fn, args, ts))(*tangents), expected)
    primal, linear = jax.linearize(fn, *args)
    assert_close_tree((primal, jax.jit(linear)(*tangents)), expected)
    transposed = jax.jit(jax.linear_transpose(linear, *tangents))(y)
    wanted = jax.jit(lambda cot: jax.vjp(function("jax"), *args)[1](cot))(y)
    assert_close_tree(transposed, wanted)
    # The JVP/VJP duality tests their combination for signed, nontrivial inputs.
    left = np.vdot(np.asarray(expected[1]), np.asarray(y))
    right = sum(
        np.vdot(np.asarray(t), np.asarray(c)) for t, c in zip(tangents, transposed, strict=True)
    )
    assert abs(left - right) < 3e-5 * max(abs(left), abs(right), 1)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_batching_of_forward_reverse_and_pose_jacobians(interpolation):
    g, d, poses, x, y = arrays()
    coeff = plane_coefficients(poses, g, d)

    def fp(v, backend):
        return forward_project_planes(coeff, v, g, d, backend=backend, interpolation=interpolation)

    volumes = jnp.stack([x, 0.37 * x, -0.8 * x])
    for derivative in (False, True):
        functions = []
        for backend in ("pallas", "jax"):

            def loss(v, backend=backend, interpolation=interpolation):
                return jnp.sum(jnp.sin(fp(v, backend)) * y)

            fn = jax.grad(loss) if derivative else lambda v, backend=backend: fp(v, backend)
            functions.append(jax.jit(jax.vmap(fn))(volumes))
        assert_close_tree(*functions)

    def shifted(shift, backend):
        cf = plane_coefficients(poses.at[:, :3, 3].add(shift), g, d)
        return forward_project_planes(cf, x, g, d, backend=backend, interpolation=interpolation)

    shift = jnp.array([0.11, -0.03, 0.21], jnp.float32)
    wanted = jax.jit(jax.jacfwd(lambda s: shifted(s, "jax")))(shift)
    assert_close_tree(jax.jit(jax.jacfwd(lambda s: shifted(s, "pallas")))(shift), wanted)
    assert_close_tree(jax.jit(jax.jacrev(lambda s: shifted(s, "pallas")))(shift), wanted)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_linear_transpose_and_zero_derivatives(interpolation):
    g, d, poses, x, y = arrays()
    cf = plane_coefficients(poses, g, d)

    def fn(v):
        return forward_project_planes(cf, v, g, d, backend="pallas", interpolation=interpolation)

    expected = jax.linear_transpose(
        lambda v: forward_jax(cf, v, g, d, interpolation=interpolation), x
    )(y)
    assert_close_tree(jax.jit(jax.linear_transpose(fn, x))(y), expected)
    _, tangent = jax.jit(lambda a: jax.jvp(lambda _: fn(x), (a,), (a,)))(jnp.float32(1))
    np.testing.assert_array_equal(tangent, np.zeros_like(y))
    result = jax.jit(jax.jacfwd(lambda scale: fn(x * scale)))(jnp.float32(0.5))
    assert_close_tree(result, fn(x))


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_pose_tangent_matches_independent_physical_finite_difference(interpolation):
    g, d, poses, x, _ = arrays()
    # Move away from interpolation knots. Translation keeps the dominant plane
    # axis fixed, including views whose direction lies on an argmax tie.
    shifted = poses.at[:, :3, 3].add(jnp.array([0.043, 0.057, -0.079]))
    direction = jnp.array([0.17, -0.31, 0.23], jnp.float32)

    def fn(t):
        cf = plane_coefficients(shifted.at[:, :3, 3].add(t * direction), g, d)
        return forward_project_planes(cf, x, g, d, backend="pallas", interpolation=interpolation)

    actual = jax.jit(lambda: jax.jvp(fn, (jnp.float32(0),), (jnp.float32(1),))[1])()
    epsilon = 1e-6
    plus, minus = np.asarray(shifted, np.float64).copy(), np.asarray(shifted, np.float64).copy()
    plus[:, :3, 3] += epsilon * np.asarray(direction, np.float64)
    minus[:, :3, 3] -= epsilon * np.asarray(direction, np.float64)
    expected = (
        (matrix(plus, g, d, interpolation) - matrix(minus, g, d, interpolation))
        @ np.asarray(x).ravel()
        / (2 * epsilon)
    )
    np.testing.assert_allclose(np.asarray(actual).ravel(), expected, atol=8e-6, rtol=3e-5)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_rigid_rotation_tangent_matches_independent_physical_finite_difference(interpolation):
    g, d, poses, x, _ = arrays()
    poses = poses.at[:, :3, 3].add(jnp.array([0.043, 0.057, -0.079]))

    def turn(theta, xp):
        cosine, sine = xp.cos(theta), xp.sin(theta)
        return xp.array([[cosine, -sine, 0, 0], [sine, cosine, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def fn(theta):
        cf = plane_coefficients(turn(theta, jnp) @ poses, g, d)
        return forward_project_planes(cf, x, g, d, backend="pallas", interpolation=interpolation)

    theta = jnp.float32(0.071)
    actual = jax.jit(lambda: jax.jvp(fn, (theta,), (jnp.float32(1),))[1])()
    epsilon = 1e-6
    plus = turn(float(theta) + epsilon, np) @ np.asarray(poses, np.float64)
    minus = turn(float(theta) - epsilon, np) @ np.asarray(poses, np.float64)
    expected = (
        (matrix(plus, g, d, interpolation) - matrix(minus, g, d, interpolation))
        @ np.asarray(x).ravel()
        / (2 * epsilon)
    )
    np.testing.assert_allclose(np.asarray(actual).ravel(), expected, atol=4e-5, rtol=3e-5)


def test_cubic_derivatives_remain_accurate_near_zero_residual():
    g, d, poses, x, _ = arrays()
    cf = plane_coefficients(poses, g, d)
    target = jax.jit(
        lambda v: forward_project_planes(cf, v, g, d, backend="pallas", interpolation="cubic")
    )(x) * jnp.float32(0.99999)

    def loss(t, v, backend):
        pred = forward_project_planes(
            plane_coefficients(t, g, d), v, g, d, backend=backend, interpolation="cubic"
        )
        return 0.5 * jnp.sum((pred - target) ** 2)

    expected = jax.jit(jax.value_and_grad(lambda t, v: loss(t, v, "jax"), argnums=(0, 1)))(poses, x)
    actual = jax.jit(jax.value_and_grad(lambda t, v: loss(t, v, "pallas"), argnums=(0, 1)))(
        poses, x
    )
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert np.linalg.norm(np.asarray(a) - b) < 1e-5 * np.linalg.norm(b)
