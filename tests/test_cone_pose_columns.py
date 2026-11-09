"""CUDA pose columns follow the requested analytic or finite-difference model."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import (
    ConeBeam,
    ConeGeometry,
    Detector,
    Grid,
    stack_view_poses,
)

from ._pose_reference import pose_update

pytestmark = pytest.mark.numerical


@pytest.mark.gpu
@pytest.mark.parametrize("distance", [80.0, 8000.0])
def test_cuda_forward_pose_and_volume_ad_match_reference(distance):
    from tomojax.core import cone

    if not cone.use_cuda_cone():
        pytest.skip("requires CUDA cone kernels")
    grid = Grid(7, 6, 5, 0.7, 1.1, 0.9, vol_center=(0.23, -0.17, 0.31))
    detector = Detector(9, 7, 1.3, 0.8, (0.17, -0.13))
    geometry = ConeGeometry(
        grid, detector, [13.7, 78.1, 147.3], ConeBeam(distance, 2 * distance), tilt_deg=53
    )
    nominal = stack_view_poses(geometry, 3)
    rng = np.random.default_rng(892)
    # Preserve the physical random field across the volume-axis convention.
    volume = jnp.asarray(rng.uniform(-0.1, 1.1, (7, 6, 5)).transpose(2, 1, 0).copy(), jnp.float32)
    params = jnp.asarray(rng.normal(0, 0.02, (3, 6)), jnp.float32)
    tangent = jnp.asarray(rng.normal(0, 0.1, params.shape), jnp.float32)
    dvolume = jnp.asarray(rng.normal(size=(7, 6, 5)).transpose(2, 1, 0).copy(), jnp.float32)
    cotangent = jnp.asarray(rng.normal(size=(3, 7, 9)), jnp.float32)

    def coefficients(p):
        poses = jax.vmap(lambda t, q: pose_update(t, q, translation_frame="detector"))(nominal, p)
        return cone.cone_coefficients(poses, grid, detector, geometry.beam)

    def project_coeff(coeff, x, backend):
        return cone.cone_project(x, coeff, grid, detector, backend=backend)

    def project(p, x, backend):
        return project_coeff(coefficients(p), x, backend)

    cuda, reference = (partial(project, backend=b) for b in ("cuda", "jax"))
    expected = jax.jit(lambda p, x, dp, dx: jax.jvp(reference, (p, x), (dp, dx)))(
        params, volume, tangent, dvolume
    )
    actual = jax.jit(lambda p, x, dp, dx: jax.jvp(cuda, (p, x), (dp, dx)))(
        params, volume, tangent, dvolume
    )
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=2e-4 * float(jnp.max(jnp.abs(b))))

    def grad(f):
        return jax.jit(jax.grad(lambda p, x: jnp.sum(f(p, x) * cotangent), argnums=(0, 1)))(
            params, volume
        )

    actual_grad, expected_grad = grad(cuda), grad(reference)
    coeff = coefficients(params)

    def coefficient_grad(backend):
        return jax.jit(
            jax.grad(lambda c, x: jnp.sum(project_coeff(c, x, backend) * cotangent), argnums=(0, 1))
        )(coeff, volume)

    core_grad, ref_grad = coefficient_grad("cuda"), coefficient_grad("jax")
    # A one-ULP difference in t becomes eps*|source| voxels after a distant
    # ray's source cancellation. CUDA uses a correctly rounded reciprocal;
    # XLA division need not select that same FP32 t, even with fixed rows.
    ray_roundoff = np.finfo(np.float32).eps * float(jnp.max(jnp.abs(coeff[:, :6])))
    for a, b in zip(core_grad, ref_grad, strict=True):
        np.testing.assert_allclose(
            a, b, rtol=0, atol=(2e-4 + ray_roundoff) * float(jnp.max(jnp.abs(b)))
        )
    # Distant-source rotations cancel large source/detector terms. Include
    # their FP32 dot-product roundoff, not just the small final pose gradient.
    jc = jax.jit(jax.jacfwd(coefficients))(params)
    roundoff = (
        4 * np.finfo(np.float32).eps * jnp.einsum("ncjk,nc->jk", jnp.abs(jc), jnp.abs(ref_grad[0]))
    )
    for i, (a, b) in enumerate(zip(actual_grad, expected_grad, strict=True)):
        bound = (2e-4 + ray_roundoff) * float(jnp.max(jnp.abs(b))) + (roundoff if i == 0 else 0)
        assert np.all(np.abs(a - b) <= bound)
    # Freeze the evaluated FP32 geometry for transpose consistency: recompiling
    # pose->coefficient algebra can round distant rays differently.
    core = partial(project_coeff, backend="cuda")
    dc = jax.jvp(coefficients, (params,), (tangent,))[1]
    _, linear = jax.linearize(core, coeff, volume)
    transposed = jax.linear_transpose(linear, dc, dvolume)(cotangent)
    for a, b in zip(transposed, core_grad, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=2e-5 * float(jnp.max(jnp.abs(b))))

    lhs = jnp.sum(linear(dc, dvolume) * cotangent)
    products = (dc * core_grad[0], dvolume * core_grad[1])
    rhs = sum(jnp.sum(x) for x in products)
    scale = sum(jnp.sum(jnp.abs(x)) for x in products)
    assert abs(float(lhs - rhs)) < 2e-5 * float(scale)

    # Forward-mode batching shares the primal volume while mapping directions.
    def directional(d):
        return jax.jvp(lambda c: core(c, volume), (coeff,), (d,))[1]

    mapped = jax.jit(jax.vmap(directional))(jnp.stack([dc, 2 * dc]))
    expected_direction = jax.jit(
        lambda c, d: jax.jvp(lambda q: project_coeff(q, volume, "jax"), (c,), (d,))[1]
    )(coeff, dc)
    np.testing.assert_allclose(
        mapped[0],
        expected_direction,
        rtol=0,
        atol=(2e-4 + ray_roundoff) * float(jnp.max(jnp.abs(expected_direction))),
    )
    np.testing.assert_allclose(mapped[1], 2 * mapped[0], rtol=2e-5, atol=2e-5)
    # Volume-only differentiation must keep the original exact transpose.
    actual = jax.jit(jax.grad(lambda x: jnp.sum(core(coeff, x) * cotangent)))(volume)
    np.testing.assert_allclose(actual, core_grad[1], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(("position", "expected"), [(-1.0, 3.0), (1.0, 0.0), (3.0, 0.0)])
def test_cone_cell_derivative_preserves_constants_and_boundary_support(position, expected):
    from tomojax.core import cone

    grid, detector = Grid(3, 3, 3, 1, 1, 1), Detector(1, 1, 1, 1)
    # The central cone ray crosses exact integer in-plane coordinates.
    coeff = jnp.asarray(
        [[-8, position, 1, 8, position, 1, 0, 1, 0, 0, 0, 1] + [0] * 13], jnp.float32
    )
    direction = jnp.zeros_like(coeff).at[:, 1].set(1).at[:, 4].set(1)
    volume = jnp.ones((3, 3, 3), jnp.float32)

    def evaluate(backend):
        def project(c):
            return cone.cone_project(volume, c, grid, detector, backend=backend)

        _, derivative = jax.jit(lambda c, d: jax.jvp(project, (c,), (d,)))(coeff, direction)
        np.testing.assert_allclose(derivative, expected, atol=1e-6)
        gradient = jax.jit(jax.grad(lambda c: project(c).sum()))(coeff)
        np.testing.assert_allclose(jnp.sum(gradient * direction), expected, atol=1e-6)

    for backend in ("jax", "cuda") if cone.use_cuda_cone() else ("jax",):
        evaluate(backend)


@pytest.mark.gpu
@pytest.mark.parametrize("shared", [False, True])
def test_cone_column_batching_keeps_independent_volumes_and_view_order(shared):
    from tomojax.core import cone

    if not cone.use_cuda_cone():
        pytest.skip("requires CUDA cone kernels")
    grid, detector = Grid(7, 6, 5, 0.7, 1.1, 0.9), Detector(9, 7, 1.3, 0.8)
    geometry = ConeGeometry(grid, detector, [13.7, 78.1, 147.3], ConeBeam(80, 160), tilt_deg=53)
    coeff = cone.cone_coefficients(stack_view_poses(geometry, 3), grid, detector, geometry.beam)
    rng = np.random.default_rng(19)
    volume = jnp.asarray(rng.uniform(size=(2, 5, 6, 7)), jnp.float32)
    directions = jnp.asarray(rng.normal(0, 0.1, (2, 3, 2, 12)), jnp.float32)

    def columns(x, c, d):
        return cone.cone_pose_columns(x, c[None], tangent=d[None], grid=grid, detector=detector)[0]

    def views(x, ds):
        return jax.vmap(columns, in_axes=(None, 0, 0))(x, coeff, ds)

    # Nested maps exercise independently broadcast view and volume dimensions.
    batched = jax.jit(jax.vmap(views, in_axes=(None if shared else 0, 0)))
    actual = batched(volume[0] if shared else volume, directions)
    expected = jnp.stack(
        [
            jnp.stack(
                [columns(volume[0 if shared else b], coeff[v], directions[b, v]) for v in range(3)]
            )
            for b in range(2)
        ]
    )
    np.testing.assert_array_equal(actual, expected)
