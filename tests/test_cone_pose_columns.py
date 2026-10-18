"""CUDA pose columns follow the requested analytic or finite-difference model."""

from __future__ import annotations

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment._geometry.parametrizations import apply_pose_update

# check-public-imports: allow-private
from tomojax.alignment._pose._pose_jacobian import PoseJacobianOptions, pose_prediction_and_columns
from tomojax.core.cone import frame_coefficients, use_cuda_cone
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import (
    ConeBeam,
    ConeGeometry,
    Detector,
    Grid,
    LaminographyGeometry,
    stack_view_poses,
)

pytestmark = pytest.mark.numerical


@pytest.mark.gpu
@pytest.mark.parametrize("width", [5, 6])
@pytest.mark.parametrize(
    ("grid_case", "direction"),
    [
        ("small", (1, 0, 0)),
        ("small", (0.2, 0.7, -0.8)),
        ("large", (0.2, 0.7, -0.8)),
        ("tie", (1, 1, 1)),
    ],
)
def test_cuda_columns_retain_individual_central_differences(
    monkeypatch, width, direction, grid_case
):
    monkeypatch.setenv("TOMOJAX_CUDA_KERNELS", "1")
    if not use_cuda_cone():
        pytest.skip("CUDA cone columns require an NVIDIA GPU and CuPy")
    if grid_case == "small":
        grid = Grid(7, 6, 5, 0.7, 1.1, 0.9, vol_center=(0.23, -0.17, 0.31))
        detector = Detector(9, 7, 1.3, 0.8)
    elif grid_case == "tie":
        # One ray and isotropic spacing make (1,1,1) an EXACT index-axis tie.
        # Odd/even voxel centres give the axes distinct quadrature phases.
        grid = Grid(7, 6, 5, 1.0, 1.0, 1.0)
        detector = Detector(1, 1, 1.3, 0.8)
    else:
        grid = Grid(101, 101, 101, 50 / 101, 53 / 101, 47 / 101, vol_center=(0.23, -0.17, 0.31))
        detector = Detector(192, 243, 0.6, 0.6)
    beam = np.asarray(direction, np.float64)
    beam /= np.linalg.norm(beam)
    u = np.cross(beam, np.eye(3)[np.argmin(np.abs(beam))])
    u /= np.linalg.norm(u)
    v = np.cross(beam, u)
    u, v = np.cos(0.13) * u + np.sin(0.13) * v, -np.sin(0.13) * u + np.cos(0.13) * v
    frame = jnp.asarray(np.stack([40 * beam, u, v, -80 * beam]), jnp.float32)
    nominal = jnp.eye(4)
    params = jnp.zeros(width)
    if direction == (0.2, 0.7, -0.8):
        params = jnp.asarray([0.012, -0.018, 0.031, 0.11, -0.07, 0.04][:width])
    rng = np.random.default_rng(533)
    volume = jnp.asarray(rng.uniform(-0.1, 1.1, (grid.nx, grid.ny, grid.nz)), jnp.float32)
    target = jnp.asarray(rng.normal(size=(detector.nv, detector.nu)), jnp.float32)
    weight = jnp.asarray(rng.uniform(0.05, 0.99, target.shape), jnp.float32)
    if grid_case == "tie":
        row = np.asarray(frame_coefficients(nominal[None], frame, grid, detector))[0]
        ray = row[3:6] - row[:3]
        np.testing.assert_array_equal(np.abs(ray), np.repeat(abs(ray[0]), 3))
    else:
        weight = weight.at[0].set(0)
    options = PoseJacobianOptions(0.001, "object", True, 1, "fp32", "joseph", "central", True)
    candidate = jax.jit(
        partial(
            pose_prediction_and_columns,
            grid=grid,
            detector=detector,
            det_grid=None,
            options=options,
        )
    )

    @jax.jit
    def individual(p, nominal, frame, volume, target, weight):
        def prediction(q):
            pose = apply_pose_update(nominal, q, translation_frame="object")
            image = forward_project_view_T(
                pose, grid, detector, volume, frames=frame, projector_backend="pallas"
            )
            return image.ravel()

        displacement = 0.001 * min(grid.vx, grid.vy, grid.vz)
        radius = 0.5 * math.sqrt(
            (grid.nx * grid.vx) ** 2 + (grid.ny * grid.vy) ** 2 + (grid.nz * grid.vz) ** 2
        )
        steps = jnp.asarray([displacement / radius] * 3 + [displacement] * (width - 3), jnp.float32)
        columns = jax.vmap(
            lambda direction, step: weight.ravel()
            * (prediction(p + step * direction) - prediction(p - step * direction))
            / (2 * step)
        )(jnp.eye(width, dtype=jnp.float32), steps)
        return weight.ravel() * (prediction(p) - target.ravel()), columns

    actual = candidate(params, nominal, frame, volume, target, weight)
    expected = individual(params, nominal, frame, volume, target, weight)
    # Batching may reorder float32 pose arithmetic; differencing at a ~1e-5 step
    # magnifies that rounding, so compare at each column's scale, not bitwise.
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-4 * float(np.max(np.abs(b))))


@pytest.mark.gpu
@pytest.mark.parametrize("width", [5, 6])
@pytest.mark.parametrize("translation_frame", ["object", "detector"])
@pytest.mark.parametrize("direction", [(1, 0.2, 0.1), (0.2, 1, 0.1), (0.1, 0.2, 1)])
def test_cuda_analytic_pose_columns_match_jax(width, translation_frame, direction):
    if not use_cuda_cone():
        pytest.skip("CUDA cone columns require an NVIDIA GPU and CuPy")
    grid = Grid(7, 6, 5, 0.7, 1.1, 0.9, vol_center=(0.23, -0.17, 0.31))
    detector = Detector(9, 7, 1.3, 0.8)
    beam = np.asarray(direction, np.float64)
    beam /= np.linalg.norm(beam)
    u = np.cross(beam, np.eye(3)[np.argmin(np.abs(beam))])
    u /= np.linalg.norm(u)
    v = np.cross(beam, u)
    u, v = np.cos(0.13) * u + np.sin(0.13) * v, -np.sin(0.13) * u + np.cos(0.13) * v
    frame = jnp.asarray(np.stack([40 * beam, u, v, -80 * beam]), jnp.float32)
    nominal = jnp.eye(4).at[:3, 3].set(jnp.asarray([0.11, 0.09, -0.13]))
    params = jnp.asarray([0.013, -0.019, 0.031, 0.11, -0.07, 0.04][:width])
    rng = np.random.default_rng(177)
    volume = jnp.asarray(rng.uniform(-0.1, 1.1, (7, 6, 5)), jnp.float32)
    target = jnp.asarray(rng.normal(size=(7, 9)), jnp.float32)
    weight = jnp.asarray(rng.uniform(0.1, 1, (7, 9)), jnp.float32).at[0].set(0)
    options = PoseJacobianOptions(
        0.001, translation_frame, True, 1, "fp32", "joseph", "autodiff", True
    )
    candidate = jax.jit(
        partial(
            pose_prediction_and_columns,
            grid=grid,
            detector=detector,
            det_grid=None,
            options=options,
        )
    )

    @jax.jit
    def predict(p):
        pose = apply_pose_update(nominal, p, translation_frame=translation_frame)
        image = forward_project_view_T(
            pose, grid, detector, volume, frames=frame, projector_backend="jax"
        )
        return weight.ravel() * (image.ravel() - target.ravel())

    actual = candidate(params, nominal, frame, volume, target, weight)
    expected = predict(params), jax.jit(jax.jacfwd(predict))(params).T
    # These rays avoid interpolation knots and dominant-axis ties, where the
    # Joseph quadrature is not differentiable. Each backend retains FP32 rays.
    for a, b in zip(actual, expected, strict=True):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-4 * float(jnp.max(jnp.abs(b))))


@pytest.mark.parametrize("cone", [False, True])
def test_central_pose_columns_do_not_depend_on_measurement_offsets(cone):
    # For fixed pixel weights, d[W(A(p)x-y)]/dp is independent of y.
    # Subtracting a large measurement before differencing can round away
    # the perturbations, even though the represented predictions distinguish p.
    grid = Grid(7, 6, 5, 0.7, 1.1, 0.9, vol_center=(0.23, -0.17, 0.31))
    detector = Detector(9, 7, 1.3, 0.8, (0.13, -0.19))
    geometry = (
        ConeGeometry(grid, detector, [31.7], ConeBeam(40, 60), tilt_deg=23)
        if cone
        else LaminographyGeometry(grid, detector, [31.7], tilt_deg=23)
    )
    nominal = stack_view_poses(geometry, 1)[0]
    frame = (
        jnp.asarray([[0, 40, 0], [1, 0, 0], [0, 0, 1], [0, -80, 0]], jnp.float32) if cone else None
    )
    rng = np.random.default_rng(687)
    volume = jnp.asarray(rng.uniform(0.2, 1.1, (grid.nx, grid.ny, grid.nz)), jnp.float32)
    params = jnp.asarray([0.012, -0.018, 0.031, 0.11, -0.07, 0.04], jnp.float32)
    weight = jnp.asarray(rng.uniform(0.1, 1, (detector.nv, detector.nu)), jnp.float32).at[0].set(0)
    target = jnp.asarray(rng.normal(size=weight.shape), jnp.float32)
    options = PoseJacobianOptions(0.001, "object", True, 1, "fp32", "joseph", "central", cone)
    evaluate = jax.jit(
        partial(
            pose_prediction_and_columns,
            grid=grid,
            detector=detector,
            det_grid=None,
            options=options,
        )
    )
    residual, columns = evaluate(params, nominal, frame, volume, target, weight)
    assert np.linalg.norm(columns) > 0
    np.testing.assert_array_equal(np.asarray(residual).reshape(weight.shape)[0], 0)
    np.testing.assert_array_equal(np.asarray(columns).reshape(6, *weight.shape)[:, 0], 0)
    for offset in (1.0, 1e5):
        shifted_residual, shifted_columns = evaluate(
            params, nominal, frame, volume, target + offset, weight
        )
        np.testing.assert_array_equal(shifted_columns, columns)
        assert not np.array_equal(shifted_residual, residual)


@pytest.mark.gpu
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("distance", [80.0, 8000.0])
def test_cuda_forward_pose_and_volume_ad_match_reference(packed, distance):
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
    volume = jnp.asarray(rng.uniform(-0.1, 1.1, (7, 6, 5)), jnp.float32)
    params = jnp.asarray(rng.normal(0, 0.02, (3, 6)), jnp.float32)
    tangent = jnp.asarray(rng.normal(0, 0.1, params.shape), jnp.float32)
    dvolume = jnp.asarray(rng.normal(size=volume.shape), jnp.float32)
    cotangent = jnp.asarray(rng.normal(size=(3, 7, 9)), jnp.float32)

    def coefficients(p):
        poses = jax.vmap(lambda t, q: apply_pose_update(t, q, translation_frame="detector"))(
            nominal, p
        )
        return cone.cone_coefficients(poses, grid, detector, geometry.beam)

    def project_coeff(coeff, x, backend):
        if packed and backend == "cuda":
            return cone.prepare_cone_projection(x, grid, detector)(coeff)
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
    volume = jnp.asarray(rng.uniform(size=(2, 7, 6, 5)), jnp.float32)
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
