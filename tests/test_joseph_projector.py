"""Independent physical-ray matrices verify the plane model and its transpose."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# check-public-imports: allow-private
from tomojax.core.joseph import adjoint_jax, forward_jax, plane_coefficients

# check-public-imports: allow-private
from tomojax.core.pallas._pallas_joseph import adjoint_pallas, forward_pallas
from tomojax.geometry import Detector, Grid, grid_volume_origin
from tomojax.recon import CGLSConfig, cgls


def matrix(poses, g, d, interpolation="linear"):
    # Scalar FP64 cubic polynomial is independent of the JAX/Triton weight code.
    def cubic(t):
        r = abs(t)
        if r <= 1:
            return (3 * r**3 - 5 * r**2 + 2) / 2
        if r < 2:
            return (-(r**3) + 5 * r**2 - 8 * r + 4) / 2
        return 0.0

    offsets = [0, 1] if interpolation == "linear" else [-1, 0, 1, 2]
    shape = (g.nx, g.ny, g.nz)
    voxel = np.array([g.vx, g.vy, g.vz])
    origin = np.array(grid_volume_origin(g))
    out = np.zeros((len(poses) * d.nv * d.nu, np.prod(shape)))
    for vi, t in enumerate(poses.astype(np.float64)):
        direction = t[1, :3]
        axis = np.argmax(np.abs(direction / voxel))
        b = (axis + 1) % 3
        c = (axis + 2) % 3
        for iv in range(d.nv):
            for iu in range(d.nu):
                camera = np.array(
                    [
                        (iu - (d.nu - 1) / 2) * d.du + d.det_center[0],
                        0,
                        (iv - (d.nv - 1) / 2) * d.dv + d.det_center[1],
                    ]
                )
                base = t[:3, :3].T @ (camera - t[:3, 3])
                row = (vi * d.nv + iv) * d.nu + iu
                for k in range(shape[axis]):
                    raytime = (origin[axis] + k * voxel[axis] - base[axis]) / direction[axis]
                    q = (base + raytime * direction - origin) / voxel
                    ib = int(np.floor(q[b]))
                    ic = int(np.floor(q[c]))
                    wb = q[b] - ib
                    wc = q[c] - ic
                    for db in offsets:
                        for dc in offsets:
                            idx = [0, 0, 0]
                            idx[axis] = k
                            idx[b] = ib + db
                            idx[c] = ic + dc
                            if all(0 <= i < n for i, n in zip(idx, shape, strict=False)):
                                out[row, np.ravel_multi_index(idx, shape)] += (
                                    (
                                        (wb if db else 1 - wb)
                                        if interpolation == "linear"
                                        else cubic(wb - db)
                                    )
                                    * (
                                        (wc if dc else 1 - wc)
                                        if interpolation == "linear"
                                        else cubic(wc - dc)
                                    )
                                    * voxel[axis]
                                    / abs(direction[axis])
                                )
    return out


def problem(voxel=(0.8, 1.1, 1.3), spacing=(0.7, 1.2)):
    grid = Grid(5, 3, 4, *voxel, vol_origin=(-1.7, -0.9, -1.1))
    detector = Detector(9, 7, *spacing, (0.17, -0.23))
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (6, 4, 4)).copy()
    poses[:, :3, :3] = Rotation.from_euler(
        "xyz",
        [[0, 0, 0], [0, 0, 90], [90, 0, 0], [20, 30, 17], [0, 0, 45], [80, 130, -10]],
        degrees=True,
    ).as_matrix()
    poses[:, :3, 3] = [0.13, -0.17, 0.29]
    return grid, detector, poses


@pytest.mark.parametrize(
    ("voxel", "spacing"),
    [((0.8, 1.1, 1.3), (0.7, 1.2)), ((2.0, 0.4, 1.3), (0.3, 0.5)), ((0.5, 0.6, 0.7), (1.4, 1.8))],
)
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_plane_operators_match_independent_physical_ray_matrix(
    voxel, spacing, backend, interpolation
):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, poses = problem(voxel, spacing)
    expected = matrix(poses, grid, detector, interpolation)
    rng = np.random.default_rng(37)
    volume = jnp.asarray(rng.normal(size=(5, 3, 4)), jnp.float32)
    images = jnp.asarray(rng.normal(size=(6, 7, 9)), jnp.float32)
    coefficients = plane_coefficients(jnp.asarray(poses), grid, detector)
    fp, bp = (forward_pallas, adjoint_pallas) if backend == "pallas" else (forward_jax, adjoint_jax)
    forward = jax.jit(lambda x: fp(coefficients, x, grid, detector, interpolation=interpolation))
    adjoint = jax.jit(lambda y: bp(coefficients, y, grid, detector, interpolation=interpolation))
    projected = np.asarray(forward(volume))
    backprojected = np.asarray(adjoint(images))
    for actual, reference in [
        (projected.ravel(), expected @ np.asarray(volume).ravel()),
        (backprojected.ravel(), expected.T @ np.asarray(images).ravel()),
    ]:
        assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < 8e-6
    assert abs(np.vdot(projected, images) - np.vdot(volume, backprojected)) < 2e-6 * np.linalg.norm(
        projected
    ) * np.linalg.norm(images)
    if backend == "pallas":
        # The gather transpose has no floating-point atomics or scheduling-dependent sum.
        np.testing.assert_array_equal(adjoint(images), backprojected)


@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_plane_cgls_matches_independent_damped_system_with_tail_batch(
    warm_start, backend, interpolation
):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, poses = problem()
    reference = matrix(poses, grid, detector, interpolation)

    class PosedGeometry:
        def pose_for_view(self, index):
            return poses[index]

    rng = np.random.default_rng(14)
    data = rng.normal(size=(6, 7, 9)).astype(np.float32)
    initial = rng.normal(size=(5, 3, 4)).astype(np.float32) if warm_start else None
    damping = 0.3
    expected = np.linalg.lstsq(
        np.concatenate([reference, damping * np.eye(60)]),
        np.concatenate([data.ravel(), np.zeros(60)]),
        rcond=None,
    )[0]
    volume, info = cgls(
        PosedGeometry(),
        grid,
        detector,
        data,
        init_x=initial,
        config=CGLSConfig(
            iterations=120,
            rtol=1e-7,
            damping=damping,
            views_per_batch=4,
            projector_backend=backend,
            projector_model="joseph",
            joseph_interpolation=interpolation,
        ),
    )
    np.testing.assert_allclose(np.asarray(volume).ravel(), expected, atol=5e-5, rtol=3e-4)
    assert info["projector_model"] == "joseph"
    assert info["joseph_interpolation"] == interpolation
    gradient = (
        reference.T @ (data.ravel() - reference @ np.asarray(volume).ravel())
        - damping**2 * np.asarray(volume).ravel()
    )
    assert np.linalg.norm(gradient) < 5e-6 * np.linalg.norm(reference.T @ data.ravel()), info


def test_plane_model_rejects_custom_detector_coordinates():
    from tomojax.geometry import ParallelGeometry

    grid, detector, _ = problem()
    geometry = ParallelGeometry(grid, detector, np.arange(6))
    with pytest.raises(ValueError, match="canonical detector"):
        cgls(
            geometry,
            grid,
            detector,
            jnp.zeros((6, 7, 9)),
            config=CGLSConfig(projector_model="joseph"),
            det_grid=(jnp.zeros(63), jnp.zeros(63)),
        )


@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_changing_poses_switches_plane_axes_without_recompiling(backend):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, poses = problem()
    volume = jnp.asarray(np.random.default_rng(39).normal(size=(5, 3, 4)), jnp.float32)
    projector = forward_pallas if backend == "pallas" else forward_jax
    project = jax.jit(
        lambda t, x: projector(plane_coefficients(t, grid, detector), x, grid, detector)
    )
    outputs = []
    for turn in (0.0, 90.0):
        changed = poses.copy()
        changed[:, :3, :3] = (
            Rotation.from_euler("z", turn, degrees=True).as_matrix() @ poses[:, :3, :3]
        )
        changed[:, :3, 3] += turn * 0.002
        expected = matrix(changed, grid, detector) @ np.asarray(volume).ravel()
        result = np.asarray(project(jnp.asarray(changed), volume)).ravel()
        assert np.linalg.norm(result - expected) / np.linalg.norm(expected) < 8e-6
        outputs.append(result)
    assert project._cache_size() == 1
    assert not np.allclose(*outputs)


@pytest.mark.gpu
@pytest.mark.parametrize("spacing", [(0.43, 0.72), (1.71, 2.19)])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_long_oblique_planes_and_gather_bounds_agree_with_autodiff(spacing, interpolation):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid = Grid(129, 127, 5, 0.8, 1.1, 1.3, vol_center=(0.3, -0.6, 0.2))
    detector = Detector(151, 17, *spacing, (0.31, -0.43))
    _, _, poses = problem()
    coeff = plane_coefficients(jnp.asarray(poses), grid, detector)
    rng = np.random.default_rng(400)
    volume = jnp.asarray(rng.normal(size=(129, 127, 5)), jnp.float32)
    images = jnp.asarray(rng.normal(size=(6, 17, 151)), jnp.float32)
    reference_forward = jax.jit(
        lambda x: forward_jax(coeff, x, grid, detector, interpolation=interpolation)
    )
    reference_adjoint = jax.jit(
        lambda y: adjoint_jax(coeff, y, grid, detector, interpolation=interpolation)
    )
    forward = jax.jit(
        lambda x: forward_pallas(coeff, x, grid, detector, interpolation=interpolation)
    )
    adjoint = jax.jit(
        lambda y: adjoint_pallas(coeff, y, grid, detector, interpolation=interpolation)
    )
    for actual, expected in [
        (forward(volume), reference_forward(volume)),
        (adjoint(images), reference_adjoint(images)),
    ]:
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 3e-5
    dot_error = abs(np.vdot(forward(volume), images) - np.vdot(volume, adjoint(images)))
    assert dot_error < 2e-6 * np.linalg.norm(forward(volume)) * np.linalg.norm(images)


@pytest.mark.parametrize("bad", ["origin", "center", "singular_pose", "scaled_pose", "bottom_row"])
def test_joseph_rejects_invalid_geometry_before_kernel_execution(bad):
    from dataclasses import replace

    grid, detector, poses = problem()
    if bad == "origin":
        grid = replace(grid, vol_origin=(np.nan, 0, 0))
    if bad == "center":
        detector = replace(detector, det_center=(np.inf, 0))
    if bad == "singular_pose":
        poses[0, :3, :3] = 0
    if bad == "scaled_pose":
        poses[0, :3, :3] *= 2
    if bad == "bottom_row":
        poses[0, 3, 0] = 1

    class PosedGeometry:
        def pose_for_view(self, i):
            return poses[i]

    with pytest.raises(ValueError, match="finite|rigid homogeneous"):
        cgls(
            PosedGeometry(),
            grid,
            detector,
            jnp.ones((6, 7, 9)),
            config=CGLSConfig(projector_model="joseph", iterations=1),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("reverse_rows", [False, True])
@pytest.mark.parametrize("row_scale", [1.0, 0.75, 1.25])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("absolute_weights", [False, True])
def test_separable_row_adjoint_matches_physical_matrix_and_dynamic_tilts(
    reverse_rows, row_scale, interpolation, absolute_weights
):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    g, d, poses = problem(spacing=(0.47, 1.3 * row_scale))
    poses[:, :3, :3] = Rotation.from_euler(
        "z", np.array([0, 90, 31, 88, 145, 179])[:, None], degrees=True
    ).as_matrix()
    if reverse_rows:
        poses[:, :3, :3] = poses[:, :3, :3] @ np.diag([1, -1, -1])
    rng = np.random.default_rng(94)
    images = jnp.asarray(rng.normal(size=(6, d.nv, d.nu)), jnp.float32)
    bp = jax.jit(
        lambda t, y: adjoint_pallas(
            plane_coefficients(t, g, d),
            y,
            g,
            d,
            interpolation=interpolation,
            absolute_weights=absolute_weights,
        )
    )
    changed_poses = []
    for tilt in (0.0, 0.001, 17.0):
        changed = poses.copy()
        changed[:, :3, :3] = (
            Rotation.from_euler("y", tilt, degrees=True).as_matrix() @ poses[:, :3, :3]
        )
        changed_poses.append(changed)
        operator = matrix(changed, g, d, interpolation)
        if absolute_weights:
            operator = np.abs(operator)
        expected = operator.T @ np.asarray(images).ravel()
        actual = bp(jnp.asarray(changed), images)
        np.testing.assert_allclose(np.asarray(actual).ravel(), expected, rtol=2e-5, atol=8e-6)
    assert bp._cache_size() == 1
    # A batched transpose can choose a different row path for each pose stack.
    actual_batch = jax.jit(jax.vmap(bp, in_axes=(0, None)))(jnp.asarray(changed_poses), images)
    for actual, poses_i in zip(actual_batch, changed_poses, strict=True):
        assert np.linalg.norm(actual - bp(jnp.asarray(poses_i), images)) < 1e-5


@pytest.mark.gpu
@pytest.mark.parametrize("row_scale", [1.0, 1 - 1e-4, 1 + 1e-4])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_long_separable_rows_preserve_small_spacing_changes_and_tilts(row_scale, interpolation):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    g = Grid(3, 4, 257, 0.8, 1.1, 1.3, vol_center=(0.37, -0.51, 0.13))
    d = Detector(7, 263, 0.47, g.vz * row_scale, (0.17, -0.23))
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (3, 4, 4)).copy()
    poses[:, :3, :3] = Rotation.from_euler(
        "z", np.array([0, 90, 31])[:, None], degrees=True
    ).as_matrix()
    images = jnp.asarray(np.random.default_rng(115).normal(size=(3, d.nv, d.nu)), jnp.float32)
    candidate = jax.jit(
        lambda t: adjoint_pallas(
            plane_coefficients(t, g, d), images, g, d, interpolation=interpolation
        )
    )
    reference = jax.jit(
        lambda t: adjoint_jax(
            plane_coefficients(t, g, d), images, g, d, interpolation=interpolation
        )
    )
    for tilt in (0.0, 0.001):
        changed = poses.copy()
        changed[:, :3, :3] = (
            Rotation.from_euler("y", tilt, degrees=True).as_matrix() @ poses[:, :3, :3]
        )
        actual, expected = candidate(jnp.asarray(changed)), reference(jnp.asarray(changed))
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 2e-6


@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_cubic_cancellation_bound_uses_magnitudes_of_physical_weights(backend):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    # check-public-imports: allow-private
    from tomojax.recon.cgls import _operators

    g, d, poses = problem()
    a = matrix(poses, g, d, "cubic")
    rng = np.random.default_rng(587)
    data = np.exp(rng.normal(size=(len(poses), d.nv, d.nu))).astype(np.float32)
    _, absolute_adjoint = _operators(
        jnp.asarray(poses), g, d, None, backend, 4, "joseph", "cubic", absolute_weights=True
    )
    actual = np.asarray(jax.jit(absolute_adjoint)(jnp.asarray(data))).ravel()
    expected = np.abs(a).T @ data.ravel()
    np.testing.assert_allclose(actual, expected, rtol=8e-6, atol=8e-6)
    assert np.all(actual >= 0)
    # Signed cubic weights cancel and underestimate the bound even for positive
    # data. This difference matters for componentwise FP32 stagnation checks.
    assert np.linalg.norm(a.T @ data.ravel() - expected) > 0.01 * np.linalg.norm(expected)


@pytest.mark.gpu
@pytest.mark.parametrize("nz", [4, 5, 7, 13])
@pytest.mark.parametrize(
    ("voxel", "spacing"), [((0.8, 1.1, 1.3), (0.7, 1.2)), ((2.0, 0.4, 1.3), (0.3, 0.5))]
)
def test_cuda_gather_matches_reference_transpose_and_accumulates(nz, voxel, spacing):
    # check-public-imports: allow-private
    from tomojax.core._cuda_joseph import cuda_gather_available, gather_transpose_cuda

    if not cuda_gather_available():
        pytest.skip("requires CUDA and CuPy")
    g, d, poses = problem(voxel, spacing)
    g = Grid(g.nx, g.ny, nz, *voxel, vol_origin=(-1.7, -0.9, -1.1))
    coeff = plane_coefficients(jnp.asarray(poses), g, d)
    rng = np.random.default_rng(nz)
    images = jnp.asarray(rng.normal(size=(len(poses), d.nv, d.nu)), jnp.float32)
    start = jnp.asarray(rng.normal(size=(g.nx, g.ny, g.nz)), jnp.float32)
    expected = adjoint_jax(coeff, images, g, d)
    swapped = jnp.transpose(images, (0, 2, 1))
    actual = jax.jit(lambda c, y, x: gather_transpose_cuda(c, y, g, d, x))(coeff, swapped, start)
    np.testing.assert_allclose(actual - start, expected, rtol=2e-5, atol=2e-5)
    # The forward projection and this gather are matched transposes.
    volume = jnp.asarray(rng.normal(size=(g.nx, g.ny, g.nz)), jnp.float32)
    lhs = float(jnp.vdot(forward_pallas(coeff, volume, g, d), images))
    rhs = float(jnp.vdot(volume, gather_transpose_cuda(coeff, swapped, g, d)))
    assert abs(lhs - rhs) <= 1e-5 * max(abs(lhs), 1.0)
