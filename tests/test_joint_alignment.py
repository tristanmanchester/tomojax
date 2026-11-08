"""The joint volume and per-view pose fit behind automatic tj.align."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.alignment._joint import (  # check-public-imports: allow-private
    EdgeFilter,
    MotionPrior,
    fit_poses,
)
from tomojax.alignment._joint._fit import _gauge  # check-public-imports: allow-private
from tomojax.alignment._joint._operators import (  # check-public-imports: allow-private
    Operators,
    posed,
)
from tomojax.core.joseph import forward_project_planes, plane_coefficients, plane_weighted_diagonal
from tomojax.geometry import stack_view_poses


def _laminography(*, nx=5, ny=4, nz=3, nu=7, nv=6, views=4, tilt=-30.0):
    grid = tj.Grid(nx, ny, nz, 1.0, 1.1, 0.9)
    detector = tj.Detector(nu, nv, 1.0, 1.0)
    angles = np.linspace(0, 360, views, endpoint=False)
    geometry = tj.LaminographyGeometry(grid, detector, angles, tilt_deg=tilt)
    return grid, detector, geometry, angles


@pytest.mark.numerical
def test_weighted_diagonal_is_the_explicit_matrix_diagonal():
    grid, detector, geometry, angles = _laminography()
    poses = jnp.asarray(stack_view_poses(geometry, len(angles)))
    cf = plane_coefficients(poses, grid, detector)
    rng = np.random.default_rng(3)
    sw = jnp.asarray(rng.uniform(0.2, 1.5, (len(angles), detector.nv, detector.nu)), jnp.float32)
    columns = []
    for k in range(grid.nx * grid.ny * grid.nz):
        unit = jnp.zeros(grid.nx * grid.ny * grid.nz, jnp.float32).at[k].set(1.0)
        volume = unit.reshape(grid.nz, grid.ny, grid.nx)
        columns.append(forward_project_planes(cf, volume, grid, detector, backend="jax"))
    matrix = jnp.stack(columns, -1)
    expected = jnp.sum((sw[..., None] * matrix) ** 2, axis=(0, 1, 2)).reshape(
        grid.nz, grid.ny, grid.nx
    )
    zeros = jnp.zeros((grid.nz, grid.ny, grid.nx), jnp.float32)
    got = plane_weighted_diagonal(cf, sw, zeros, grid, detector, backend="jax")
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_weighted_diagonal_gather_matches_the_scatter(dtype):
    grid, detector, geometry, angles = _laminography(nx=19, ny=17, nz=13, nu=23, nv=21, views=7)
    cf = plane_coefficients(jnp.asarray(stack_view_poses(geometry, len(angles))), grid, detector)
    sw = jnp.asarray(np.random.default_rng(4).uniform(0, 1, (len(angles), 21, 23)), jnp.float32)
    start = jnp.full((grid.nz, grid.ny, grid.nx), 0.25, jnp.float32)
    with jax.enable_x64():
        sw, start = sw.astype(dtype), start.astype(dtype)
        cuda = plane_weighted_diagonal(cf, sw, start, grid, detector, backend="cuda")
        scatter = plane_weighted_diagonal(cf, sw, start, grid, detector, backend="jax")
    assert cuda.dtype == jnp.float32
    np.testing.assert_allclose(cuda, scatter, rtol=1e-4, atol=1e-6)


def _dense_constrained(h, constraints, rhs):
    k = constraints.shape[1]
    kkt = np.block([[h, constraints], [constraints.T, np.zeros((k, k))]])
    return np.linalg.solve(kkt, np.concatenate([rhs, np.zeros(k)]))[: len(rhs)]


@pytest.mark.parametrize("dofs", [2, 5])
def test_global_pose_solves_match_dense_constrained_systems(dofs):
    views, k = 37, dofs
    rng = np.random.default_rng(5)
    gauge = rng.normal(size=(k * views, 6))
    with jax.enable_x64(True):
        prior = MotionPrior(views, gauge, knot_spacing=(10,) * k, delta=0.02)
        a = rng.normal(size=(views, k, k))
        blocks = np.einsum("nij,nkj->nik", a, a) + 0.5 * np.eye(k)
        w = rng.uniform(0.1, 2.0, views - 1)
        rhs = rng.normal(size=k * views)
        f = np.zeros((k * views, k * views))
        for i in range(views):
            f[k * i : k * i + k, k * i : k * i + k] = blocks[i]
        diff = np.kron(np.diff(np.eye(views), axis=0), np.eye(k))
        laplacian = diff.T @ np.diag(np.repeat(w, k)) @ diff
        u, g, c = (np.asarray(m) for m in (prior.U, prior.G, prior.C))
        q = np.eye(k * views) - u @ u.T
        expected = _dense_constrained(f + q @ laplacian @ q, g, rhs)
        got = prior.solve(prior.factor(jnp.asarray(blocks), jnp.asarray(w)), jnp.asarray(rhs))
        np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-9)
        expected = _dense_constrained(f + laplacian, c, rhs)
        got = prior.innovation_solve(jnp.asarray(blocks), jnp.asarray(w), jnp.asarray(rhs))
        np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-9)


def test_edge_filter_transpose_is_exact_with_masks_and_odd_sizes():
    rng = np.random.default_rng(6)
    shape = (3, 13, 17)
    precision = rng.uniform(0, 1, shape).astype(np.float32)
    precision[precision < 0.2] = 0
    precision[1, 4] = 0  # an unobserved row
    a, z = (jnp.asarray(rng.normal(size=shape), jnp.float32) for _ in range(2))
    h = EdgeFilter((0.5, 0.75))
    left = float(jnp.vdot(h.forward(a, precision), z))
    right = float(jnp.vdot(a, h.transpose(z, precision)))
    assert abs(left - right) <= 2e-5 * max(abs(left), 1.0)


@pytest.mark.numerical
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_parallel_pose_columns_match_finite_differences_at_nonzero_poses(interpolation):
    grid, detector, geometry, angles = _laminography(
        nx=9, ny=8, nz=7, nu=13, nv=11, views=3, tilt=-37.0
    )
    nominal = np.array(stack_view_poses(geometry, len(angles)))
    nominal[:, :3, 3] = [[0.27, 0.3, -0.42], [-0.39, 0.7, 0.11], [0.17, -0.2, 0.31]]
    rng = np.random.default_rng(118)
    volume = jnp.asarray(rng.uniform(0.1, 1.0, (grid.nz, grid.ny, grid.nx)), jnp.float32)
    with jax.enable_x64(True):
        ops = Operators(
            grid,
            detector,
            nominal,
            lever=5.0,
            native=(0.7, 0.9),
            batch_size=3,
            interpolation=interpolation,
        )
        q = jnp.asarray(rng.uniform(-0.2, 0.2, (len(angles), 5)), jnp.float32)
        raw = ops.columns(ops.nominal, q, ops.valid, volume)
        for k in range(5):
            direction = np.zeros((len(angles), 5), np.float32)
            direction[:, k] = 0.001
            plus = ops.project(ops.coefficients(ops.nominal, q + direction, ops.valid), volume)
            minus = ops.project(ops.coefficients(ops.nominal, q - direction, ops.valid), volume)
            difference = (np.asarray(plus) - np.asarray(minus)) / 0.002
            exact = np.asarray(raw[:, 1 + k])
            relative = np.linalg.norm(exact - difference) / np.linalg.norm(exact)
            assert relative < 0.02, (k, relative)


@pytest.mark.numerical
@pytest.mark.parametrize("column_dtype", [jnp.float16, jnp.float32])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_parallel_image_pose_cross_blocks_use_the_matched_transpose(column_dtype, interpolation):
    grid, detector, geometry, angles = _laminography()
    nominal = np.asarray(stack_view_poses(geometry, len(angles)))
    rng = np.random.default_rng(119)
    with jax.enable_x64(True):
        ops = Operators(
            grid,
            detector,
            nominal,
            lever=4.0,
            native=(0.7, 0.9),
            batch_size=3,
            interpolation=interpolation,
        )
        q = ops.pad_poses(rng.normal(0, 0.1, (len(angles), 5)))
        volume = jnp.asarray(rng.normal(size=ops.shape), jnp.float32)
        weights = rng.uniform(0, 1, (len(angles), detector.nv, detector.nu))
        weights[weights < 0.2] = 0
        sw = ops.pad_images(weights)
        # Compressed and full-precision columns must have the same transpose semantics.
        columns = jnp.asarray(
            rng.normal(size=(ops.padded_n, 5, detector.nv, detector.nu)), column_dtype
        )
        p = jnp.asarray(rng.normal(size=len(angles) * 5), jnp.float64)
        _, cross = ops.weighted_forward(q, volume, sw, columns)
        back = ops.pose_back(q, sw, columns, p)
        left = float(jnp.vdot(p, cross))
        right = float(jnp.vdot(volume.astype(jnp.float64), back.astype(jnp.float64)))
        assert abs(left - right) < 2e-5 * max(abs(left), abs(right), 1)


@pytest.mark.numerical
def test_pose_normal_blocks_use_the_same_columns_as_the_schur_operator():
    """A different Gram precision can make the nominally SPD Schur system indefinite."""
    grid, detector, geometry, angles = _laminography()
    nominal = np.asarray(stack_view_poses(geometry, len(angles)))
    rng = np.random.default_rng(122)
    with jax.enable_x64(True):
        ops = Operators(grid, detector, nominal, lever=4.0, native=(0.7, 0.9), batch_size=3)
        q = ops.pad_poses(rng.normal(0, 0.1, (len(angles), 5)))
        volume = jnp.asarray(rng.uniform(0.1, 1.0, ops.shape), jnp.float32)
        measured = ops.pad_images(rng.normal(size=(len(angles), detector.nv, detector.nu)))
        precision = rng.uniform(0.1, 1.0, measured.shape).astype(np.float32)
        precision[ops.n :] = 0
        _, _, sw, cached, score, blocks, _, _ = ops.linearize(
            q, volume, measured, jnp.asarray(precision), 100.0
        )
        prediction = ops.project(ops.coefficients(ops.nominal, q, ops.valid), volume)
        residual = np.asarray(sw * (prediction - measured), np.float64)[: ops.n]
        columns = np.asarray(cached, np.float64)[: ops.n]
        expected_score = np.einsum("nkvu,nvu->nk", columns, residual).ravel()
        expected_blocks = np.einsum("nkvu,nlvu->nkl", columns, columns)
        # The fused column kernel and separate projection can differ by a few
        # FP32 ulps; compare the score norm, not cancellation-prone entries.
        score_error = np.linalg.norm(np.asarray(score) - expected_score)
        relative_score_error = score_error / np.linalg.norm(expected_score)
        assert relative_score_error < 2e-6, relative_score_error
        np.testing.assert_allclose(blocks, expected_blocks, rtol=2e-7, atol=1e-8)


def test_rigid_gauge_reproduces_one_shared_object_transform():
    from scipy.linalg import expm
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(120)
    nominal = np.tile(np.eye(4), (7, 1, 1))
    nominal[:, :3, :3] = Rotation.random(7, random_state=rng).as_matrix()
    nominal[:, :3, 3] = rng.normal(size=(7, 3))
    lever, native = 11.0, (0.6, 1.1)
    epsilon = 1e-5
    with jax.enable_x64(True):
        columns = _gauge(nominal, lever=lever, native=native).reshape(7, 5, 6)
        for axis in range(6):
            generator = np.zeros((4, 4))
            if axis < 3:
                vector = np.eye(3)[axis]
                generator[:3, :3] = np.array(
                    [
                        [0, -vector[2], vector[1]],
                        [vector[2], 0, -vector[0]],
                        [-vector[1], vector[0], 0],
                    ]
                )
            else:
                generator[axis - 3, 3] = 1
            reference = (
                nominal @ expm(-epsilon * generator) - nominal @ expm(epsilon * generator)
            ) / (2 * epsilon)
            positive = posed(jnp.asarray(nominal), epsilon * columns[:, :, axis], lever, native)
            negative = posed(jnp.asarray(nominal), -epsilon * columns[:, :, axis], lever, native)
            actual = np.asarray((positive - negative) / (2 * epsilon))
            np.testing.assert_allclose(actual[:, :3, :3], reference[:, :3, :3], atol=1e-8)
            np.testing.assert_allclose(actual[:, [0, 2], 3], reference[:, [0, 2], 3], atol=1e-8)


def test_pose_composition_preserves_geometry_with_reduced_default_matmul_precision():
    """CUDA must not quantize nominal poses through TF32 or bfloat16 products."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(121)
    nominal = np.tile(np.eye(4), (11, 1, 1)).astype(np.float32)
    nominal[:, :3, :3] = Rotation.random(11, random_state=rng).as_matrix()
    nominal[:, :3, 3] = rng.normal(size=(11, 3))
    lever, native = 64.0, (0.7, 0.9)
    calculate = jax.jit(lambda n, q: posed(n, q, lever, native))
    with jax.default_matmul_precision("bfloat16"):
        zero = np.asarray(calculate(jnp.asarray(nominal), jnp.zeros((11, 5), jnp.float32)))
        np.testing.assert_array_equal(zero, nominal)
        q = rng.normal(0, 0.7, (11, 5)).astype(np.float32)
        actual = np.asarray(calculate(jnp.asarray(nominal), jnp.asarray(q)))
    expected = nominal.astype(np.float64).copy()
    correction = Rotation.from_euler("YXZ", q[:, [1, 0, 2]].astype(np.float64) / lever)
    expected[:, :3, :3] = expected[:, :3, :3] @ correction.as_matrix()
    expected[:, 0, 3] += native[0] * q[:, 3]
    expected[:, 2, 3] += native[1] * q[:, 4]
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("frame", ["detector", "object"])
@pytest.mark.parametrize("tilt", [0.0, -31.0])
def test_public_joint_alignment_composes_saved_nested_poses(monkeypatch, tmp_path, frame, tilt):
    """The solver fits an increment; the public result stores complete physical poses."""
    from scipy.spatial.transform import Rotation

    from tomojax import _smooth_alignment
    from tomojax._data.geometry_meta import (  # check-public-imports: allow-private
        AugmentedGeometry,
        DetectorRollGeometry,
    )

    grid, detector, _, angles = _laminography(views=5)
    geometry = (
        tj.ParallelGeometry(grid, detector, angles)
        if tilt == 0
        else tj.LaminographyGeometry(grid, detector, angles, tilt_deg=tilt)
    )
    calibrated = DetectorRollGeometry(geometry, 0.3)
    rng = np.random.default_rng(123)
    inner, outer, increment = (
        rng.normal(size=(len(angles), 6)) * [0.04, 0.03, 0.02, 0.3, 0.2, 0.1] for _ in range(3)
    )
    increment[:, 5] = 0  # parallel rays cannot fit translation along their beam
    nested = AugmentedGeometry(AugmentedGeometry(calibrated, inner, frame), outer, "detector")
    scan = tj.Scan(np.ones((len(angles), detector.nv, detector.nu), np.float32), nested)
    tj.save(tmp_path / "input.npz", scan)
    scan = tj.load(tmp_path / "input.npz")
    before = np.asarray(stack_view_poses(scan.geometry, len(angles)), np.float64)
    expected = before.copy()
    expected[:, :3, :3] = (
        before[:, :3, :3] @ Rotation.from_euler("YXZ", increment[:, [1, 0, 2]]).as_matrix()
    )
    expected[:, :3, 3] += increment[:, [3, 5, 4]]
    record = {
        "fit": ["translations", "rotations"],
        "corrections": {},
        "detector_binning": 1,
        "levels": [{"status": "budget_exhausted"}],
        "refinement": {"status": "converged"},
    }

    def fit(
        projections,
        nominal,
        working,
        current_detector,
        *,
        fit,
        keep_grid,
        native_pixel_size,
        ray_binning,
        frames,
        detector_roll_deg,
    ):
        assert fit == ("translations", "rotations")
        assert frames is None and detector_roll_deg == 0.3
        assert projections is scan.projections
        assert working == grid and current_detector == detector and keep_grid is False
        assert native_pixel_size == (detector.du, detector.dv) and ray_binning == (1, 1)
        np.testing.assert_allclose(nominal, before, atol=5e-7)
        return np.zeros((grid.nz, grid.ny, grid.nx), np.float32), grid, increment, record

    monkeypatch.setattr(_smooth_alignment, "fit_poses", fit)
    aligned = tj.align(scan, fit=("translations", "rotations"))
    assert aligned.scan.projections is scan.projections
    assert aligned.scan.detector == detector and aligned.scan.grid == grid
    assert aligned.info == {**record, "method": "joint_volume_pose", "calibration": {}}
    np.testing.assert_allclose(
        stack_view_poses(aligned.scan.geometry, len(angles)), expected, atol=2e-6
    )
    np.testing.assert_allclose(aligned.poses, aligned.scan.poses, atol=1e-7)
    tj.save(tmp_path / "aligned.npz", aligned)
    loaded = tj.load(tmp_path / "aligned.npz")
    np.testing.assert_allclose(stack_view_poses(loaded.geometry, len(angles)), expected, atol=2e-6)


def _moving_scan(slip=0.0, wobble_deg=0.0):
    grid = tj.Grid(28, 28, 20, 1.0, 1.0, 1.0)
    detector = tj.Detector(40, 32, 1.0, 1.0)
    views = 90
    angles = np.linspace(0, 360, views, endpoint=False)
    geometry = tj.LaminographyGeometry(grid, detector, angles, tilt_deg=-30)
    z, y, x = np.mgrid[:20, :28, :28]
    volume = np.zeros((20, 28, 28), np.float32)
    for cz, cy, cx, r, mu in ((10, 10, 9, 5, 1.0), (8, 17, 18, 4, 0.6), (13, 19, 8, 3, 1.4)):
        volume[(z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= r * r] = mu
    table = np.zeros((views, 6))
    table[45:, 3] = slip
    phase = np.deg2rad(angles)
    table[:, 0] = np.deg2rad(wobble_deg) * np.sin(3 * phase)
    table[:, 1] = np.deg2rad(wobble_deg) * np.cos(2 * phase + 0.5)
    empty = tj.Scan(np.zeros((views, 32, 40), np.float32), geometry)
    projections = np.array(tj.project(empty.with_poses(table).geometry, volume))
    projections += np.random.default_rng(7).normal(0, 0.01, projections.shape).astype(np.float32)
    truth = np.asarray(stack_view_poses(empty.with_poses(table).geometry, views))
    return projections, np.asarray(stack_view_poses(geometry, views)), grid, detector, truth


def test_backend_policy_respects_cpu_platform_context():
    from tomojax.core.backend_policy import cuda_backend

    capability = cuda_backend(jax.default_backend())
    with jax.default_device("cpu"):
        assert not cuda_backend()
        assert cuda_backend(jax.default_backend()) == capability


def _landmark_error(estimated, truth, grid):
    """RMS detector-pixel error of eight landmarks after one common rigid motion."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    marks = np.stack(np.meshgrid(*[[-0.3, 0.3]] * 3, indexing="ij"), -1).reshape(-1, 3)
    marks = marks * np.array([grid.nx, grid.ny, grid.nz])

    def pixels(m, p):
        return (np.einsum("nij,kj->nki", m[:, :3, :3], p) + m[:, None, :3, 3])[..., [0, 2]]

    target = pixels(truth, marks)

    def residual(par):
        moved = marks @ Rotation.from_rotvec(par[:3]).as_matrix().T + par[3:]
        return (pixels(estimated, moved) - target).ravel()

    fit = least_squares(residual, np.zeros(6))
    return float(np.sqrt(np.mean(residual(fit.x).reshape(-1, 2) ** 2) * 2))


@pytest.mark.numerical
@pytest.mark.slow
def test_pose_fit_recovers_a_slip_and_keeps_a_clean_scan_still():
    projections, nominal, grid, detector, _ = _moving_scan(slip=1.5)
    _, _, table, record = fit_poses(projections, nominal, grid, detector)
    jumps = np.abs(np.diff(table[:, 3]))
    assert abs(jumps[44] - 1.5) < 0.15, record["levels"][-1]["status"]
    assert np.median(np.delete(jumps, 44)) < 0.05
    projections, nominal, grid, detector, _ = _moving_scan()
    _, _, table, record = fit_poses(projections, nominal, grid, detector)
    lever = record["lever_px_per_radian"]
    assert np.sqrt(np.mean(table[:, 3:5] ** 2)) < 0.05
    assert np.sqrt(np.mean((lever * table[:, :3]) ** 2)) < 0.05


@pytest.mark.numerical
@pytest.mark.slow
def test_pose_fit_recovers_per_view_rotations():
    projections, nominal, grid, detector, truth = _moving_scan(wobble_deg=1.0)
    geometry = tj.LaminographyGeometry(
        grid, detector, np.linspace(0, 360, len(nominal), endpoint=False), tilt_deg=-30
    )
    empty = tj.Scan(np.zeros_like(projections), geometry)
    _, _, table, _ = fit_poses(
        projections, nominal, grid, detector, fit=("translations", "rotations")
    )
    estimated = np.asarray(stack_view_poses(empty.with_poses(table).geometry, len(nominal)))
    before, after = _landmark_error(nominal, truth, grid), _landmark_error(estimated, truth, grid)
    assert after < 0.1 and after < 0.2 * before, (before, after)
