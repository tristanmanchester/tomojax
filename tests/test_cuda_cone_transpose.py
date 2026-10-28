"""Dense forward-basis oracles for CUDA cone sampling at axis boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core import cone
from tomojax.core.cone import beam_frame, cone_backproject, cone_project, frame_coefficients
from tomojax.geometry import ConeBeam, ConeGeometry, Detector, Grid
from tomojax.recon import CGLSConfig, cgls


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(1, 5, 3), (23, 17, 11), (501, 3, 2)])
def test_uniform_cone_rows_keep_near_parallel_boundary_weights(shape):
    grid, detector = Grid(*shape, 0.7, 1.1, 0.9), Detector(7, 5, 1.0, 1.0)
    coefficients = []
    for axis in range(3):
        b, c = (axis + 1) % 3, (axis + 2) % 3
        for edge in (-1.1, -0.3, 0.0, shape[b] - 1.3, shape[b] - 0.3, shape[b] + 0.1):
            row = np.zeros(25, np.float32)
            row[axis], row[b], row[c] = -100, edge, 0.4
            row[3:6] = row[:3]
            row[3 + axis] += 400
            row[3 + b] += 1e-4
            row[6 + b], row[9 + c] = 0.9, 0.7
            coefficients.append(row)
    coeff = jnp.asarray(np.stack(coefficients))
    expected = cone_project(jnp.ones(shape[::-1]), coeff, grid, detector, backend="cuda")
    actual = cone.cone_project_ones(coeff, grid, detector, backend="cuda")
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual) == 0, np.asarray(expected) == 0)
    gradient = jax.grad(lambda x: cone.cone_project_ones(x, grid, detector, backend="cuda").sum())(
        coeff
    )
    np.testing.assert_array_equal(gradient, jnp.zeros_like(coeff))


def _frame(direction, roll, *, reverse_v=False):
    direction = np.asarray(direction, np.float32)
    direction /= np.linalg.norm(direction)
    u = np.cross(direction, np.eye(3)[np.argmin(np.abs(direction))]).astype(np.float32)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    if reverse_v:
        v = -v
    u_rotated = u * np.cos(roll) + v * np.sin(roll)
    v_rotated = -u * np.sin(roll) + v * np.cos(roll)
    return np.stack([40 * direction, u_rotated, v_rotated, -80 * direction]).astype(np.float32)


def _check_dense_transpose(grid, detector, frames, poses=None):
    if poses is None:
        poses = jnp.broadcast_to(jnp.eye(4), (len(frames), 4, 4))
    coeff = frame_coefficients(poses, np.asarray(frames), grid, detector)
    shape = (grid.nz, grid.ny, grid.nx)
    count = np.prod(shape)
    basis = jnp.eye(count, dtype=jnp.float32).reshape((count, *shape))
    forward = jax.jit(jax.vmap(lambda x: cone_project(x, coeff, grid, detector, backend="cuda")))
    # No backprojector/JAX-scatter weights in the oracle: use each CUDA forward
    # basis column, and independently transpose the resulting matrix on the host.
    matrix = np.asarray(forward(basis), np.float64).reshape((count, -1))
    rng = np.random.default_rng(43)
    for signed in (False, True):
        values = rng.normal(size=matrix.shape[1]) if signed else rng.random(matrix.shape[1])
        images = values.astype(np.float32).reshape((len(frames), detector.nv, detector.nu))
        expected = (matrix @ images.ravel()).reshape(shape)
        actual = np.asarray(cone_backproject(images, coeff, grid, detector, backend="cuda"))
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 5e-7
        initial = rng.normal(size=shape).astype(np.float32)
        accumulated = np.asarray(
            cone_backproject(images, coeff, grid, detector, backend="cuda", accumulate=initial)
        )
        scale = np.abs(expected).max() + np.abs(initial).max()
        np.testing.assert_allclose(accumulated, expected + initial, rtol=0, atol=5e-7 * scale)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "direction", [(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (-1, 1, -1), (-1, -1, 0)]
)
@pytest.mark.parametrize("roll", [0.0, 0.13, 0.61])
@pytest.mark.parametrize("reverse_v", [False, True])
def test_cuda_cone_dense_transpose_at_dominant_axis_ties(direction, roll, reverse_v):
    # Odd detector/volume sizes, shifted voxel centres, pair and triple ties.
    # For (1,1,1), roll=.13, subtracting the source before adding v changes
    # the FP32 dominant axis and formerly produced a 1.6% transpose error.
    grid = Grid(7, 6, 5, 1.0, 1.0, 1.0, vol_center=(0.2, -0.1, 0.3))
    detector = Detector(9, 7, 1.3, 0.8)
    _check_dense_transpose(grid, detector, [_frame(direction, roll, reverse_v=reverse_v)])


@pytest.mark.gpu
def test_cuda_cone_dense_transpose_with_anisotropy_and_a_view_chunk_tail():
    grid = Grid(5, 4, 3, 0.7, 1.1, 0.9, vol_center=(0.2, -0.1, 0.3))
    detector = Detector(9, 7, 1.3, 0.8)
    directions = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    spacing = np.asarray([grid.vx, grid.vy, grid.vz])
    frames = [
        _frame(np.asarray(directions[i % 4]) * spacing, 0.0 if i % 3 == 0 else 0.13)
        for i in range(33)
    ]
    _check_dense_transpose(grid, detector, frames)


@pytest.mark.gpu
def test_cuda_cone_dense_transpose_with_sparse_and_cancelling_pixels():
    grid = Grid(7, 6, 5, 0.7, 1.1, 0.9, vol_center=(0.2, -0.1, 0.3))
    detector = Detector(9, 7, 1.3, 0.8)
    frame = _frame((1, 1, 1), 0.13)
    coeff = frame_coefficients(
        jnp.broadcast_to(jnp.eye(4), (2, 4, 4)), np.stack([frame, frame]), grid, detector
    )
    shape = (grid.nz, grid.ny, grid.nx)
    basis = jnp.eye(np.prod(shape), dtype=jnp.float32).reshape((-1, *shape))
    forward = jax.jit(jax.vmap(lambda x: cone_project(x, coeff, grid, detector, backend="cuda")))
    matrix = np.asarray(forward(basis), np.float64).reshape((np.prod(shape), -1))
    rng = np.random.default_rng(43)
    sparse = rng.normal(size=(2, detector.nv, detector.nu)).astype(np.float32)
    sparse[:, ::2] = 0
    cancelling = sparse.copy()
    cancelling[1] = -cancelling[0]
    impulse = np.zeros_like(sparse)
    impulse[0, detector.nv // 2, detector.nu // 2] = 1
    for images in (np.zeros_like(sparse), impulse, sparse, cancelling):
        expected = matrix @ images.ravel()
        # An absolute contribution bound remains meaningful under cancellation.
        scale = (np.abs(matrix) @ np.abs(images).ravel()).max()
        actual = np.asarray(cone_backproject(images, coeff, grid, detector, backend="cuda"))
        np.testing.assert_allclose(actual.ravel(), expected, rtol=0, atol=5e-7 * scale)


@pytest.mark.gpu
@pytest.mark.parametrize("reverse_v", [False, True])
def test_cuda_cone_dense_transpose_across_wide_contiguous_tiles(reverse_v):
    # A thin, long odd grid exercises multiple wide z tiles and their tail
    # without making the independent forward-basis matrix prohibitively large.
    grid = Grid(5, 4, 129, 0.7, 1.1, 0.3, vol_center=(0.2, -0.1, 0.3))
    detector = Detector(7, 131, 1.3, 0.4, center=(0.3, -0.2))
    frames = [
        _frame(direction, 0.13, reverse_v=reverse_v)
        for direction in ((1, 1, 0), (1, 0, 1), (1, 1, 1))
    ]
    _check_dense_transpose(grid, detector, frames)


@pytest.mark.gpu
def test_cuda_cone_dense_transpose_across_row_and_plane_tile_tails():
    grid = Grid(17, 5, 7, 0.7, 1.1, 0.9, vol_center=(0.2, -0.1, 0.3))
    detector = Detector(29, 17, 1.3, 0.8, center=(0.3, -0.2))
    # Cross row-tile boundaries and odd plane tails on each dominant axis.
    frames = [_frame(direction, 0.13) for direction in ((1, 0, 0.3), (0.2, 1, 0), (0.1, 0.2, 1))]
    _check_dense_transpose(grid, detector, frames)


@pytest.mark.gpu
def test_cuda_cone_dense_transpose_with_many_rays_per_cell():
    # Pixels a fifth of a voxel at the centre put ~25 rays on each cell per view: the
    # fixed-point tiles' largest sums, from positive images, must not wrap.
    grid = Grid(7, 6, 5, 1.0, 1.0, 1.0, vol_center=(0.2, -0.1, 0.3))
    detector = Detector(41, 37, 0.3, 0.3)
    frames = [_frame((1, 0.2 * i, 0.3), 0.13) for i in range(5)]
    _check_dense_transpose(grid, detector, frames)


@pytest.mark.gpu
def test_cuda_cone_pair_transpose_is_reproducible_with_one_bright_view():
    # A view a thousand times brighter sets the fixed-point range of every tile it
    # reaches; each voxel must still keep FP32 accuracy relative to its own terms.
    grid = Grid(7, 6, 5, 1.0, 1.0, 1.0)
    detector = Detector(9, 7, 1.1, 1.1)
    frames = np.stack([_frame((1, 0.3 * i, 0.2), 0.13) for i in range(40)])
    coeff = frame_coefficients(jnp.broadcast_to(jnp.eye(4), (40, 4, 4)), frames, grid, detector)
    count = grid.nx * grid.ny * grid.nz
    basis = jnp.eye(count, dtype=jnp.float32).reshape((count, grid.nz, grid.ny, grid.nx))
    forward = jax.jit(jax.vmap(lambda x: cone_project(x, coeff, grid, detector, backend="cuda")))
    matrix = np.asarray(forward(basis), np.float64).reshape((count, -1))
    rng = np.random.default_rng(47)
    images = rng.normal(size=(2, 40, detector.nv, detector.nu)).astype(np.float32)
    images[0, 3] *= 1e3
    first = cone.cone_backproject_pair(tuple(images), coeff, grid, detector, backend="cuda")
    second = cone.cone_backproject_pair(tuple(images), coeff, grid, detector, backend="cuda")
    for channel in (0, 1):
        # Integer tile sums do not depend on the order threads add them in.
        np.testing.assert_array_equal(first[channel], second[channel])
        expected = matrix @ images[channel].ravel()
        bound = np.abs(matrix) @ np.abs(images[channel]).ravel()
        error = np.abs(np.asarray(first[channel]).ravel() - expected)
        assert np.all(error <= 5e-7 * bound)


@pytest.mark.gpu
@pytest.mark.parametrize("tilt", [0.0, 20.0])
def test_cuda_cone_cgls_recovers_an_image_across_wide_tiles(tilt):
    # A fixed-budget wide-grid image control, independent of the CUDA forward
    # implementation. Dense basis tests separately enforce transpose accuracy.
    grid = Grid(129, 67, 131, 0.7, 0.9, 1.1, vol_center=(0.3, -0.2, 0.1))
    detector = Detector(193, 191, 1.0, 1.2, center=(0.2, -0.1))
    geometry = ConeGeometry(
        grid, detector, np.linspace(0, 360, 65, endpoint=False), ConeBeam(387, 580.5), tilt_deg=tilt
    )
    frames = np.broadcast_to(beam_frame(geometry.beam, detector), (65, 4, 3))
    coeff = frame_coefficients(jnp.asarray(geometry.poses(), jnp.float32), frames, grid, detector)
    z, y, x = np.ogrid[-65:66, -33:34, -64:65]
    truth = (
        np.exp(-((x / 27) ** 2 + (y / 15) ** 2 + (z / 28) ** 2))
        + 0.5 * np.exp(-((x / 15 - 1) ** 2 + (y / 10 + 0.8) ** 2 + (z / 12 + 0.6) ** 2))
    ).astype(np.float32)
    data = cone_project(truth, coeff, grid, detector, backend="jax")
    volume, _ = cgls(
        geometry,
        grid,
        detector,
        data,
        config=CGLSConfig(iterations=30, backend="cuda"),
    )
    assert np.linalg.norm(np.asarray(volume) - truth) / np.linalg.norm(truth) < 0.02


@pytest.mark.gpu
@pytest.mark.parametrize("yaw", [0.0, 17.0])
def test_cuda_cone_dense_transpose_on_separable_turntable_views(yaw):
    grid = Grid(7, 6, 5, 0.7, 1.1, 0.9, vol_center=(0.2, -0.1, 0.3))
    detector = Detector(9, 7, 1.3, 0.8, center=(0.3, -0.4))
    geometry = ConeGeometry(
        grid, detector, np.arange(0, 360, 30), ConeBeam(40, 60, detector_yaw_deg=yaw)
    )
    frames = np.broadcast_to(beam_frame(geometry.beam, detector), (12, 4, 3))
    _check_dense_transpose(grid, detector, frames, jnp.asarray(geometry.poses(), jnp.float32))


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("kernel", "nz"),
    [(k, nz) for k in ("sep_adjoint", "sep_adjoint_cached") for nz in (3, 67)]
    + [("sep_adjoint_thin", 3), ("sep_adjoint_cached_thin", 3)],  # thin tiles: 32 planes
)
@pytest.mark.parametrize("pitch", [0.4, 2.0])
def test_cuda_separable_transpose_kernels_with_wide_support_and_view_tail(
    kernel, pitch, nz, monkeypatch
):
    # Keep illuminated voxels at the far z end, across the 64-cell tile boundary.
    z_center = 0.3 - (nz - 3) * 0.9 / 2
    grid = Grid(5, 4, nz, 0.7, 1.1, 0.9, vol_center=(0.2, -0.1, z_center))
    detector = Detector(9, 9 if pitch < 1 else 5, 1.3, pitch, center=(0.3, -0.4))
    geometry = ConeGeometry(
        grid, detector, np.linspace(0, 360, 33, endpoint=False), ConeBeam(40, 60)
    )
    module = cone.cuda_module()
    calls = []
    selected = []

    class Kernels:
        def get_function(self, name):
            if name.startswith("sep_adjoint"):
                selected.append(name)
                name = kernel
            calls.append(name)
            return module.get_function(name)

    monkeypatch.setattr(cone, "cuda_module", Kernels)
    frames = np.broadcast_to(beam_frame(geometry.beam, detector), (33, 4, 3))
    _check_dense_transpose(grid, detector, frames, jnp.asarray(geometry.poses(), jnp.float32))
    assert kernel in calls
    expected = "sep_adjoint" if nz == 3 and pitch < 1 else "sep_adjoint_cached"
    assert selected and set(selected) == {expected + "_thin" * (nz == 3)}
    if kernel.endswith("_thin"):
        return  # the case below has 64 planes; thin tiles share its arithmetic

    # The inverse vertical bound rounds past a tiny positive forward weight.
    grid, detector = Grid(64, 64, 64, 1, 1, 1), Detector(64, 64, 2, 2.3)
    geometry = ConeGeometry(grid, detector, [74.5], ConeBeam(256, 512))
    coeff = frame_coefficients(
        jnp.asarray(geometry.poses(), jnp.float32),
        np.asarray([beam_frame(geometry.beam, detector)]),
        grid,
        detector,
    )
    point, ray = (44, 22, 48), (0, 41, 44)
    volume = jnp.zeros((64, 64, 64), jnp.float32).at[point].set(1)
    images = jnp.zeros((1, 64, 64), jnp.float32).at[ray].set(1)
    forward = float(cone_project(volume, coeff, grid, detector, backend="cuda")[ray])
    backward = float(cone_backproject(images, coeff, grid, detector, backend="cuda")[point])
    assert forward > 1e-6
    np.testing.assert_allclose(backward, forward, rtol=2e-6, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("yaw", [56.4, 70.0, 80.0])
@pytest.mark.parametrize("tilt", [0.0, 5.0])
def test_cuda_cone_dense_transpose_when_a_footprint_crosses_infinity(yaw, tilt):
    # The source remains outside the volume. A close source and yawed detector
    # put a pole inside the tile's inverse projection, so finite corner bounds
    # exclude contributing pixels. Check both separable and plane-tile paths.
    grid = Grid(5, 4, 3, 1.16, 1.46, 1.26)
    detector = Detector(9, 7, 1.29, 0.52, center=(-0.77, -1.05))
    geometry = ConeGeometry(
        grid, detector, [45.0], ConeBeam(7.0, 10.4, detector_yaw_deg=yaw), tilt_deg=tilt
    )
    frames = np.asarray([beam_frame(geometry.beam, detector)])
    _check_dense_transpose(grid, detector, frames, jnp.asarray(geometry.poses(), jnp.float32))


@pytest.mark.gpu
@pytest.mark.parametrize("tilt", [0.0, 5.0])
def test_cuda_cone_cgls_recovers_the_image_with_unbounded_footprints(tilt):
    grid = Grid(5, 4, 3, 1.16, 1.46, 1.26)
    detector = Detector(9, 11, 1.29, 0.52, center=(-0.77, 0.0))
    geometry = ConeGeometry(
        grid,
        detector,
        np.arange(0, 360, 15),
        ConeBeam(7.0, 10.4, detector_yaw_deg=70),
        tilt_deg=tilt,
    )
    poses = jnp.asarray(geometry.poses(), jnp.float32)
    frames = np.broadcast_to(beam_frame(geometry.beam, detector), (24, 4, 3))
    coeff = frame_coefficients(poses, frames, grid, detector)
    truth = np.random.default_rng(43).uniform(0.1, 1.0, (3, 4, 5)).astype(np.float32)
    # Independently implemented JAX sampling supplies the data. Check image
    # error too: the formerly mismatched adjoint stagnated far from the truth.
    data = cone_project(truth, coeff, grid, detector, backend="jax")
    volume, info = cgls(
        geometry,
        grid,
        detector,
        data,
        config=CGLSConfig(iterations=120, backend="cuda"),
    )
    assert info["converged"]
    assert np.linalg.norm(np.asarray(volume) - truth) / np.linalg.norm(truth) < 1e-4
    projected = np.asarray(cone_project(volume, coeff, grid, detector, backend="jax"))
    assert np.linalg.norm(projected - np.asarray(data)) / np.linalg.norm(np.asarray(data)) < 1e-5


@pytest.mark.gpu
@pytest.mark.parametrize("pitch", [(1e-9, 0.52), (1.29, 1e-9)])
@pytest.mark.parametrize("tilt", [0.0, 5.0])
def test_cuda_cone_dense_transpose_with_footprint_bounds_beyond_int32(pitch, tilt):
    # Narrow detector windows are valid. Their inverse footprints can exceed
    # int32 although the actual image has only nine by seven pixels.
    grid = Grid(5, 4, 3, 1.16, 1.46, 1.26)
    detector = Detector(9, 7, *pitch, center=(-0.77, -1.05))
    geometry = ConeGeometry(grid, detector, [45.0], ConeBeam(7.0, 10.4), tilt_deg=tilt)
    frames = np.asarray([beam_frame(geometry.beam, detector)])
    _check_dense_transpose(grid, detector, frames, jnp.asarray(geometry.poses(), jnp.float32))


@pytest.mark.gpu
@pytest.mark.parametrize("yaw", [70.0, 110.0])
@pytest.mark.parametrize("angle", [0.0, 45.0, 90.0])
def test_cuda_cone_dense_transpose_with_reversed_separable_row_steps(yaw, angle):
    # A wide yawed detector includes rays whose source-side plane intersections
    # run downwards in z as the detector row increases. The forward projector
    # samples those lines, so their transpose must enumerate negative steps too.
    grid = Grid(5, 4, 3, 1.16, 1.46, 1.26)
    detector = Detector(9, 9, 8.0, 0.52, center=(-0.77, 0.0))
    geometry = ConeGeometry(grid, detector, [angle], ConeBeam(7.0, 10.4, detector_yaw_deg=yaw))
    frames = np.asarray([beam_frame(geometry.beam, detector)])
    _check_dense_transpose(grid, detector, frames, jnp.asarray(geometry.poses(), jnp.float32))


@pytest.mark.gpu
def test_cuda_cone_dense_transpose_with_empty_views_and_masked_pixels():
    grid = Grid(5, 4, 3, 0.7, 1.1, 0.9, vol_center=(0.2, -0.1, 0.3))
    detector = Detector(9, 17, 1.3, 0.8)
    directions = [(1, 1, 0), (1, 0, 1), (1, 1, 1)]
    # More views than one block has threads: each kernel walks them all.
    frames = [
        _frame(directions[i % 3], 0.0 if i % 3 == 0 else 0.13, reverse_v=i % 2 == 0)
        for i in range(300)
    ]
    coeff = frame_coefficients(
        jnp.broadcast_to(jnp.eye(4), (300, 4, 4)), np.asarray(frames), grid, detector
    )
    shape = (grid.nz, grid.ny, grid.nx)
    basis = jnp.eye(np.prod(shape), dtype=jnp.float32).reshape((-1, *shape))
    forward = jax.jit(jax.vmap(lambda x: cone_project(x, coeff, grid, detector, backend="cuda")))
    matrix = np.asarray(forward(basis), np.float64).reshape((np.prod(shape), -1))
    rng = np.random.default_rng(2043)
    initial = rng.normal(size=shape).astype(np.float32)
    for included in ([], [299], [0, 150, 299]):
        images = np.zeros((300, detector.nv, detector.nu), np.float32)
        images[included] = rng.normal(size=(len(included), detector.nv, detector.nu))
        images[:, ::2, 1::2] = 0
        expected = (matrix @ images.ravel()).reshape(shape) + initial
        # Absolute contributions bound cancellation without using the adjoint.
        scale = (np.abs(matrix) @ np.abs(images).ravel()).max() + np.abs(initial).max()
        actual = np.asarray(
            cone_backproject(images, coeff, grid, detector, backend="cuda", accumulate=initial)
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=7e-7 * scale)


@pytest.mark.gpu
@pytest.mark.parametrize("reverse_v", [False, True])
def test_cuda_cone_dense_transpose_across_sparse_image_tile_seams(reverse_v):
    grid = Grid(5, 4, 3, 0.7, 1.1, 0.9, vol_center=(0.2, -0.1, 0.3))
    detector = Detector(35, 37, 0.14, 0.17, center=(0.3, -0.2))
    directions = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    spacing = np.array([grid.vx, grid.vy, grid.vz])
    frames = [
        _frame(np.asarray(directions[i % 4]) * spacing, 0.13, reverse_v=reverse_v)
        for i in range(33)
    ]
    coeff = frame_coefficients(
        jnp.broadcast_to(jnp.eye(4), (33, 4, 4)), np.asarray(frames), grid, detector
    )
    shape = (grid.nz, grid.ny, grid.nx)
    count = np.prod(shape)
    basis = jnp.eye(count, dtype=jnp.float32).reshape((count, *shape))
    forward = jax.jit(jax.vmap(lambda x: cone_project(x, coeff, grid, detector, backend="cuda")))
    matrix = np.asarray(forward(basis), np.float64).reshape((count, -1))
    rng = np.random.default_rng(4610)
    initial = rng.normal(size=shape).astype(np.float32)
    for included in ([], [32], [0, 16, 32]):
        images = np.zeros((33, detector.nv, detector.nu), np.float32)
        # Sparse values straddle both 32-pixel image seams and the odd tails,
        # with wholly empty views in both the full and the short view chunk.
        for view in included:
            images[view, 30:37, 30:35] = rng.normal(size=(7, 5))
            images[view, ::8, ::8] = rng.normal(size=(5, 5))
        expected = (matrix @ images.ravel()).reshape(shape) + initial
        bound = (np.abs(matrix) @ np.abs(images).ravel()).max() + np.abs(initial).max()
        actual = np.asarray(
            cone_backproject(images, coeff, grid, detector, backend="cuda", accumulate=initial)
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=7e-7 * bound)


@pytest.mark.gpu
def test_cuda_cone_rounds_inverse_corner_bounds_outward():
    grid, detector = Grid(1, 1, 1, 1, 1, 1), Detector(256, 256, 2, 2)
    geometry = ConeGeometry(grid, detector, [45], ConeBeam(1024, 2048))
    poses = geometry.poses().astype(np.float32)
    poses[0, :3, :3] @= np.array([[1, 0, 0], [0, 1, -1e-6], [0, 1e-6, 1]], np.float32)
    poses[0, :3, 3] = [75.462677, -215.025162, 33.546272]
    frames = [beam_frame(geometry.beam, detector)]
    _check_dense_transpose(grid, detector, frames, jnp.asarray(poses))


@pytest.mark.gpu
@pytest.mark.parametrize("rhs", [1, 2])
def test_cuda_cone_model_matches_dense_transpose_with_donated_accumulation(rhs):
    grid = Grid(5, 4, 7, 0.7, 1.1, 0.9, vol_center=(0.2, -0.1, 0.3))
    detector = Detector(9, 7, 1.3, 0.8)
    directions = [(1, 1, 0), (1, 0, 1), (0, 1, -1), (-1, 1, -2)]
    frames = [_frame(directions[i % 4], 0.13, reverse_v=i % 2 == 0) for i in range(70)]
    coeff = frame_coefficients(
        jnp.broadcast_to(jnp.eye(4), (70, 4, 4)), np.asarray(frames), grid, detector
    )
    shape = (grid.nz, grid.ny, grid.nx)
    basis = jnp.eye(np.prod(shape), dtype=jnp.float32).reshape((-1, *shape))
    forward = jax.jit(jax.vmap(lambda x: cone_project(x, coeff, grid, detector, backend="cuda")))
    matrix = np.asarray(forward(basis), np.float64).reshape((np.prod(shape), -1))
    rng = np.random.default_rng(576)
    images = rng.normal(size=(rhs, 70, detector.nv, detector.nu)).astype(np.float32)
    images[:, ::3] = 0
    images[:, :, ::2, 1::2] = 0
    initial = tuple(rng.normal(size=shape).astype(np.float32) for _ in range(rhs))

    model = cone.ConeModel(((70, ConeBeam(40, 80), detector),))

    def transpose(*values):
        kwargs = dict(grid=grid, detector=detector, backend="cuda")
        if rhs == 1:
            return (
                model.backproject(jnp.asarray(images[0]), coeff, accumulate=values[0], **kwargs),
            )
        return model.backproject_pair(
            tuple(jnp.asarray(image) for image in images), coeff, accumulate=values, **kwargs
        )

    for donated in (False, True):
        apply = jax.jit(transpose, donate_argnums=tuple(range(rhs)) if donated else ())
        actual = apply(*(jnp.array(value, copy=True) for value in initial))
        for result, image, value in zip(actual, images, initial, strict=True):
            expected = (matrix @ image.ravel()).reshape(shape) + value
            bound = (np.abs(matrix) @ np.abs(image).ravel()).max() + np.abs(value).max()
            np.testing.assert_allclose(result, expected, rtol=0, atol=7e-7 * bound)
