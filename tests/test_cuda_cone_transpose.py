"""Dense forward-basis oracles for CUDA cone sampling at axis boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core.cone import beam_frame, cone_backproject, cone_project, frame_coefficients
from tomojax.geometry import ConeBeam, ConeGeometry, Detector, Grid
from tomojax.recon import CGLSConfig, cgls


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
    shape = (grid.nx, grid.ny, grid.nz)
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
    truth = np.random.default_rng(43).uniform(0.1, 1.0, (5, 4, 3)).astype(np.float32)
    # Independently implemented JAX sampling supplies the data. Check image
    # error too: the formerly mismatched adjoint stagnated far from the truth.
    data = cone_project(truth, coeff, grid, detector, backend="jax")
    volume, info = cgls(
        geometry,
        grid,
        detector,
        data,
        config=CGLSConfig(iterations=120, projector_backend="pallas"),
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
