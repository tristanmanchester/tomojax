"""The alignment gauge: estimates that predict the same data, and the least-motion one."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import tomojax as tj

# check-public-imports: allow-private
from tomojax._data.geometry_meta import AugmentedGeometry

# check-public-imports: allow-private
from tomojax.alignment._gauge import Gauge, apply_to_poses, apply_to_volume, least_motion_gauge
from tomojax.alignment.api import apply_pose_updates
from tomojax.geometry import stack_view_poses

pytestmark = pytest.mark.numerical


def _scan(kind: str, views: int = 36, size: int = 24):
    grid = tj.Grid(size, size, size, 1.0, 1.0, 1.0)
    detector = tj.Detector(size + 8, size + 8, 1.0, 1.0)
    angles = np.linspace(0.0, 360.0, views, endpoint=False)
    if kind == "lamino":
        geometry = tj.LaminographyGeometry(grid, detector, angles, tilt_deg=30)
    elif kind == "cone":
        geometry = tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(80.0, 120.0))
    else:
        geometry = tj.ParallelGeometry(grid, detector, angles)
    return geometry, grid, np.asarray(stack_view_poses(geometry, views), np.float64)


def _params(views: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [np.deg2rad(rng.uniform(-2, 2, (views, 3))), rng.uniform(-1.5, 1.5, (views, 3))], axis=1
    ).astype(np.float32)


def _poses(nominal, params, frame):
    return np.asarray(
        apply_pose_updates(
            jnp.asarray(nominal, jnp.float32), jnp.asarray(params), translation_frame=frame
        ),
        np.float64,
    )


def _gauge(seed: int = 5) -> Gauge:
    rng = np.random.default_rng(seed)
    rotation = Rotation.from_rotvec(np.deg2rad(rng.uniform(-3, 3, 3))).as_matrix()
    return Gauge(rotation, rng.uniform(-2, 2, 3))


@pytest.mark.parametrize("frame", ["detector", "object"])
@pytest.mark.parametrize("kind", ["parallel", "lamino", "cone"])
def test_moving_the_poses_composes_every_view_with_the_rigid_motion(kind, frame):
    _, _, nominal = _scan(kind)
    params, gauge = _params(len(nominal)), _gauge()
    moved = apply_to_poses(params, nominal, gauge, translation_frame=frame)
    motion = np.eye(4)
    motion[:3, :3], motion[:3, 3] = gauge.rotation, gauge.shift
    np.testing.assert_allclose(
        _poses(nominal, moved, frame), _poses(nominal, params, frame) @ motion, atol=2e-5
    )


def test_a_detector_offset_moves_every_view_and_the_detector_centre():
    _, _, nominal = _scan("parallel")
    params = _params(len(nominal))
    gauge = Gauge(np.eye(3), np.zeros(3), detector_offset=0.7)
    moved = apply_to_poses(params, nominal, gauge, translation_frame="detector")
    np.testing.assert_allclose(moved[:, 3] - params[:, 3], 0.7, atol=1e-6)


@pytest.mark.parametrize("kind", ["parallel", "cone"])
def test_the_moved_volume_and_poses_predict_the_same_data(kind):
    geometry, grid, nominal = _scan(kind)
    centres = (np.arange(grid.nx) - (grid.nx - 1) / 2) / 3.0  # compact: ~0 at the edges
    x, y, z = np.meshgrid(centres, centres, centres, indexing="ij")
    volume = np.exp(-((x - 0.4) ** 2 + (y + 0.3) ** 2 / 2 + z**2 / 3)).astype(np.float32)
    params, gauge = _params(len(nominal)), _gauge()
    moved = apply_to_poses(params, nominal, gauge, translation_frame="detector")

    def data(p, v):
        posed = AugmentedGeometry(geometry, np.asarray(p), translation_frame="detector")
        return np.asarray(tj.project(posed, v))  # pyright: ignore[reportArgumentType]

    before = data(params, volume)
    after = data(moved, apply_to_volume(volume, grid, gauge))
    assert np.linalg.norm(after - before) / np.linalg.norm(before) < 0.02


def test_a_sliver_of_motion_keeps_the_faces_of_the_grid():
    # An object filling the grid, moved a ten-thousandth of a voxel: trilinear
    # resampling with zeros outside must change its faces by as little.
    grid = tj.Grid(8, 8, 8, 1.0, 1.0, 1.0)
    volume = np.ones((8, 8, 8), np.float32)
    for shift in ((0, 0, -1e-4), (0, 0, 1e-4), (1e-4, -1e-4, 0)):
        moved = apply_to_volume(volume, grid, Gauge(np.eye(3), np.asarray(shift)))
        np.testing.assert_allclose(moved, volume, atol=2e-4)


@pytest.mark.parametrize("frame", ["detector", "object"])
def test_the_least_motion_estimate_is_reached_from_any_member_of_the_orbit(frame):
    _, _, nominal = _scan("lamino")
    params = _params(len(nominal))
    first = apply_to_poses(
        params,
        nominal,
        least_motion_gauge(params, nominal, translation_frame=frame),
        translation_frame=frame,
    )
    shifted = apply_to_poses(params, nominal, _gauge(), translation_frame=frame)
    second = apply_to_poses(
        shifted,
        nominal,
        least_motion_gauge(shifted, nominal, translation_frame=frame),
        translation_frame=frame,
    )
    np.testing.assert_allclose(second, first, atol=2e-4)
    # It has no common rotation left and no remaining rigid shift to remove.
    again = least_motion_gauge(first, nominal, translation_frame=frame)
    np.testing.assert_allclose(again.rotation, np.eye(3), atol=1e-5)
    np.testing.assert_allclose(again.shift, 0.0, atol=1e-4)


def test_frozen_parameters_stay_fixed_and_invisible_ones_do_not_count():
    _, _, nominal = _scan("parallel")
    params = _params(len(nominal))
    params[:, DOF_DZ] = 0.0
    active = ("alpha", "beta", "phi", "dx")  # dz frozen; dy is invisible in a parallel beam
    gauge = least_motion_gauge(
        params, nominal, translation_frame="detector", active=active, invisible=("dy",)
    )
    moved = apply_to_poses(params, nominal, gauge, translation_frame="detector", keep=("dz", "dy"))
    np.testing.assert_allclose(moved[:, DOF_DZ], 0.0)
    # Frozen dz pins the shift along the axis; only the in-plane shift is free.
    assert abs(gauge.shift[2]) < 1e-6
    np.testing.assert_allclose(moved[:, 5], params[:, 5])


DOF_DZ = 4


def test_align_returns_the_least_motion_estimate_and_the_detector_centre():
    # A half-turn parallel scan with an offset detector and per-view motion:
    # left alone, the solve drifts along a rigid object shift.
    size, views = 20, 48
    grid = tj.Grid(size, size, size, 1.0, 1.0, 1.0)
    detector = tj.Detector(size + 8, size + 4, 1.0, 1.0)
    angles = np.linspace(0.0, 180.0, views, endpoint=False)
    geometry = tj.ParallelGeometry(grid, detector, angles)
    nominal = np.asarray(stack_view_poses(geometry, views), np.float64)
    rng = np.random.default_rng(11)
    truth = np.zeros((views, 6), np.float32)
    truth[:, 3:5] = rng.uniform(-1, 1, (views, 2))
    truth = apply_to_poses(
        truth,
        nominal,
        least_motion_gauge(truth, nominal, translation_frame="detector", invisible=("dy",)),
        translation_frame="detector",
        keep=("dy",),
    )
    centres = (np.arange(size) - (size - 1) / 2) / 3.0
    x, y, z = np.meshgrid(centres, centres, centres, indexing="ij")
    volume = (
        np.exp(-((x - 0.5) ** 2 + y**2 + (z + 0.4) ** 2))
        + 0.5 * np.exp(-((x + 0.8) ** 2 + (y - 0.6) ** 2 + z**2 / 2))
    ).astype(np.float32)
    shifted = tj.ParallelGeometry(
        grid, tj.Detector(size + 8, size + 4, 1.0, 1.0, (1.5, 0.0)), angles
    )
    data = np.asarray(tj.project(AugmentedGeometry(shifted, truth, "detector"), volume))  # pyright: ignore[reportArgumentType]

    result = tj.align(tj.Scan(data, geometry), mode="cor-then-pose")

    assert result.info["gauge"] is not None
    left = least_motion_gauge(
        result.poses, nominal, translation_frame="detector", active=("dx", "dz"), invisible=("dy",)
    )
    np.testing.assert_allclose(left.shift, 0.0, atol=1e-3)
    np.testing.assert_allclose(result.poses[:, 3:5], truth[:, 3:5], atol=0.1)
    assert result.scan.detector.center[0] == pytest.approx(1.5, abs=0.1)
