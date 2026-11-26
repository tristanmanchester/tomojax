from __future__ import annotations

from dataclasses import dataclass
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tomojax import InvalidParameter

# check-public-imports: allow-private
from tomojax.core.geometry.transforms import align_u_to_v
from tomojax.geometry import (
    Detector,
    Grid,
    RotationAxisGeometry,
)
from tomojax.geometry.api import axis_pose_stack


def test_align_u_to_v_handles_exact_antiparallel_vectors() -> None:
    u = np.asarray([0.0, 0.0, 1.0])
    v = np.asarray([0.0, 0.0, -1.0])

    rotation = align_u_to_v(u, v)

    np.testing.assert_allclose(rotation @ u, v, atol=1e-9)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.parametrize("tilt", [0.05, 0.09, 1.0, 5.0])
def test_align_ez_to_axis_preserves_near_minus_z_tilt(tilt: float) -> None:
    tilt_rad = np.deg2rad(tilt)
    axis = np.asarray([np.sin(tilt_rad), 0.0, -np.cos(tilt_rad)], dtype=np.float32)

    rotation = np.asarray(axis_pose_stack([0.0], axis))[0, :3, :3]

    np.testing.assert_allclose(rotation @ np.asarray([0.0, 0.0, 1.0]), axis, atol=1e-4)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("azimuth", [-70, 210])
@pytest.mark.parametrize("tilt", [0, 90, 171.88, 171.9, 179.9])
def test_axis_poses_retain_the_minimum_rotation_frame(tilt, azimuth):
    # Specifying the full independent rotation catches a changed in-plane
    # frame even when the third column still matches the requested axis.
    phi = np.deg2rad(azimuth)
    alignment = Rotation.from_rotvec(
        np.deg2rad(tilt) * np.asarray([-np.sin(phi), np.cos(phi), 0])
    ).as_matrix()
    angles = np.asarray([0, 17, 90, 137])
    expected = alignment @ Rotation.from_euler("z", angles[:, None], degrees=True).as_matrix()
    actual = np.asarray(jax.jit(axis_pose_stack)(angles, alignment[:, 2]))[:, :3, :3]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_exact_reversed_axis_uses_the_same_half_turn_as_host_geometry():
    grid, detector = Grid(5, 4, 3, 1, 1, 1), Detector(7, 5, 1, 1)
    angles = np.asarray([0, 17, 90], dtype=np.float32)
    geometry = RotationAxisGeometry(grid, detector, angles, (0, 0, -1))
    expected = np.asarray([geometry.pose_matrix(i) for i in range(len(angles))])
    listed = RotationAxisGeometry(grid, detector, angles.tolist(), (0, 0, -1))
    np.testing.assert_array_equal(expected, [listed.pose_matrix(i) for i in range(len(angles))])
    actual = jax.jit(axis_pose_stack)(angles, np.asarray([0, 0, -1]))
    np.testing.assert_allclose(actual, expected, atol=2e-7)


@pytest.mark.parametrize("sign", [-1, 1])
def test_exact_axis_alignment_has_finite_forward_and_reverse_derivatives(sign):
    axis = jnp.asarray([0.0, 0.0, sign])

    def poses(a):
        return axis_pose_stack(jnp.asarray([0.0, 17.0]), a)

    for derivative in (jax.jacfwd, jax.jacrev):
        assert np.isfinite(jax.jit(derivative(poses))(axis)).all()


@pytest.mark.parametrize("tilt", [0, 30, 90, 171.9, 179.9])
def test_axis_pose_derivative_matches_the_physical_rotation(tilt):
    azimuth = np.deg2rad(30)
    direction = np.asarray([-np.sin(azimuth), np.cos(azimuth), 0])

    def poses(theta):
        axis = jnp.asarray(
            [jnp.sin(theta) * np.cos(azimuth), jnp.sin(theta) * np.sin(azimuth), jnp.cos(theta)]
        )
        return axis_pose_stack(jnp.asarray([0.0]), axis)[0, :3, :3]

    theta, step = np.deg2rad(tilt), 1e-5
    expected = (
        Rotation.from_rotvec((theta + step) * direction).as_matrix()
        - Rotation.from_rotvec((theta - step) * direction).as_matrix()
    ) / (2 * step)
    for derivative in (jax.jacfwd, jax.jacrev):  # including exactly at +z (tilt 0)
        actual = jax.jit(derivative(poses))(theta)
        np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("as_array", [list, jnp.asarray])
def test_axis_pose_stack_accepts_array_like_inputs(as_array):
    angles, axis = as_array([0.0, 17.0, 90.0]), as_array([0.0, 0.0, 1.0])
    expected = Rotation.from_euler("z", np.asarray(angles)[:, None], degrees=True).as_matrix()
    np.testing.assert_allclose(axis_pose_stack(angles, axis)[:, :3, :3], expected, atol=2e-7)


@pytest.mark.parametrize(
    "placement",
    [{}, {"vol_center": (3, -2, 1)}, {"vol_origin": (2, 4, -3), "vol_center": (9, 9, 9)}],
)
def test_grid_resampling_preserves_cell_edges_and_placement(placement):
    from tomojax.geometry import grid_volume_origin

    grid = Grid(11, 8, 5, 0.7, 1.3, 2.1, **placement)
    assert grid.shape == (5, 8, 11)
    assert Grid.from_dict(grid.to_dict()) == grid
    resampled = grid.resampled(shape=(3, 13, 4))
    assert resampled.shape == (3, 13, 4)
    old_pitch = np.array([grid.vx, grid.vy, grid.vz])
    new_pitch = np.array([resampled.vx, resampled.vy, resampled.vz])
    np.testing.assert_allclose(new_pitch * [4, 13, 3], old_pitch * [11, 8, 5])
    np.testing.assert_allclose(
        np.array(grid_volume_origin(resampled)) - new_pitch / 2,
        np.array(grid_volume_origin(grid)) - old_pitch / 2,
    )
    assert grid.resampled(shape=(5, 8, 11)) is grid


def test_rotation_axis_vocabulary_and_cone_segments_beam():
    from tomojax.geometry import (
        ConeBeam,
        ConeGeometry,
        ConeSegments,
        LaminographyGeometry,
        beam_of,
    )

    grid, detector = Grid(5, 4, 3, 1, 1, 1), Detector(7, 5, 1, 1)
    lamino = LaminographyGeometry(grid, detector, [0.0, 90.0], tilt=25.0)
    axis = lamino.rotation_axis()
    rot = RotationAxisGeometry(grid, detector, [0.0, 90.0], axis_unit=axis)
    cone = ConeGeometry(grid, detector, [0.0, 90.0], ConeBeam(30.0, 45.0), axis_unit=tuple(axis))
    np.testing.assert_allclose(rot.rotation_axis(), axis)
    np.testing.assert_allclose(cone.rotation_axis(), axis)
    with pytest.raises(ValueError, match="cone_parts"):
        beam_of(ConeSegments((cone,)))


def test_detector_grid_is_flattened_row_by_row_about_the_center() -> None:
    from tomojax.geometry.detector_grid import zero_center_detector_grid

    detector = Detector(nu=3, nv=2, du=2.0, dv=4.0, center=(10.0, -5.0))

    u, v = zero_center_detector_grid(detector)

    np.testing.assert_allclose(u, [-2.0, 0.0, 2.0, -2.0, 0.0, 2.0])
    np.testing.assert_allclose(v, [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0])


def test_grid_and_detector_round_trip_through_pickle() -> None:
    grid = Grid(5, 4, 3, 0.5, 0.25, 2.0, vol_origin=(1, 2, 3), vol_center=(4, 5, 6))
    detector = Detector(7, 6, 0.5, 0.75, (0.25, -1.5))
    assert pickle.loads(pickle.dumps(grid)) == grid
    assert pickle.loads(pickle.dumps(detector)) == detector


@pytest.mark.parametrize(
    ("kind", "field", "value", "message"),
    [
        ("grid", "nx", 0, "nx must be a positive integer"),
        ("grid", "vz", -1.0, "vz must be finite and positive"),
        ("grid", "vol_origin", (np.nan, 0.0, 0.0), "vol_origin must be finite"),
        ("detector", "du", 0.0, "du must be finite and positive"),
        ("detector", "center", (np.inf, 0.0), "center must be finite"),
    ],
)
def test_unpickling_rebuilds_through_the_constructor(kind, field, value, message) -> None:
    item = Grid(4, 4, 4, 1, 1, 1) if kind == "grid" else Detector(4, 4, 1, 1)
    # A stored object written before the constructor checked this field: its pickle
    # holds the same field state that a plain dataclass pickle always held.
    object.__setattr__(item, field, value)
    with pytest.raises(InvalidParameter, match=message) as error:
        pickle.loads(pickle.dumps(item))
    assert error.value.path == field


@pytest.mark.parametrize("field", ["vol_origin", "vol_center"])
def test_grid_rejects_nonfinite_placement(field) -> None:
    with pytest.raises(InvalidParameter, match=f"{field} must be finite"):
        Grid(4, 4, 4, 1, 1, 1, **{field: (0.0, np.inf, 0.0)})


@dataclass(frozen=True)
class ShiftedDetector(Detector):
    shift: float = 0.0


@dataclass(frozen=True)
class CalibratedDetector(Detector):
    offset: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        u, v = self.center
        object.__setattr__(self, "center", (u + self.offset[0], v + self.offset[1]))


class SlottedGrid(Grid):
    __slots__ = ("label",)


def test_copies_keep_grid_and_detector_subclasses_and_their_fields() -> None:
    import copy

    detector = ShiftedDetector(4, 3, 1.0, 1.0, shift=2.5)
    copied = copy.deepcopy(detector)
    assert type(copied) is ShiftedDetector and copied.shift == 2.5
    # A subclass's own initialization runs once, at construction, not again on a copy.
    calibrated = CalibratedDetector(4, 3, 1.0, 1.0, (1.0, 2.0), offset=(1.5, 1.5))
    assert pickle.loads(pickle.dumps(calibrated)).center == calibrated.center == (2.5, 3.5)
    assert copy.copy(calibrated).center == (2.5, 3.5)
    slotted = SlottedGrid(4, 4, 4, 1, 1, 1)
    object.__setattr__(slotted, "label", "slab")
    for restored in (
        copy.copy(slotted),
        copy.deepcopy(slotted),
        pickle.loads(pickle.dumps(slotted)),
    ):
        assert restored == slotted and restored.label == "slab"


def test_a_saved_tilt_may_be_a_numeric_string() -> None:
    from tomojax.geometry.api import nominal_axis_unit_from_inputs

    as_text = nominal_axis_unit_from_inputs({"geometry_type": "lamino", "tilt": "30.0"})
    as_number = nominal_axis_unit_from_inputs({"geometry_type": "lamino", "tilt": 30.0})
    np.testing.assert_array_equal(as_text, as_number)
