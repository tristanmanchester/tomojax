"""Ray inspection agrees with the physical detector lattice of fitted roll."""

import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax._data.geometry_meta import DetectorRollGeometry
from tomojax.geometry import (
    ConeBeam,
    ConeGeometry,
    Detector,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
)
from tomojax.geometry.detector_grid import detector_grid_from_geometry_inputs


@pytest.mark.parametrize("roll", [0.0, 0.75])
@pytest.mark.parametrize("nested", [False, True])
def test_parallel_ray_inspection_matches_the_projectors_rolled_detector_grid(roll, nested):
    grid = Grid(24, 24, 24, 2, 2, 2, vol_origin=(-23, -19, -21))
    detector = Detector(48, 48, 1.3, 0.7, center=(2.2, -0.9))
    base = LaminographyGeometry(grid, detector, np.linspace(0, 360, 60), tilt_deg=25)
    inner = DetectorRollGeometry(base, -2.4) if nested else base
    geometry = DetectorRollGeometry(inner, roll)
    origin, direction = geometry.rays_for_view(17)
    points = np.array([origin(u, v) for v in range(48) for u in range(48)])
    lattice = detector_grid_from_geometry_inputs(detector, geometry)
    if lattice is None:
        plain_origin, _ = base.rays_for_view(17)
        expected = np.array([plain_origin(u, v) for v in range(48) for u in range(48)])
        np.testing.assert_allclose(points, expected, atol=1e-14)
    else:
        np.testing.assert_allclose(points[:, [0, 2]], np.stack(lattice, axis=-1), atol=3e-6)
    np.testing.assert_array_equal(points[:, 1], -19)
    assert direction(0, 0) == direction(47, 47) == (0.0, 1.0, 0.0)
    np.testing.assert_array_equal(geometry.pose_for_view(17), base.pose_for_view(17))


def test_cone_inspection_retains_its_source_and_beam_detector_frame():
    grid = Grid(24, 24, 24, 2, 2, 2)
    detector = Detector(48, 48, 1.3, 0.7, center=(2.2, -0.9))
    beam = ConeBeam(192, 288, detector_roll_deg=-0.75, detector_pitch_deg=1, source_height=3)
    base = ConeGeometry(grid, detector, np.linspace(0, 360, 60), beam)
    # Parallel roll metadata must not move a cone source or rotate its beam frame again.
    geometry = DetectorRollGeometry(base, 7.0)
    origin, direction = geometry.rays_for_view(17)
    centre, u_axis, v_axis = beam.detector_frame(detector)
    for u, v in ((0, 0), (24, 24), (47, 47)):
        pixel = centre + (u - 23.5) * detector.du * u_axis + (v - 23.5) * detector.dv * v_axis
        expected = pixel - beam.source()
        expected /= np.linalg.norm(expected)
        np.testing.assert_array_equal(origin(u, v), beam.source())
        np.testing.assert_allclose(direction(u, v), expected, atol=1e-14)


@pytest.mark.parametrize("roll", [-3.0, 0.0, 3.0])
def test_parallel_and_cone_roll_the_detector_the_same_way(roll):
    # Positive roll turns detector u towards v: a point above the axis (lab +z)
    # projects to u = h sin(roll) (times the magnification in cone), so it lands
    # on the +u side for positive roll in both geometries.
    import tomojax as tj

    grid, detector = Grid(33, 33, 33, 1.0, 1.0, 1.0), Detector(65, 65, 1.0, 1.0)
    volume = np.zeros((33, 33, 33), np.float32)
    volume[16 + 12, 16, 16] = 1.0  # (z, y, x): 12 voxels above the centre
    parallel = DetectorRollGeometry(ParallelGeometry(grid, detector, [0.0]), roll)
    cone = ConeGeometry(grid, detector, [0.0], ConeBeam(100.0, 150.0, detector_roll_deg=roll))
    columns = np.arange(65) - 32
    for geometry in (parallel, cone):
        image = np.asarray(tj.project(geometry, volume))[0]
        u = float(np.sum(image.sum(0) * columns) / image.sum())
        if roll == 0:
            assert abs(u) < 0.05
        else:
            assert np.sign(u) == np.sign(roll) and abs(u) > 0.3


@pytest.mark.parametrize(
    ("angle", "voxel", "axis"), [("pitch", (28, 16, 16), 0), ("yaw", (16, 16, 28), 1)]
)
def test_cone_pitch_and_yaw_move_the_detector_edges_as_documented(angle, voxel, axis):
    # Pitch tilts the +v edge towards the source and yaw swings the +u edge away
    # from it. A point above the axis (pitch) or beside it (yaw) projects to that
    # edge, where the source-detector distance, and so the magnification, changes:
    # nearer the source projects it less far out, further away more.
    import tomojax as tj

    grid, detector = Grid(33, 33, 33, 1.0, 1.0, 1.0), Detector(65, 65, 1.0, 1.0)
    volume = np.zeros((33, 33, 33), np.float32)
    volume[voxel] = 1.0  # (z, y, x): 12 voxels above or beside the centre
    offset = np.arange(65) - 32

    def outward(degrees):
        beam = ConeBeam(100.0, 150.0, **{f"detector_{angle}_deg": degrees})
        image = np.asarray(tj.project(ConeGeometry(grid, detector, [0.0], beam), volume))[0]
        profile = image.sum(1 - axis)  # along v (rows) for pitch, along u for yaw
        return float(np.sum(profile * offset) / profile.sum())

    assert outward(0.0) > 15
    # Foreshortening moves the point outwards for either sign; the documented
    # direction is the part odd in the angle.
    odd = outward(10.0) - outward(-10.0)
    assert odd < -0.5 if angle == "pitch" else odd > 0.5
