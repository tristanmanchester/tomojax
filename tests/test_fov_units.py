"""ROI grids and supports follow the scan's own rays, whatever the physical unit."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj

# check-public-imports: allow-private
from tomojax.cli._reconstruction_region import cylinder_support, region_grid
from tomojax.geometry import Detector, Grid, grid_volume_origin

pytestmark = pytest.mark.numerical


def _scan(grid, detector, geometry=None):
    if geometry is None:
        geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 60, endpoint=False))
    views = len(geometry.angles)
    return tj.Scan(np.zeros((views, detector.nv, detector.nu), np.float32), geometry)


def _scaled(grid: Grid, detector: Detector, scale: float) -> tuple[Grid, Detector]:
    centre = None if grid.vol_center is None else tuple(v * scale for v in grid.vol_center)
    return (
        replace(
            grid, vx=grid.vx * scale, vy=grid.vy * scale, vz=grid.vz * scale, vol_center=centre
        ),
        replace(
            detector,
            du=detector.du * scale,
            dv=detector.dv * scale,
            center=tuple(c * scale for c in detector.center),
        ),
    )


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("roi", ["auto", "bbox", "cube", "cyl"])
def test_roi_cropping_gives_the_same_dimensions_at_any_unit(scale, roi) -> None:
    grid, detector = _scaled(Grid(11, 11, 11, 1.0, 1.0, 1.0), Detector(9, 9, 1.0, 1.0), scale)
    cropped, _ = region_grid(_scan(grid, detector), roi=roi, grid=None)
    assert (cropped.nx, cropped.ny, cropped.nz) == (9, 9, 9)
    assert cropped.vx == grid.vx


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("centre", [(0.0, 0.0, 0.0), (5.0, -1.0, 0.0)])
def test_cylinder_matches_independent_world_coordinates_at_any_unit(scale, centre) -> None:
    grid = Grid(nx=11, ny=13, nz=3, vx=1.0, vy=0.75, vz=1.0, vol_center=centre)
    detector = Detector(nu=9, nv=11, du=0.8, dv=1.0, center=(0.2, 0.0))
    x = np.arange(grid.nx) - (grid.nx - 1) / 2 + centre[0]
    y = (np.arange(grid.ny) - (grid.ny - 1) / 2) * grid.vy + centre[1]
    radius = (detector.nu - 1) * detector.du / 2 - abs(detector.center[0])
    expected = np.hypot(x[None, :], y[:, None]) <= radius
    grid, detector = _scaled(grid, detector, scale)
    angles = np.linspace(0, 360, 720, endpoint=False)
    scan = _scan(grid, detector, tj.ParallelGeometry(grid, detector, angles))
    _, cylinder = region_grid(scan, roi="cyl", grid=None)
    assert cylinder is not None
    np.testing.assert_array_equal(cylinder_support(grid, cylinder)[0] > 0, expected)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_cylinder_includes_voxels_exactly_on_a_diagonal_boundary(scale) -> None:
    grid = Grid(nx=11, ny=11, nz=1, vx=scale, vy=scale, vz=scale)
    detector = Detector(nu=11, nv=1, du=scale, dv=scale)
    _, cylinder = region_grid(_scan(grid, detector), roi="cyl", grid=None)
    assert cylinder is not None
    coords = np.arange(11) - 5
    expected = coords[:, None] ** 2 + coords[None, :] ** 2 <= 25
    np.testing.assert_array_equal(cylinder_support(grid, cylinder)[0] > 0, expected)


def test_single_pixel_detector_cylinder_keeps_only_the_on_axis_voxel() -> None:
    grid, detector = Grid(5, 5, 1, 1.0, 1.0, 1.0), Detector(1, 1, 1.0, 1.0)
    cropped, cylinder = region_grid(_scan(grid, detector), roi="cyl", grid=None)
    assert cylinder is not None and (cropped.nx, cropped.ny) == (1, 1)
    expected = np.zeros((5, 5), bool)
    expected[2, 2] = True
    np.testing.assert_array_equal(cylinder_support(grid, cylinder)[0] > 0, expected)


def test_roi_keeps_placement_only_when_shape_is_unchanged():
    # The grid sits off the axis in-plane but inside the detector's rows axially.
    grid = Grid(9, 9, 9, 0.5, 0.5, 0.5, vol_origin=(2.0, 3.0, -2.0))
    assert region_grid(_scan(grid, Detector(31, 31, 0.5, 0.5)), roi="bbox", grid=None)[0] is grid
    cropped, _ = region_grid(_scan(grid, Detector(21, 21, 0.5, 0.5)), roi="bbox", grid=None)
    # Voxel centres keep their positions: the crop is a window of the grid.
    offsets = (np.asarray(grid_volume_origin(cropped)) - grid_volume_origin(grid)) / 0.5
    np.testing.assert_allclose(offsets, np.round(offsets), atol=1e-9)
    assert cropped.nx < grid.nx


def test_axial_roi_follows_off_centre_detector_rows():
    grid = Grid(12, 12, 10, 1, 1, 1, vol_center=(0, 0, 1))
    detector = Detector(12, 2, 1, 1, center=(0, 2))
    result, _ = region_grid(_scan(grid, detector), roi="bbox", grid=None)
    assert result.nz == 2 and result.vz == 1
    assert grid_volume_origin(result)[2] == 1.5


@pytest.mark.parametrize("shape", [(9, 7, 3), (5, 4, 3), (7, 9, 3)])
def test_automatic_roi_keeps_a_rectangular_grid_the_detector_covers(shape):
    grid = Grid(*shape, 0.7, 1.1, 1.9, vol_origin=(2, 3, 4))
    assert region_grid(_scan(grid, Detector(101, 101, 1, 1)), roi="auto", grid=None)[0] == grid


def test_automatic_roi_still_crops_what_the_detector_cannot_see():
    cropped, _ = region_grid(
        _scan(Grid(11, 11, 5, 1, 1, 1), Detector(9, 3, 1, 1)), roi="auto", grid=None
    )
    assert (cropped.nx, cropped.ny, cropped.nz) == (9, 9, 3)


def test_cone_roi_measures_the_field_in_object_lengths():
    from tomojax.geometry import ConeBeam, ConeGeometry

    grid, detector = Grid(41, 41, 31, 1, 1, 1), Detector(40, 20, 1, 1)
    # Magnification 2 halves the detector's reach in the object: about 9.5 x 4.5.
    angles = np.linspace(0, 360, 90, endpoint=False)
    scan = _scan(grid, detector, ConeGeometry(grid, detector, angles, ConeBeam(100, 200)))
    cropped, _ = region_grid(scan, roi="bbox", grid=None)
    assert 17 <= cropped.nx <= 21 and 17 <= cropped.ny <= 21 and 7 <= cropped.nz <= 10


def test_laminography_roi_keeps_voxels_the_tilted_rays_measure():
    from tomojax.geometry import LaminographyGeometry, grid_volume_origin

    grid, detector = Grid(9, 9, 21, 1, 1, 1), Detector(30, 10, 1, 1)
    angles = np.linspace(0, 360, 90, endpoint=False)
    scan = _scan(grid, detector, LaminographyGeometry(grid, detector, angles, tilt_deg=60))
    cropped, _ = region_grid(scan, roi="bbox", grid=None)
    # The axis point at height 8 projects to detector v = 4, inside the 10 rows.
    top = grid_volume_origin(cropped)[2] + (cropped.nz - 1) * cropped.vz
    assert top >= 8
