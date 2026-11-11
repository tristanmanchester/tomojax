"""FOV masks and ROI grids depend on geometry, not the physical unit chosen."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tomojax.geometry import (
    Detector,
    Grid,
    compute_roi,
    cylindrical_mask_xy,
    grid_from_detector_fov,
    grid_from_detector_fov_cube,
    grid_from_detector_fov_slices,
)

pytestmark = pytest.mark.numerical


def _scale(grid: Grid, detector: Detector, scale: float) -> tuple[Grid, Detector]:
    return (
        replace(
            grid,
            vx=grid.vx * scale,
            vy=grid.vy * scale,
            vz=grid.vz * scale,
            vol_center=None
            if grid.vol_center is None
            else tuple(v * scale for v in grid.vol_center),
        ),
        replace(
            detector,
            du=detector.du * scale,
            dv=detector.dv * scale,
            center=tuple(c * scale for c in detector.center),
        ),
    )


@pytest.mark.parametrize("scale", [1e-30, 1e-12, 1e-6, 1.0, 1e6, 1e20, 1e30])
@pytest.mark.parametrize("center", [(0.0, 0.0, 0.0), (5.0, -1.0, 0.0)])
def test_cylindrical_mask_matches_independent_world_coordinates_at_any_unit(scale, center) -> None:
    grid = Grid(nx=11, ny=13, nz=3, vx=1.0, vy=0.75, vz=1.0, vol_center=center)
    detector = Detector(nu=9, nv=11, du=0.8, dv=1.0, center=(0.2, 0.0))
    x = np.arange(grid.nx) - (grid.nx - 1) / 2 + center[0]
    y = (np.arange(grid.ny) - (grid.ny - 1) / 2) * grid.vy + center[1]
    radius = (detector.nu - 1) * detector.du / 2 - abs(detector.center[0])
    expected = np.hypot(x[None, :], y[:, None]) <= radius
    scaled_grid, scaled_detector = _scale(grid, detector, scale)
    np.testing.assert_array_equal(cylindrical_mask_xy(scaled_grid, scaled_detector), expected)


@pytest.mark.parametrize("scale", [1e-30, 1e-12, 1e-6, 1.0, 1e20, 1e30])
@pytest.mark.parametrize(
    "crop", [grid_from_detector_fov, grid_from_detector_fov_cube, grid_from_detector_fov_slices]
)
def test_fov_cropping_gives_the_same_dimensions_at_any_unit(scale, crop) -> None:
    grid = Grid(nx=11, ny=11, nz=11, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=9, nv=9, du=1.0, dv=1.0)
    scaled_grid, scaled_detector = _scale(grid, detector, scale)
    actual = crop(scaled_grid, scaled_detector)
    assert (actual.nx, actual.ny, actual.nz) == (9, 9, 9)
    assert actual.vx == scaled_grid.vx
    assert actual is not scaled_grid


@pytest.mark.parametrize("scale", [1e-30, 1e-6, 1.0, 1e30])
def test_single_pixel_detector_cylinder_keeps_only_the_on_axis_voxel(scale) -> None:
    grid = Grid(nx=5, ny=5, nz=1, vx=scale, vy=scale, vz=scale)
    detector = Detector(nu=1, nv=1, du=scale, dv=scale)
    expected = np.zeros((5, 5), bool)
    expected[2, 2] = True
    np.testing.assert_array_equal(cylindrical_mask_xy(grid, detector), expected)


@pytest.mark.parametrize("scale", [1e-30, 1e-6, 1.0, 1e30])
def test_roi_count_does_not_discard_voxels_on_rounding_boundaries(scale) -> None:
    # 7*0.1/0.14 is exactly 5 mathematically, across all physical unit choices.
    grid = Grid(nx=13, ny=13, nz=3, vx=0.14 * scale, vy=0.14 * scale, vz=scale)
    detector = Detector(nu=8, nv=3, du=0.1 * scale, dv=scale)
    info = compute_roi(grid, detector)
    assert (info.nx_roi, info.ny_roi, info.nz_roi) == (6, 6, 3)


def test_cylinder_tolerance_does_not_grow_with_unused_voxel_spacing() -> None:
    grid = Grid(nx=1, ny=1, nz=1, vx=1e30, vy=1e30, vz=1e30, vol_center=(1e10, 0.0, 0.0))
    detector = Detector(nu=3, nv=1, du=1.0, dv=1.0)
    np.testing.assert_array_equal(cylindrical_mask_xy(grid, detector), [[False]])


@pytest.mark.parametrize("scale", [1e-30, 1e-6, 1.0, 1e30])
def test_cylinder_includes_voxels_exactly_on_a_diagonal_boundary(scale) -> None:
    grid = Grid(nx=11, ny=11, nz=1, vx=scale, vy=scale, vz=scale)
    detector = Detector(nu=11, nv=1, du=scale, dv=scale)
    coords = np.arange(11) - 5
    expected = coords[:, None] ** 2 + coords[None, :] ** 2 <= 25
    np.testing.assert_array_equal(cylindrical_mask_xy(grid, detector), expected)


@pytest.mark.parametrize(
    "crop", [grid_from_detector_fov, grid_from_detector_fov_cube, grid_from_detector_fov_slices]
)
def test_roi_keeps_placement_only_when_shape_is_unchanged(crop):
    from tomojax.geometry import grid_volume_origin

    # The grid sits off the axis in-plane but inside the detector's rows axially.
    grid = Grid(9, 9, 9, 0.5, 0.5, 0.5, vol_origin=(2.0, 3.0, -2.0))
    assert crop(grid, Detector(11, 11, 0.5, 0.5)) is grid
    cropped = crop(grid, Detector(5, 5, 0.5, 0.5))
    assert (cropped.nx, cropped.ny, cropped.nz) == (5, 5, 5)
    origin = np.asarray(grid_volume_origin(cropped))
    np.testing.assert_allclose(origin[:2] + 2 * 0.5, 0.0)  # in-plane centred on the axis


@pytest.mark.parametrize(
    "crop", [grid_from_detector_fov, grid_from_detector_fov_cube, grid_from_detector_fov_slices]
)
def test_axial_roi_follows_off_centre_detector_rows(crop):
    from tomojax.geometry import grid_volume_origin

    grid = Grid(12, 12, 10, 1, 1, 1, vol_center=(0, 0, 1))
    detector = Detector(12, 2, 1, 1, center=(0, 2))
    result = crop(grid, detector)
    assert result.nz == 2 and result.vz == 1
    assert grid_volume_origin(result)[2] == 1.5


def _scan(grid, detector, geometry=None):
    import tomojax as tj

    if geometry is None:
        geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 60, endpoint=False))
    views = len(geometry.angles)
    return tj.Scan(np.zeros((views, detector.nv, detector.nu), np.float32), geometry)


@pytest.mark.parametrize("shape", [(9, 7, 3), (5, 4, 3), (7, 9, 3)])
def test_automatic_roi_keeps_a_rectangular_grid_the_detector_covers(shape):
    # check-public-imports: allow-private
    from tomojax.cli._reconstruction_region import region_grid

    grid = Grid(*shape, 0.7, 1.1, 1.9, vol_origin=(2, 3, 4))
    assert region_grid(_scan(grid, Detector(101, 101, 1, 1)), roi="auto", grid=None)[0] == grid


def test_automatic_roi_still_crops_what_the_detector_cannot_see():
    # check-public-imports: allow-private
    from tomojax.cli._reconstruction_region import region_grid

    cropped, _ = region_grid(
        _scan(Grid(11, 11, 5, 1, 1, 1), Detector(9, 3, 1, 1)), roi="auto", grid=None
    )
    assert (cropped.nx, cropped.ny, cropped.nz) == (9, 9, 3)


def test_cone_roi_measures_the_field_in_object_lengths():
    # check-public-imports: allow-private
    from tomojax.cli._reconstruction_region import region_grid
    from tomojax.geometry import ConeBeam, ConeGeometry

    grid, detector = Grid(41, 41, 31, 1, 1, 1), Detector(40, 20, 1, 1)
    # Magnification 2 halves the detector's reach in the object: about 9.5 x 4.5.
    angles = np.linspace(0, 360, 90, endpoint=False)
    scan = _scan(grid, detector, ConeGeometry(grid, detector, angles, ConeBeam(100, 200)))
    cropped, _ = region_grid(scan, roi="bbox", grid=None)
    assert 17 <= cropped.nx <= 21 and 17 <= cropped.ny <= 21 and 7 <= cropped.nz <= 10


def test_laminography_roi_keeps_voxels_the_tilted_rays_measure():
    # check-public-imports: allow-private
    from tomojax.cli._reconstruction_region import region_grid
    from tomojax.geometry import LaminographyGeometry, grid_volume_origin

    grid, detector = Grid(9, 9, 21, 1, 1, 1), Detector(30, 10, 1, 1)
    angles = np.linspace(0, 360, 90, endpoint=False)
    scan = _scan(grid, detector, LaminographyGeometry(grid, detector, angles, tilt_deg=60))
    cropped, _ = region_grid(scan, roi="bbox", grid=None)
    # The axis point at height 8 projects to detector v = 4, inside the 10 rows.
    top = grid_volume_origin(cropped)[2] + (cropped.nz - 1) * cropped.vz
    assert top >= 8
