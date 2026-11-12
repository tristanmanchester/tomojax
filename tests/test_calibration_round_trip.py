"""A geometry built from what tj.align reports equals the geometry it fitted.

This is the property a user relies on when they copy a reported calibration into their own
script or into another program. Each case fits one global value on an analytic scan with a
known error, rebuilds the geometry from ``info["calibration"]`` through the public
constructors, and compares projections of a random volume through both.

Known gap: a reported roll on parallel or laminography data has no public constructor yet
(the fit builds a private wrapper), so those two cells are not covered here. A tilted parallel
axis is a laminography geometry with that tilt.
"""

from bench.fit_alignment import make_case
import numpy as np
import pytest

import tomojax as tj


def _fitted(case, geometry, name):
    scan, _, _, _ = make_case(case, geometry=geometry, size=24, views=30, grid_size=12)
    result = tj.align(scan, fit=(name,), grid=scan.grid)
    return scan, result.scan, result.info["calibration"][name]


def _same_projections(a, b):
    volume = np.random.default_rng(3).uniform(0, 1, (a.grid.nz, a.grid.ny, a.grid.nx))
    volume = volume.astype(np.float32)
    np.testing.assert_allclose(
        np.asarray(tj.project(a.geometry, volume)),
        np.asarray(tj.project(b.geometry, volume)),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("geometry", ["parallel", "lamino", "cone"])
def test_a_reported_centre_rebuilds_the_fitted_geometry(geometry):
    scan, fitted, centre = _fitted("centre", geometry, "centre")
    centre = tuple(centre) if geometry == "cone" else (centre, scan.detector.center[1])
    d = scan.detector
    detector = tj.Detector(d.nu, d.nv, d.du, d.dv, centre)
    if geometry == "cone":
        rebuilt = tj.ConeGeometry(scan.grid, detector, scan.angles, scan.geometry.beam)
    elif geometry == "lamino":
        rebuilt = tj.LaminographyGeometry(scan.grid, detector, scan.angles, tilt_deg=25.0)
    else:
        rebuilt = tj.ParallelGeometry(scan.grid, detector, scan.angles)
    _same_projections(fitted, tj.Scan(scan.projections, rebuilt))


def test_a_reported_cone_roll_rebuilds_the_fitted_geometry():
    scan, fitted, roll = _fitted("roll", "cone", "roll")
    beam = tj.ConeBeam(192, 288, detector_roll_deg=roll)
    rebuilt = tj.ConeGeometry(scan.grid, scan.detector, scan.angles, beam)
    _same_projections(fitted, tj.Scan(scan.projections, rebuilt))


@pytest.mark.parametrize("geometry", ["parallel", "lamino", "cone"])
def test_a_reported_tilt_rebuilds_the_fitted_geometry(geometry):
    scan, fitted, tilt = _fitted("tilt", geometry, "tilt")
    # The reported tilt is the axis's signed inclination atan2(axis_y, axis_z).
    radians = np.deg2rad(tilt)
    axis = (0.0, float(np.sin(radians)), float(np.cos(radians)))
    if geometry == "cone":
        rebuilt = tj.ConeGeometry(
            scan.grid, scan.detector, scan.angles, scan.geometry.beam, axis_unit=axis
        )
    else:
        rebuilt = tj.LaminographyGeometry(scan.grid, scan.detector, scan.angles, tilt_deg=tilt)
    _same_projections(fitted, tj.Scan(scan.projections, rebuilt))
