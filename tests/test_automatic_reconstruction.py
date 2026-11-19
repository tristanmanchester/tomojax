"""Automatic reconstruction selects a measured inverse without changing geometry."""

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj
from tomojax.alignment import api as alignment
from tomojax.corrections import Rings, Stripes
from tomojax.geometry import RotationAxisGeometry
from tomojax.io import ProjectionDataset
from tomojax.recon import FBPConfig


@pytest.mark.parametrize(
    ("complete", "raw"),
    [(True, False), (False, False), (True, True), (True, "rings"), (True, "stripes")],
)
def test_default_reconstruction_matches_explicit_inverse_and_reports_coverage(
    complete, raw, tmp_path
):
    grid = tj.Grid(12, 14, 2, 1, 1, 1)
    detector = tj.Detector(20, 2, 1, 1, center=(1.3, 0))
    angles = np.linspace(0, 180 if complete else 90, 32, endpoint=False)
    geometry = tj.ParallelGeometry(grid, detector, angles)
    u = np.arange(detector.nu) - (detector.nu - 1) / 2 + detector.center[0]
    data = np.broadcast_to(np.exp(-(u**2) / 10), (32, 2, 20)).astype(np.float32).copy()
    steps = {"rings": (Rings(geometry=geometry),), "stripes": (Stripes(),)}.get(raw, ())
    scan = (
        tj.Frames(np.exp(-data), geometry, white_level=1).corrected(*steps)
        if raw
        else tj.Scan(data, geometry)
    )
    expected = tj.reconstruct(scan, "fourier" if complete else "fbp")
    actual = tj.reconstruct(scan)
    np.testing.assert_array_equal(actual.volume, expected.volume)
    assert actual.scan is scan and actual.method == expected.method
    limited = bool(raw) or not complete
    assert actual.info["automatic"]["status"] == ("limited" if limited else "ok")
    assert bool(actual.info["automatic"]["reasons"]) == limited
    if raw:
        reasons = actual.info["automatic"]["reasons"]
        assert any("stripe correction" in reason for reason in reasons) == (raw is True)
    assert actual.info["automatic"]["center_status"] is None  # nothing recorded it either way
    tj.save(tmp_path / "reconstruction.nxs", actual)
    loaded = tj.load_reconstruction(tmp_path / "reconstruction.nxs")
    assert loaded.info["automatic"] == actual.info["automatic"]


def test_explicit_method_bypasses_automatic_geometry_policy():
    grid = tj.Grid(4, 4, 2, 1, 1, 1)
    detector = tj.Detector(8, 3, 1, 1)
    geometry = tj.LaminographyGeometry(grid, detector, np.arange(0, 360, 30), tilt=30)
    scan = tj.Scan(np.ones((12, 3, 8), np.float32), geometry)
    result = tj.reconstruct(scan, "fbp", config=FBPConfig())
    assert "automatic" not in result.info and result.scan is scan
    with pytest.raises(ValueError, match="method 'fbp' does not take iterations"):
        tj.reconstruct(scan, iterations=3)


@pytest.mark.parametrize("status", [None, "recorded", "provided", "inferred", "not_recorded"])
def test_automatic_reconstruction_never_estimates_a_center(status, monkeypatch):
    # tj.reconstruct reconstructs with the centre it is given, whatever the file recorded.
    def no_estimate(*_args, **_kwargs):
        raise AssertionError("tj.reconstruct estimated a centre")

    monkeypatch.setattr(alignment, "estimate_center_report", no_estimate)
    grid, detector = tj.Grid(8, 8, 2, 1, 1, 1), tj.Detector(12, 2, 1, 1, center=(-2.2, 0))
    angles = np.linspace(0, 180, 33)
    geometry = tj.ParallelGeometry(grid, detector, angles)
    scan = tj.Frames(np.full((33, 2, 12), 0.5, np.float32), geometry, white_level=1).corrected()
    meta = {} if status is None else {"detector_center_status": status}
    record = ProjectionDataset(scan.projections, angles, geometry_metadata=meta)
    scan = replace(scan, source=record)
    actual = tj.reconstruct(scan)
    assert actual.scan is scan and actual.scan.detector.center == (-2.2, 0)
    assert actual.info["automatic"]["geometry"] == "preserved"
    assert actual.info["automatic"]["center_status"] == status
    assert record.geometry_metadata == meta


def test_automatic_segmented_cone_preserves_calibration():
    from tomojax.geometry import ConeBeam, ConeGeometry, ConeSegments

    grid, detector = tj.Grid(4, 4, 3, 1, 1, 1), tj.Detector(8, 5, 1, 1)
    geometry = ConeSegments(
        tuple(
            ConeGeometry(grid, replace(detector, center=(offset, 0.2)), [angle], ConeBeam(20, 30))
            for offset, angle in zip([-0.4, 0.7], [0, 90], strict=True)
        )
    )
    scan = tj.Scan(np.ones((2, 5, 8), np.float32), geometry)
    expected = tj.reconstruct(scan, "cgls", iterations=2)
    actual = tj.reconstruct(scan, iterations=2)
    np.testing.assert_allclose(actual.volume, expected.volume, rtol=2e-6, atol=1e-7)
    assert actual.scan is scan and actual.method == "cgls"
    assert actual.info["automatic"]["status"] == "limited"
    assert "Segmented cone" in actual.info["automatic"]["reasons"][0]


def _phantom(n=16):
    # An off-centre Gaussian, so a wrong centre or ignored poses move it visibly.
    xyz = np.indices((n, n, n)).astype(float)
    center = np.array([7.5, 6.5, 8.5])[:, None, None, None]
    return np.exp(-np.sum((xyz - center) ** 2, axis=0) / 10).astype(np.float32)


def _error(result, volume):
    return np.linalg.norm(np.asarray(result.volume) - volume) / np.linalg.norm(volume)


_ANGLES = np.linspace(0, 360, 128, endpoint=False)
_GRID = tj.Grid(16, 16, 16, 1, 1, 1)


def _moved(geometry, volume):
    """Data from known per-view shifts, and those shifts as the scan's pose corrections."""
    turn = np.deg2rad(_ANGLES)
    shifts = np.zeros((len(_ANGLES), 6))
    shifts[:, 3], shifts[:, 4] = 1.5 * np.sin(2 * turn), np.cos(turn)
    blank = tj.Scan(
        np.zeros((len(_ANGLES), geometry.detector.nv, geometry.detector.nu), np.float32), geometry
    )
    data = np.asarray(tj.project(blank.with_poses(shifts).geometry, volume))
    return tj.Scan(data, geometry).with_poses(shifts)


_ROUTES = {
    "parallel": lambda: tj.ParallelGeometry(_GRID, tj.Detector(24, 24, 1, 1), _ANGLES),
    "cone": lambda: tj.ConeGeometry(
        _GRID, tj.Detector(24, 24, 1.5, 1.5), _ANGLES, tj.ConeBeam(192, 288)
    ),
    "axis": lambda: RotationAxisGeometry(
        _GRID, tj.Detector(24, 24, 1, 1), _ANGLES, (0.2, -0.3, 0.93)
    ),
}


def test_automatic_route_reconstructs_a_plain_cone_scan():
    volume = _phantom()
    geometry = _ROUTES["cone"]()
    result = tj.reconstruct(tj.Scan(np.asarray(tj.project(geometry, volume)), geometry))
    assert result.method == "fbp" and result.info["automatic"]["status"] == "ok"
    assert _error(result, volume) < 0.08  # FDK at 0.038 measured


@pytest.mark.parametrize(
    ("kind", "method", "geometry", "bound"),
    [
        ("parallel", "fbp", "preserved", 0.1),  # 0.053 measured, 0.23 with the poses ignored
        ("cone", "fbp", "preserved", 0.1),  # 0.053 measured, 0.23 with the poses ignored
        # A regression guard for FBP, which leaves the cone empty: 0.151 measured, 0.30 ignored.
        ("axis", "fbp", "preserved", 0.2),
    ],
)
def test_automatic_route_uses_a_scans_poses(kind, method, geometry, bound):
    volume = _phantom()
    result = tj.reconstruct(_moved(_ROUTES[kind](), volume))
    assert result.method == method
    assert result.info["automatic"]["geometry"] == geometry
    assert _error(result, volume) < bound


@pytest.mark.parametrize("kind", ["laminography", "axis"])
def test_automatic_route_reconstructs_an_unposed_oblique_scan_with_its_declared_geometry(
    kind, monkeypatch
):
    # tj.reconstruct reconstructs with the geometry it is given: no pose fit, the scan's grid,
    # and the same image as the same scan with its declared poses written out.
    from tomojax import _smooth_alignment  # check-public-imports: allow-private

    def no_fit(*_args, **_kwargs):
        raise AssertionError("tj.reconstruct ran an alignment")

    monkeypatch.setattr(_smooth_alignment, "fit_poses", no_fit)
    volume = _phantom()
    geometry = (
        tj.LaminographyGeometry(_GRID, tj.Detector(24, 24, 1, 1), _ANGLES, tilt=30)
        if kind == "laminography"
        else _ROUTES["axis"]()
    )
    scan = tj.Scan(np.asarray(tj.project(geometry, volume)), geometry)
    result = tj.reconstruct(scan)
    assert result.method == "fbp" and result.scan is scan
    assert np.shape(result.volume) == volume.shape
    declared = tj.reconstruct(scan.with_poses(np.zeros((len(_ANGLES), 6))))
    np.testing.assert_allclose(result.volume, declared.volume, rtol=0, atol=1e-6)
    # A regression guard, not a correctness test (the comparison with the declared poses
    # above is the oracle): FBP leaves the tilted axis's missing cone empty, 0.300
    # (laminography) and 0.150 (axis) measured, where fista at 100 iterations reaches 0.009
    # and 0.003.
    assert _error(result, volume) < (0.4 if kind == "laminography" else 0.2)


def test_one_orbit_with_per_view_detector_offsets_reconstructs_with_fdk():
    # A Zeiss scan is one circular orbit whose recorded per-view shifts move only the detector;
    # the automatic rule tested the type (ConeSegments) and sent it to CGLS, 90 times slower on
    # CuO_2. FDK keeps each view's own arrangement exactly: with the offsets known it should
    # cost nothing but interpolation against the same orbit without them.
    from tomojax.geometry import ConeSegments

    volume = _phantom()
    detector, beam = tj.Detector(24, 24, 1.5, 1.5), tj.ConeBeam(192, 288)
    shift = np.random.default_rng(5).uniform(-1.5, 1.5, (len(_ANGLES), 2)) * 1.5
    offset = ConeSegments(
        tuple(
            tj.ConeGeometry(_GRID, replace(detector, center=tuple(s)), [a], beam)
            for s, a in zip(shift, _ANGLES, strict=True)
        ),
        trajectory_lengths=(len(_ANGLES),),
    )
    plain = tj.ConeGeometry(_GRID, detector, _ANGLES, beam)
    shifted = tj.reconstruct(tj.Scan(np.asarray(tj.project(offset, volume)), offset))
    unshifted = tj.reconstruct(tj.Scan(np.asarray(tj.project(plain, volume)), plain))
    assert shifted.method == "fbp" and shifted.info["automatic"]["status"] == "ok"
    # 0.0589 against 0.0377 measured. The same orbit with one shared detector offset of these
    # sizes (a plain ConeGeometry, no segments) gives 0.043-0.068: the extra is FDK's own
    # behaviour with an off-centre detector on a 16^3 case, which per-view offsets do not worsen.
    assert _error(shifted, volume) < 2 * _error(unshifted, volume)
    # Two orbits whose arrangements differ (a source at two heights) stay with CGLS.
    two = ConeSegments(
        (
            tj.ConeGeometry(_GRID, detector, _ANGLES[::2], beam),
            tj.ConeGeometry(_GRID, detector, _ANGLES[1::2], replace(beam, source_height=4.0)),
        )
    )
    blank = tj.Scan(np.zeros((len(_ANGLES), 24, 24), np.float32), two)
    assert tj.reconstruct(blank, iterations=1).method == "cgls"
