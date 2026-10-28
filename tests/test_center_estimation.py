"""Projection-only centre estimation returns the scan with its physical detector-u centre."""

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj
from tomojax.geometry import Detector, Grid, ParallelGeometry


def _scan(*, pitch=0.8, views=180, span=180):
    angles = 350 + np.linspace(0, span, views, endpoint=False)
    detector = Detector(64, 8, pitch, 1.3, (2.7, -4.2))
    grid = Grid(80, 80, 8, pitch, pitch, 1.3, vol_center=(0, 0, -4.2))
    true_u = -5.25 * pitch
    u = (np.arange(64) - 31.5) * pitch + true_u
    theta = np.deg2rad(angles)
    sino = np.zeros((views, 64))
    truth = np.zeros((80, 80))
    x, y = np.meshgrid(
        (np.arange(80) - 39.5) * pitch, (np.arange(80) - 39.5) * pitch, indexing="xy"
    )
    components = np.asarray([(5, 2, 3, 1), (-6, -3, 2, 0.7), (0, 9, 1.2, 0.4)])
    # Change physical units, not sampling bandwidth, with the pitch.
    components[:, :3] = components[:, :3] * pitch / 0.8
    for cx, cy, r, amplitude in components:
        projected = np.cos(theta) * cx - np.sin(theta) * cy
        sino += (
            amplitude * np.sqrt(2 * np.pi) * r * np.exp(-0.5 * ((u - projected[:, None]) / r) ** 2)
        )
        truth += amplitude * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / r**2)
    data = np.broadcast_to(sino[:, None], (views, 8, 64)).astype(np.float32).copy()
    return (
        tj.Scan(data, ParallelGeometry(grid, detector, angles), name="centre fixture"),
        true_u,
        truth,
    )


def test_encoder_variations_do_not_concentrate_centre_pairs_in_one_sector():
    from tomojax.alignment.api import estimate_center

    scan, true_u, _ = _scan(views=360, span=360)
    angles = scan.angles.copy()
    angles[180:] += np.linspace(0, 0.005, 180)
    selected = []

    class Rows:
        shape = scan.projections.shape

        def __getitem__(self, key):
            selected.extend(key[0].ravel())
            return scan.projections[key]

    result = estimate_center(replace(scan.geometry, angles=angles), Rows())
    directions = np.sort(angles[selected] % 180)
    assert np.max(np.diff(np.r_[directions, directions[0] + 180])) < 20
    assert abs(result.detector.center[0] - true_u) < 0.3 * scan.detector.du


def test_estimate_center_returns_the_scan_with_its_physical_centre():
    scan, true_u, truth = _scan()
    result = tj.estimate_center(scan)
    assert result.projections is scan.projections
    assert result.source is None
    assert result.grid is scan.grid
    assert result.detector.center[1] == scan.detector.center[1]
    assert abs(result.detector.center[0] - true_u) <= 0.3 * scan.detector.du
    assert scan.detector.center[0] == 2.7
    volume = np.asarray(tj.reconstruct(result).volume)
    assert volume.shape == (scan.grid.nz, scan.grid.ny, scan.grid.nx)
    assert np.linalg.norm(volume[4] - truth) / np.linalg.norm(truth) < 0.025


def test_estimate_center_after_crop_and_bin_keeps_absolute_physical_units():
    scan, true_u, _ = _scan()
    known = replace(
        scan,
        geometry=replace(scan.geometry, detector=replace(scan.detector, center=(true_u, -4.2))),
    )
    for factor in (1, 2):
        preview = scan.cropped(slice(1, 7), slice(2, 61)).binned(factor)
        expected = known.cropped(slice(1, 7), slice(2, 61)).binned(factor)
        result = tj.estimate_center(preview)
        assert (
            abs(result.detector.center[0] - expected.detector.center[0])
            <= 0.4 * preview.detector.du
        )


def test_estimate_center_orders_wrapped_repeated_angles_by_direction():
    scan, true_u, _ = _scan()
    indices = np.repeat(np.arange(179, -1, -1), 2)
    angles = scan.angles[indices] % 360
    angles += 360 * np.random.default_rng(8).integers(-2, 3, len(indices))
    scan = replace(
        scan, projections=scan.projections[indices], geometry=replace(scan.geometry, angles=angles)
    )
    result = tj.estimate_center(scan)
    assert abs(result.detector.center[0] - true_u) <= 0.3 * scan.detector.du


@pytest.mark.parametrize("kind", ["short_arc", "cone", "sparse_views"])
def test_estimate_center_rejects_scans_without_usable_axial_pairs(kind):
    scan, _, _ = _scan()
    if kind == "short_arc":
        scan = scan.selected(slice(0, 100))
    elif kind == "sparse_views":
        scan, _, _ = _scan(views=125, span=360)
    else:
        geometry = tj.ConeGeometry(scan.grid, scan.detector, scan.angles, tj.ConeBeam(100, 200))
        scan = replace(scan, geometry=geometry)
    message = "angular neighbours" if kind == "sparse_views" else "estimate_center"
    with pytest.raises(ValueError, match=message):
        tj.estimate_center(scan)


def test_auto_alignment_records_center_and_resolves_only_its_limitation(monkeypatch):
    from tomojax import _workflow

    scan, true_u, _ = _scan()
    context = dict(
        status="limited", reasons=["centre", "rings"], center_calibration_reason="centre"
    )
    rec = tj.Reconstruction(np.zeros((80, 80, 8)), scan, "fbp", {"automatic": context})
    monkeypatch.setattr(_workflow, "reconstruct", lambda s: replace(rec, scan=s))
    info = tj.align(scan).info["automatic"]
    assert info["reconstruction"] == context
    assert info["reasons"][-1] == "rings" and info["status"] == "limited"
    assert "not a calibrated centre interval" in info["reasons"][0]
    center = info["center_estimation"]
    assert center["input"] == list(scan.detector.center) and center["status"] == "estimated"
    assert center["diagnostics"]["pairs"][0]["local_matching_sigma_px"] >= 0
    assert abs(center["output"][0] - true_u) <= 0.3 * scan.detector.du


def test_loading_estimates_missing_center_from_recorded_pitch_and_preserves_it(tmp_path, caplog):
    import h5py

    scan, true_u, _ = _scan()
    path = tmp_path / "external.data"
    counts = 10000 * np.exp(-0.1 * scan.projections)
    with h5py.File(path, "w") as file:
        file["exchange/data"] = counts
        file["exchange/data_white"] = np.full((1, 8, 64), 10000, np.float32)
        file["exchange/data_dark"] = np.zeros((1, 8, 64), np.float32)
        file["exchange/theta"] = scan.angles
        for name, pitch in (("x", scan.detector.du), ("y", scan.detector.dv)):
            data = file.create_dataset(
                "measurement/instrument/detector/actual_pixel_size_" + name, data=pitch
            )
            data.attrs["units"] = "mm"
    frames = tj.load_frames(path)
    assert abs(frames.geometry.detector.center[0] - true_u) <= 0.3 * scan.detector.du
    assert frames.source.geometry_metadata["detector_center_status"] == "inferred"
    diagnostics = frames.source.geometry_metadata["detector_center_calibration"]["diagnostics"]
    assert diagnostics["global_coverage_calibrated"] is False
    assert "not a calibrated centre interval" in caplog.text
    assert tj.load(path).detector == frames.geometry.detector
    result = frames.cropped(slice(2, 4), slice(None)).corrected()
    expected = -np.log(counts[:, 2:4].astype(np.float64) / 10000)
    np.testing.assert_allclose(result.projections, expected, atol=2e-7)
    saved = tmp_path / "calibrated.nxs"
    tj.save(saved, result)
    loaded = tj.load(saved)
    assert loaded.detector == result.detector
    assert loaded.source.geometry_metadata["detector_center_status"] == "inferred"


def test_estimate_center_retains_signed_phase_contrast_across_rows():
    from scipy.ndimage import gaussian_filter1d

    scan, true_u, _ = _scan(views=181)
    # Signed edge contrast can cancel across rows; its matched correlations cannot.
    phase = gaussian_filter1d(scan.projections, 1, axis=-1, order=2)
    phase *= np.asarray([1, -1] * 4)[None, :, None]
    phase += np.random.default_rng(17).normal(0, 0.015, phase.shape)
    phase += np.sin(np.deg2rad(scan.angles))[:, None, None]
    result = tj.estimate_center(replace(scan, projections=phase))
    assert abs(result.detector.center[0] - true_u) < 0.15 * scan.detector.du


def test_center_resolves_subpixel_peaks_of_sharp_analytic_projections():
    from tomojax.alignment.api import estimate_center

    detector = Detector(128, 1, 1, 1)
    geometry = ParallelGeometry(Grid(128, 128, 1, 1, 1, 1), detector, [0, 180])
    for center in np.linspace(-5.4, 5.4, 21):
        u = np.arange(128) - 63.5 + center
        data = np.asarray(
            [
                sum(
                    a * np.exp(-0.5 * (u - sign * x) ** 2)
                    for x, a in [(-19, 1), (-3, 0.8), (27, 0.4)]
                )
                for sign in (1, -1)
            ]
        )[:, None]
        assert abs(estimate_center(geometry, data).detector.center[0] - center) < 0.001


def test_center_report_keeps_opposing_constraints_and_exposes_motion():
    from scipy.ndimage import shift

    from tomojax.alignment.api import estimate_center_report

    scan, truth, _ = _scan(views=360, span=360)
    data = scan.projections.copy()
    for i in range(len(data)):
        data[i] = shift(data[i], (0, 2 if i % 180 < 90 else -2), order=3)
    report = estimate_center_report(scan.geometry, data)
    assert len(report.pairs) == 16
    assert report.pair_scatter_px > 1.5
    assert report.consistency_pvalue < 0.01
    assert abs(report.geometry.detector.center[0] - truth) < 0.1
    assert all(abs(sum(w for _, w in p.first) - 1) < 1e-10 for p in report.pairs)
    assert all(abs(sum(w for _, w in p.second) - 1) < 1e-10 for p in report.pairs)
    assert len(report.to_dict()["pairs"]) == 16


def test_center_report_measures_row_trend_and_conditional_interval():
    from scipy.ndimage import shift

    from tomojax.alignment.api import estimate_center_report

    scan, truth, _ = _scan(views=360, span=360)
    data = scan.projections.copy()
    for row in range(data.shape[1]):
        data[:, row] = shift(data[:, row], (0, 0.03 * (row - 3.5)), order=3)
    data += np.random.default_rng(304).normal(0, 0.01, data.shape)
    report = estimate_center_report(scan.geometry, data)
    assert abs(report.row_slope_px_per_row + 0.03) < 0.003
    assert abs(report.geometry.detector.center[0] - truth) < 0.03
    assert all(p.local_sigma_px > 0 for p in report.pairs)
    assert all(
        p.local_interval90_px[0] < p.offset_px < p.local_interval90_px[1] for p in report.pairs
    )


def test_single_pair_reports_uncalibrated_global_coverage_without_inconsistency(monkeypatch):
    from tomojax import _workflow

    scan, _, _ = _scan(views=2, span=360)
    reconstruction = tj.Reconstruction(np.zeros((80, 80, 8)), scan, "fbp", {})
    monkeypatch.setattr(_workflow, "reconstruct", lambda s: replace(reconstruction, scan=s))
    automatic = tj.align(scan).info["automatic"]
    diagnostics = automatic["center_estimation"]["diagnostics"]
    assert diagnostics["consistency_pvalue"] is None
    assert diagnostics["global_coverage_calibrated"] is False
    assert diagnostics["uncertainty_status"] == "local_matching_only"
    assert "local_matching_interval90_px" in diagnostics["pairs"][0]
    assert automatic["status"] == "limited"
    assert "not a calibrated centre interval" in automatic["reasons"][0]
