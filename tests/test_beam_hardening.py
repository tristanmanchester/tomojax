"""Beam hardening: the physical curve, its coefficient sources and the joint fit."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.core import hardening
from tomojax.corrections import BeamHardening, tube_spectrum
from tomojax.recon import PolychromaticConfig

# A broad spectrum: the slowest component is a tenth of the mean rate.
BROAD = BeamHardening(rates=(0.15, 0.4, 0.9, 1.8, 4.0), weights=(0.1, 0.25, 0.35, 0.2, 0.1))


def test_mixture_is_gauged_and_its_inverse_is_float32_exact() -> None:
    assert sum(BROAD.weights) == pytest.approx(1.0)
    assert np.dot(BROAD.rates, BROAD.weights) == pytest.approx(1.0)
    p = np.linspace(-0.3, 80.0, 40001)
    q = BROAD.curve(p)
    assert np.all(np.diff(q) > 0) and np.all(np.diff(q, 2) < 1e-12)  # increasing, concave
    linear = BROAD.apply(jnp.asarray(q, jnp.float32))
    exact = BROAD.inverse(q.astype(np.float32).astype(np.float64))
    np.testing.assert_allclose(np.asarray(linear), exact, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(exact, p, rtol=1e-6, atol=1e-6)
    flat = jnp.asarray([[[0.5, np.inf]]], jnp.float32)
    assert np.asarray(BeamHardening().apply(flat)).tolist() == [[[0.5, np.inf]]]
    referenced = BeamHardening(rates=(2.0, 0.5), weights=(1, 3), energy_kev=60.0)
    assert referenced.rates == (0.5, 2.0) and referenced.weights == (0.75, 0.25)
    # Paths from 3 up hide BROAD's fastest component: the fit uses no rate over 3 / 3.
    p = np.linspace(3.0, 20.0, 400)
    rates, weights, _ = hardening.fit(p, BROAD.curve(p))
    assert max(rates) <= 1.0 + 1e-9
    np.testing.assert_allclose(hardening.curve(rates, weights, p), BROAD.curve(p), atol=0.01)


def test_polynomials_become_the_closest_physical_curve() -> None:
    quadratic = BeamHardening.from_polynomial((0.0, 1.0, 0.05), upto=4.0)
    q = np.linspace(0.0, 4.0, 9)
    np.testing.assert_allclose(quadratic.inverse(q), q + 0.05 * q**2, rtol=2e-3, atol=1e-3)
    # A cubic with no linear term is not a physical inverse; its fit still is one.
    cubic = BeamHardening.from_polynomial((0.0, 0.0, 0.2, 0.8), upto=3.0)
    assert np.all(np.diff(cubic.curve(np.linspace(0, 20, 201)), 2) < 1e-12)


def _phantom(
    curve: BeamHardening, *, n: int = 48, seed: int = 0, insert: bool = False, rod: bool = False
) -> tuple[tj.Scan, np.ndarray]:
    rng = np.random.default_rng(seed)
    grid = tj.Grid(n, n, 1, 1.0, 1.0, 1.0)
    detector = tj.Detector(n + n // 2, 1, 1.0, 1.0)
    views = np.linspace(0, 180, 4 * n if rod else n + n // 2, endpoint=False)
    geometry = tj.ParallelGeometry(grid, detector, views)
    y, x = np.mgrid[:n, :n] - (n - 1) / 2
    r = n / 2
    shape = (x / (0.8 * r)) ** 2 + (y / (0.5 * r)) ** 2 < 1
    shape |= (x - 0.45 * r) ** 2 + (y + 0.55 * r) ** 2 < (0.25 * r) ** 2
    shape &= (x - 0.2 * r) ** 2 + (y - 0.1 * r) ** 2 > (0.1 * r) ** 2
    truth = (shape * 6.0 / n).astype(np.float32)[:, :, None]
    if insert:  # a denser phase of the same material
        truth[(x + 0.4 * r) ** 2 + y**2 < (0.18 * r) ** 2] *= 2.5
    if rod:  # a thin rod 30 times denser, too rare to be a level
        truth[(x - 0.1 * r) ** 2 + (y + 0.25 * r) ** 2 < 6] = 180.0 / n
    q = curve.curve(np.asarray(tj.project(geometry, truth)))
    q = -np.log(rng.poisson(5e4 * np.exp(-q)) / 5e4)
    return tj.Scan(q.astype(np.float32), geometry), truth


def test_self_calibration_recovers_the_curve_and_leaves_linear_data_linear() -> None:
    scan, truth = _phantom(BROAD)
    q = np.linspace(0.0, float(np.max(scan.projections)), 12)
    fitted = BeamHardening.fit(scan)
    np.testing.assert_allclose(fitted.inverse(q), BROAD.inverse(q), rtol=0.03, atol=0.02)
    image = np.asarray(tj.reconstruct(scan.corrected(fitted), "fbp").volume)
    inside = truth > 0
    assert abs(image[inside].mean() / truth[inside].mean() - 1) < 0.05
    mono, _ = _phantom(BeamHardening())
    linear = BeamHardening.fit(mono)
    q = np.linspace(0.0, float(np.max(mono.projections)), 12)
    np.testing.assert_allclose(linear.inverse(q), q, rtol=0.02, atol=0.02)


def test_self_calibration_takes_a_second_material_as_a_level_of_its_own() -> None:
    scan, _ = _phantom(BROAD, n=64, insert=True)
    q = np.linspace(0.0, float(np.percentile(scan.projections, 99)), 12)
    fitted = BeamHardening.fit(scan)
    np.testing.assert_allclose(fitted.inverse(q), BROAD.inverse(q), rtol=0.04, atol=0.02)
    rodded, _ = _phantom(BROAD, n=64, insert=True, rod=True)
    fitted = BeamHardening.fit(rodded)
    np.testing.assert_allclose(fitted.inverse(q), BROAD.inverse(q), rtol=0.1, atol=0.03)
    counts = np.random.default_rng(1).poisson(5e4, scan.projections.shape)
    air = tj.Scan(-np.log(counts / 5e4).astype(np.float32), scan.geometry)
    with pytest.raises(ValueError, match="stands out"):
        BeamHardening.fit(air)


def test_joint_fit_lowers_the_misfit_and_keeps_the_curve() -> None:
    scan, truth = _phantom(BROAD)
    config = PolychromaticConfig(iterations=3, inner_iterations=6)
    result = tj.reconstruct(scan, "polychromatic", config=config)
    misfit = result.info["misfit"]
    assert misfit[-1] < misfit[0]
    joint = BeamHardening(**result.info["beam_hardening"])
    q = np.linspace(0.0, float(np.max(scan.projections)), 12)
    np.testing.assert_allclose(joint.inverse(q), BROAD.inverse(q), rtol=0.03, atol=0.02)
    inside = truth > 0
    volume = np.asarray(result.volume)
    assert abs(volume[inside].mean() / truth[inside].mean() - 1) < 0.05
    with pytest.raises(ValueError, match="rates and weights together"):
        PolychromaticConfig(rates=(1.0,))


def test_spectrum_curves_use_the_material_attenuation() -> None:
    xraylib = pytest.importorskip("xraylib")
    energies, weights = tube_spectrum(80.0, filters=[("Al", 1.0)], detector=("CsI", 0.2, 4.51))
    assert weights.sum() == pytest.approx(1.0) and energies.max() < 80.0
    hard = tube_spectrum(80.0, filters=[("Al", 1.0), ("Cu", 0.5)])
    assert np.dot(*hard) > np.dot(energies, weights)
    step = BeamHardening.from_spectrum([30.0, 60.0], [0.5, 0.5], "Al", energy_kev=60.0)
    ratio = xraylib.CS_Total_CP("Al", 30.0) / xraylib.CS_Total_CP("Al", 60.0)
    assert step.rates == pytest.approx((1.0, ratio)) and step.energy_kev == 60.0
    assert BeamHardening.from_spectrum([50.0], [1.0], "H2O", energy_kev=50.0).rates == (1.0,)


def test_nikon_records_its_linearisation_and_recorded_applies_it(tmp_path) -> None:
    import imageio.v3 as iio

    for i in range(2):
        iio.imwrite(tmp_path / f"part_{i + 1:04d}.tif", np.full((2, 3), 20000, np.uint16))
    (tmp_path / "part.xtekct").write_text(
        "[XTekCT]\nName=part\nSrcToObject=12\nSrcToDetector=18\nDetectorPixelSizeX=0.1\n"
        "DetectorPixelSizeY=0.1\nDetectorOffsetX=0\nDetectorOffsetY=0\nCentreOfRotationTop=0\n"
        "CentreOfRotationBottom=0\nWhiteLevel=60000\nInitialAngle=0\nAngularStep=90\n"
        "CoefX4=0\nCoefX3=0\nCoefX2=0.25\nCoefX1=0.75\nCoefX0=0\n[Xrays]\nXraykV=100\n"
    )
    (tmp_path / "part.ctprofile.xml").write_text(
        "<a><BeamHardeningPreset>2</BeamHardeningPreset></a>"
    )
    scan = tj.load_frames(tmp_path / "part.xtekct", darks=0).corrected()
    settings = scan.source.geometry_metadata["beam_hardening"]
    assert settings == {
        "vendor": "nikon", "coefficients": [0, 0.75, 0.25, 0, 0], "preset": 2, "kv": 100.0
    }  # fmt: skip
    q = float(np.max(scan.projections))
    step = BeamHardening.recorded(scan)
    preset = 0.75 * q + 0.25 * q**2
    assert step.inverse(q) == pytest.approx(preset / 0.75, rel=2e-3)  # in unit initial slope
    settings["vendor"] = "zeiss"
    with pytest.raises(ValueError, match="not published"):
        BeamHardening.recorded(scan)
