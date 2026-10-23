"""Beam hardening: the physical curve and its coefficient sources."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.core import hardening
from tomojax.corrections import BeamHardening, tube_spectrum

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


@pytest.mark.parametrize("polynomial", [True, False])
def test_nikon_records_unapplied_hardening_and_tube_settings(tmp_path, polynomial) -> None:
    import imageio.v3 as iio

    for i in range(2):
        iio.imwrite(tmp_path / f"part_{i + 1:04d}.tif", np.full((2, 3), 20000, np.uint16))
    (tmp_path / "part.xtekct").write_text(
        "[XTekCT]\nName=part\nSrcToObject=12\nSrcToDetector=18\nDetectorPixelSizeX=0.1\n"
        "DetectorPixelSizeY=0.1\nDetectorOffsetX=0\nDetectorOffsetY=0\nCentreOfRotationTop=0\n"
        "CentreOfRotationBottom=0\nWhiteLevel=60000\nInitialAngle=0\nAngularStep=90\n"
        "CoefX4=0\nCoefX3=0\nCoefX2=0.25\nCoefX1=0.75\nCoefX0=0\nScale=139\nScaling=1000\n"
        "[Xrays]\nXraykV=100\n"
    )
    if not polynomial:
        path = tmp_path / "part.xtekct"
        path.write_text(path.read_text().replace("CoefX1=0.75", "CoefX1="))
    (tmp_path / "part.ctprofile.xml").write_text(
        "<a><BeamHardeningPreset>2</BeamHardeningPreset></a>"
    )
    scan = tj.load_frames(tmp_path / "part.xtekct", darks=0).corrected()
    assert scan.source.geometry_metadata["nikon_xtekct"]["parameters"]["Scale"] == "139"
    settings = scan.source.geometry_metadata["beam_hardening"]
    assert settings == {
        "vendor": "nikon", "scale": 139, "scaling": 1000, "preset": 2, "kv": 100.0,
        **({"coefficients": [0, 0.75, 0.25, 0, 0]} if polynomial else {}),
    }  # fmt: skip
    np.testing.assert_allclose(scan.projections, np.log(3.0), rtol=1e-7)
    assert not hasattr(BeamHardening, "recorded")
