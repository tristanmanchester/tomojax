from __future__ import annotations

import pytest

from tomojax.calibration import DetectorPixelScale, DetectorPixelValue
from tomojax.core.geometry import Detector
from tomojax.recon.multires import scale_detector


def test_detector_pixel_scale_reports_native_level_and_physical_units():
    native = Detector(nu=128, nv=96, du=0.5, dv=0.25, det_center=(0.0, 0.0))
    level = scale_detector(native, 4)
    scale = DetectorPixelScale.from_detectors(native, level)

    assert scale.native_px_to_physical(-4.0, "u") == pytest.approx(-2.0)
    assert scale.native_px_to_level_px(-4.0, "u") == pytest.approx(-1.0)
    assert scale.level_px_to_native_px(-1.0, "u") == pytest.approx(-4.0)
    assert scale.physical_to_native_px(-2.0, "u") == pytest.approx(-4.0)
    assert scale.native_px_to_physical(6.0, "v") == pytest.approx(1.5)
    assert scale.native_px_to_level_px(6.0, "v") == pytest.approx(1.5)


def test_detector_pixel_value_report_keeps_native_pixel_as_canonical_unit():
    scale = DetectorPixelScale(native_du=0.5, native_dv=0.25, bin_factor_u=4, bin_factor_v=2)
    value = DetectorPixelValue(axis="u", native_px=-4.0)

    report = value.report(scale)

    assert report["axis"] == "u"
    assert report["native_px"] == pytest.approx(-4.0)
    assert report["level_px"] == pytest.approx(-1.0)
    assert report["physical"] == pytest.approx(-2.0)


def test_detector_pixel_scale_rejects_non_positive_values():
    with pytest.raises(ValueError, match="native_du"):
        DetectorPixelScale(native_du=0.0, native_dv=1.0)

    with pytest.raises(ValueError, match="bin_factor_v"):
        DetectorPixelScale(native_du=1.0, native_dv=1.0, bin_factor_v=-1.0)
