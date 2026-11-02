"""Corpus rendering integrates photons, preserving fixed detector axes."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def renderer():
    path = Path(__file__).resolve().parents[1] / "bench/phantoms/gvxr_transmission.py"
    spec = importlib.util.spec_from_file_location("gvxr_transmission_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spectrum_keeps_mono_and_normalizes_detected_photons(renderer):
    mu, weights = renderer._spectrum({"mu_mm": [0.1, 0.7]})
    np.testing.assert_array_equal(mu, [[0.1], [0.7]])
    np.testing.assert_array_equal(weights, [1.0])
    _, weights = renderer._spectrum({"mu_mm": [[0.1, 0.7]], "photon_fractions": [2, 3]})
    np.testing.assert_allclose(weights, [0.4, 0.6])
    with pytest.raises(ValueError):
        renderer._spectrum({"mu_mm": [[0.1, 0.7]], "photon_fractions": [1, -1]})


@pytest.mark.parametrize("cone", [False, True])
@pytest.mark.parametrize("refinement", [1, 3])
def test_render_integrates_energy_then_pixels_and_sets_fixed_detector_frame(
    renderer, monkeypatch, cone, refinement
):
    length_cm = np.array([[0.01, 0.1], [0.2, 0.4]])
    api = MagicMock()
    raster = np.full((2 * refinement, 2 * refinement), 99.0)
    raster[refinement // 2 :: refinement, refinement // 2 :: refinement] = length_cm
    api.computePathLength.return_value = raster
    api.getVersionOfCoreGVXR.return_value = "test"
    monkeypatch.setitem(sys.modules, "gvxrPython3", SimpleNamespace(gvxr=api))
    pose = np.eye(4)
    pose[:3, 3] = [0.2, -0.3, 0.4]
    plan = dict(
        meshes=".",
        names=["material"],
        mu_mm=[[0.2, 0.8]],
        photon_fractions=[1, 3],
        shape=[1, 1],
        pixel_mm=0.3,
        centre_mm=[0.17, -0.23],
        supersample=2,
        raster_refinement=refinement,
        poses=[pose],
    )
    if cone:
        plan.update(source_to_axis_mm=20.0, source_to_detector_mm=50.0)
    result = renderer.render(plan)
    expected = (np.exp(-10 * length_cm[..., None] * [0.2, 0.8]) @ [0.25, 0.75]).mean()
    np.testing.assert_allclose(result["transmission"], [[[expected]]], rtol=1e-7)
    np.testing.assert_allclose(api.setDetectorRightVector.call_args.args, [1, 0, 0])
    np.testing.assert_allclose(api.setDetectorUpVector.call_args.args, [0, 0, 1])
    source = [0, -20, 0] if cone else [0.17, -100, -0.23]
    np.testing.assert_allclose(api.setSourcePosition.call_args.args[:3], source - pose[:3, 3])
    (api.usePointSource if cone else api.useParallelBeam).assert_called_once()
    api.setDetectorNumberOfPixels.assert_called_once_with(2 * refinement, 2 * refinement)
    api.destroy.assert_called_once()
