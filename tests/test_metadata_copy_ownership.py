"""Persistence metadata owns current arrays while retaining scanner provenance."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj
from tomojax.io import ProjectionDataset
from tomojax.io.api import LoadedNXTomo, NXTomoMetadata
from tomojax.recon import CGLSConfig


def _stored_dataset():
    metadata = NXTomoMetadata(
        angles=np.asarray([17.123456789, 91.234567891]),
        image_key=np.asarray([0, 0], np.int32),
        grid=tj.Grid(3, 4, 2, 0.9, 1.1, 0.8),
        detector=tj.Detector(5, 3, 0.8, 0.85),
        volume=np.ones((2, 4, 3), np.float32),
        align_params=np.zeros((2, 6), np.float32),
        angle_offset_deg=np.asarray([0.031415926535, -0.271828182846]),
        geometry_calibration={"source": "stored calibration"},
        simulation_artefacts={"source": "stored simulation"},
        source_name="beamline",
        source_type="synchrotron",
        source_probe="x-ray",
        sample_name="original sample",
        frame="detector",
    )
    dataset = ProjectionDataset.from_nxtomo(
        LoadedNXTomo(projections=np.zeros((2, 3, 5), np.float32), metadata=metadata)
    )
    return dataset, metadata


@pytest.mark.parametrize("method", ["copy_metadata", "to_nxtomo_metadata"])
@pytest.mark.parametrize("arrays", [False, True])
@pytest.mark.parametrize("fortran", [False, True])
def test_metadata_uses_current_owned_arrays_and_preserves_scanner_fields(method, arrays, fortran):
    dataset, stored = _stored_dataset()
    dataset.angles = np.asarray([-31.123456789, 311.234567891])
    dataset.volume = np.arange(24, dtype=np.float32).reshape(2, 4, 3) if arrays else None
    if fortran and arrays:
        dataset.volume = np.asfortranarray(dataset.volume)
    dataset.align_params = np.full((2, 6), 0.1, np.float32) if arrays else None
    dataset.angle_offset_deg = np.asarray([0.123456789, -0.234567891]) if arrays else None
    dataset.geometry_metadata = {"updated": "current geometry"}
    dataset.align_gauge = {"pose_translation_frame": "detector"}
    dataset.misalign_spec = {"source": "current provenance"}
    dataset.sample_name = "current sample"
    copied = getattr(dataset, method)()
    for field in ("angles", "volume", "align_params", "angle_offset_deg"):
        source, value = getattr(dataset, field), getattr(copied, field)
        if source is None:
            assert value is None
        else:
            expected = source.astype(value.dtype)  # angles are stored as float32
            np.testing.assert_array_equal(value, expected)
            assert not np.shares_memory(value, source)
            before = source.copy()
            value.flat[0] += 1
            np.testing.assert_array_equal(getattr(dataset, field), before)
    assert copied.image_key is not None
    np.testing.assert_array_equal(copied.image_key, stored.image_key)
    assert not np.shares_memory(copied.image_key, stored.image_key)
    copied.image_key[0] = 3
    np.testing.assert_array_equal(stored.image_key, [0, 0])
    for field in (
        "source_name",
        "source_type",
        "source_probe",
        "frame",
        "geometry_calibration",
        "simulation_artefacts",
    ):
        assert getattr(copied, field) == getattr(stored, field)
    assert copied.sample_name == "current sample"
    assert copied.geometry_meta == dataset.geometry_metadata
    assert copied.align_gauge == dataset.align_gauge
    assert copied.misalign_spec == dataset.misalign_spec
    copied.geometry_meta["new"] = "only in copy"
    copied.align_gauge["new"] = "only in copy"
    copied.misalign_spec["new"] = "only in copy"
    assert "new" not in dataset.geometry_metadata
    assert "new" not in dataset.align_gauge
    assert "new" not in dataset.misalign_spec
    np.testing.assert_array_equal(stored.volume, np.ones((2, 4, 3)))


@pytest.mark.parametrize("suffix", [".npz", ".nxs"])
@pytest.mark.parametrize("kind", ["parallel", "cone"])
def test_resaved_reconstruction_uses_edited_volume_and_keeps_known_image(tmp_path, suffix, kind):
    grid = tj.Grid(4, 5, 3, 0.9, 1.1, 0.8, vol_center=(0.1, -0.2, 0.3))
    detector = tj.Detector(11, 9, 0.8, 0.85, center=(0.2, 0.1))
    angles = 17.123456789 + np.arange(16) * 22.5
    geometry = (
        tj.ParallelGeometry(grid, detector, angles)
        if kind == "parallel"
        else tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(20, 34), tilt_deg=27)
    )
    truth = np.random.default_rng(7).uniform(0.1, 1, (3, 5, 4)).astype(np.float32)
    data = tj.project(geometry, truth)
    config = CGLSConfig(iterations=200, rtol=1e-8)
    reconstructed = tj.reconstruct(tj.Scan(data, geometry), "cgls", config=config)
    original, edited = tmp_path / ("original" + suffix), tmp_path / ("edited" + suffix)
    tj.save(original, reconstructed)
    loaded = tj.load_reconstruction(original)
    tj.save(edited, replace(loaded, volume=truth.copy()))
    saved = tj.load_reconstruction(edited)
    np.testing.assert_array_equal(saved.volume, truth)
    np.testing.assert_allclose(saved.scan.angles, angles, rtol=1e-6)
    np.testing.assert_array_equal(saved.scan.projections, data)
    np.testing.assert_allclose(tj.project(saved.scan.geometry, truth), data, rtol=1e-5, atol=1e-5)
    again = np.asarray(tj.reconstruct(saved.scan, "cgls", config=config).volume)
    assert np.linalg.norm(again - truth) / np.linalg.norm(truth) < 2e-4
