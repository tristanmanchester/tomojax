"""A parallel-beam file that records no pixel size loads in pixel units; pixel_size= gives one."""

from pathlib import Path

import h5py
import numpy as np
import pytest

import tomojax as tj
from tomojax.io import ProjectionDataset, save_dataset


def _pitchless_data_exchange(path: Path) -> np.ndarray:
    """A Data Exchange scan of an off-center Gaussian with flats and darks and no pixel size."""
    n, views = 16, 64
    angles = np.linspace(0, 180, views, endpoint=False)
    geometry = tj.ParallelGeometry(tj.Grid(n, n, 4, 1, 1, 1), tj.Detector(n, 4, 1, 1), angles)
    z, y, x = np.mgrid[:4, :n, :n].astype(np.float32)
    volume = 0.02 * np.exp(-((x - 9) ** 2 + (y - 6) ** 2) / 8).astype(np.float32)
    line_integrals = np.asarray(tj.project(geometry, volume))
    with h5py.File(path, "w") as file:
        file["exchange/data"] = 10000 * np.exp(-line_integrals)
        file["exchange/data_white"] = np.full((1, 4, n), 10000, np.float32)
        file["exchange/data_dark"] = np.zeros((1, 4, n), np.float32)
        file["exchange/theta"] = angles
    return volume


def test_a_file_without_a_pixel_size_loads_in_pixel_units(tmp_path: Path) -> None:
    path = tmp_path / "scan.h5"
    volume = _pitchless_data_exchange(path)
    scan = tj.load(path)
    assert (scan.detector.du, scan.detector.dv) == (1.0, 1.0)
    meta = scan.source.geometry_metadata
    assert meta["length_unit"] == "pixel" and meta["detector_pixel_size_status"] == "not_recorded"
    result = tj.reconstruct(scan, "fbp")
    error = np.linalg.norm(np.asarray(result.volume) - volume) / np.linalg.norm(volume)
    assert error < 0.1
    tj.save(tmp_path / "saved.nxs", scan)
    again = tj.load(tmp_path / "saved.nxs")
    assert again.detector == scan.detector
    assert again.source.geometry_metadata["length_unit"] == "pixel"


@pytest.mark.parametrize("method", ["fbp", "fista"])
def test_a_pixel_size_scales_the_reconstruction_and_nothing_else(tmp_path: Path, method) -> None:
    # In pixel units the volume is attenuation per pixel; with the pixel p it is per length
    # unit, 1/p times as large on a grid p times as fine in length. Nothing else may differ,
    # including FISTA's automatic TV weight.
    path = tmp_path / "scan.h5"
    _pitchless_data_exchange(path)
    p = 0.65
    in_pixels = tj.reconstruct(tj.load(path), method)
    scan = tj.load(path, pixel_size=p)
    assert (scan.detector.du, scan.detector.dv) == (p, p) and scan.grid.vx == pytest.approx(p)
    assert "length_unit" not in scan.source.geometry_metadata
    physical = tj.reconstruct(scan, method)
    np.testing.assert_allclose(
        np.asarray(physical.volume), np.asarray(in_pixels.volume) / p, rtol=1e-4, atol=1e-7
    )


def test_pixel_size_with_a_detector_or_geometry_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "scan.h5"
    _pitchless_data_exchange(path)
    with pytest.raises(ValueError, match="pixel_size= and detector="):
        tj.load(path, pixel_size=0.5, detector=tj.Detector(16, 4, 0.5, 0.5))


def test_a_cone_file_without_a_pixel_size_asks_for_one(tmp_path: Path) -> None:
    # A cone beam's distances fix its unit, so a pitch cannot default to a pixel.
    import inspect
    import re

    record = ProjectionDataset(
        np.full((4, 3, 5), 500, np.uint16),
        np.arange(4, dtype=np.float32) * 90,
        geometry_type="cone",
        geometry_metadata={"cone_beam": {"source_to_axis": 100.0, "source_to_detector": 200.0}},
    )
    path = tmp_path / "cone.npz"
    save_dataset(path, record)
    with pytest.raises(ValueError, match="Missing detector pixel sizes") as caught:
        tj.load(path, flats=1000, darks=0)
    named = set(re.findall(r"\b(\w+)=\.\.\.", str(caught.value)))
    assert "pixel_size" in named and named <= set(inspect.signature(tj.load).parameters)
    scan = tj.load(path, flats=1000, darks=0, pixel_size=0.5)
    assert isinstance(scan.geometry, tj.ConeGeometry) and scan.detector.du == 0.5
