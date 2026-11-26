"""tj.load reads a PSI cSAXS ptychographic tomography file: its phase line integrals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest
from scipy.io import savemat

import tomojax as tj

if TYPE_CHECKING:
    from pathlib import Path


def _write_v73(path: Path, phase: np.ndarray, angles: np.ndarray, pixel: float) -> None:
    """A MATLAB 7.3 file as MATLAB writes one: HDF5 behind a 512-byte text header,
    each variable in MATLAB's column order (so reversed in HDF5)."""
    with h5py.File(path, "w", userblock_size=512) as file:
        # Chunks of 3 views, gzip, as the cSAXS files are chunked across views.
        file.create_dataset(
            "aligned_phase_sinogram",
            data=np.transpose(phase, (0, 2, 1)),
            chunks=(3, 1, phase.shape[1]),
            compression="gzip",
        )
        file["rotation_angle"] = angles[:, None]
        file["pixel_size"] = np.array([[pixel]])
        file["lambda"] = np.array([[2e-10]])
    with path.open("r+b") as handle:
        handle.write(b"MATLAB 7.3 MAT-file, Platform: GLNXA64".ljust(116))


def test_load_reads_the_phase_angles_and_pixel_size(tmp_path):
    rng = np.random.default_rng(5)
    views, rows, columns = 7, 4, 6  # 7 views: two whole chunks of 3 and one partial
    phase = rng.random((views, rows, columns), dtype=np.float32)
    angles = np.linspace(0.0, 179.5, views)
    path = tmp_path / "porous.mat"
    _write_v73(path, phase, angles, 6.868e-9)

    scan = tj.load(path)

    np.testing.assert_array_equal(np.asarray(scan.projections), phase)
    np.testing.assert_allclose(np.asarray(scan.geometry.angles), angles, rtol=1e-6)
    detector = scan.geometry.detector
    assert (detector.nu, detector.nv) == (columns, rows)
    # Meters in the file, millimeters in the scan, as every reader gives lengths.
    assert detector.du == detector.dv == pytest.approx(6.868e-6)
    assert detector.center == (0.0, 0.0)
    assert isinstance(scan.geometry, tj.ParallelGeometry)
    report = tj.inspect(path)
    assert report["valid"], report["issues"]
    assert report["projection"]["shape"] == [views, rows, columns]
    assert report["detector_metadata"]["du"] == pytest.approx(6.868e-6)


def test_load_refuses_complex_unaligned_projections_naming_what_the_file_holds(tmp_path):
    path = tmp_path / "example_data.mat"
    stack = np.ones((4, 6, 3), np.complex64)
    savemat(path, {"stack_object": stack, "theta": np.arange(3.0)})
    with pytest.raises(ValueError, match="stack_object holds complex projections"):
        tj.load(path)
