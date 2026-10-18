"""NumPy real scalar flat/dark levels mean the same thing as Python levels."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile

import tomojax as tj
from tomojax.io import ProjectionDataset, save_dataset

pytestmark = pytest.mark.surface


@pytest.mark.parametrize("suffix", ["tif", "h5", "npz"])
@pytest.mark.parametrize("field", ["flats", "darks"])
@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.int16, np.int64, np.uint16]
)
def test_numpy_real_levels_preserve_python_calibration(
    tmp_path: Path, suffix: str, field: str, dtype: type
) -> None:
    counts = np.broadcast_to(np.asarray([50.0, 70.0], np.float32)[:, None, None], (2, 2, 4))
    path = tmp_path / f"counts.{suffix}"
    if suffix == "tif":
        tifffile.imwrite(path, counts, photometric="minisblack")
    elif suffix == "h5":
        with h5py.File(path, "w") as handle:
            handle.create_dataset("exchange/data", data=counts)
            handle.create_dataset("exchange/theta", data=[0.0, 90.0])
    else:
        save_dataset(
            path,
            ProjectionDataset(
                counts,
                np.asarray([0.0, 90.0]),
                detector=tj.Detector(4, 2, 1.0, 1.0),
                grid=tj.Grid(4, 4, 2, 1.0, 1.0, 1.0),
            ),
        )
    reference = tj.load_frames(
        path, detector=tj.Detector(4, 2, 1, 1), angles=[0.0, 90.0], flats=100.0, darks=10.0
    ).corrected()
    levels = {"flats": 100.0, "darks": 10.0}
    levels[field] = dtype(levels[field])
    actual = tj.load_frames(
        path, detector=tj.Detector(4, 2, 1, 1), angles=[0.0, 90.0], **levels
    ).corrected()
    np.testing.assert_array_equal(actual.projections, reference.projections)
    np.testing.assert_allclose(
        actual.projections[:, 0, 0], -np.log(np.asarray([40.0, 60.0]) / 90.0), rtol=1e-6
    )
    np.testing.assert_array_equal(actual.angles, reference.angles)
    assert actual.corrections == reference.corrections
