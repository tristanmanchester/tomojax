"""Read both MATLAB encodings in one physical axis convention."""

from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from nanotomo_data import PtychographyData


def test_matlab_readers_preserve_axes_complex_values_metadata_and_reference_crop(tmp_path):
    expected = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9)
    objects = expected + 1j * (expected + 3)
    angles = np.array([0.0, 80.0, 40.0, 120.0, 160.0])
    reference = expected[:, 1:-1, 2:-2]
    for hdf5 in (False, True):
        path = tmp_path / f"sample-{hdf5}.mat"
        if hdf5:
            compound = np.empty(
                objects.transpose(0, 2, 1).shape, dtype=[("real", "f4"), ("imag", "f4")]
            )
            compound["real"] = objects.real.transpose(0, 2, 1)
            compound["imag"] = objects.imag.transpose(0, 2, 1)
            with h5py.File(path, "w") as stream:
                stream["stack_object"] = compound
                stream["rotation_angle"] = angles[:, None]
                stream["aligned_phase_sinogram"] = reference.transpose(0, 2, 1)
                stream["pixel_size"] = [[6.868e-9]]
        else:
            savemat(
                path,
                dict(
                    stack_object=objects.transpose(1, 2, 0),
                    theta=angles,
                    aligned_phase_sinogram=reference.transpose(1, 2, 0),
                    pixel_size=6.868e-9,
                ),
            )
        with PtychographyData(path) as source:
            assert source.shape == expected.shape
            assert source.reference_shape == reference.shape
            assert source.metadata["pixel_size"] == 6.868e-9
            np.testing.assert_array_equal(source.angles, angles)
            np.testing.assert_array_equal(
                source.read(slice(1, 4), rows=slice(2, 6), cols=slice(3, 8)), objects[1:4, 2:6, 3:8]
            )
            np.testing.assert_array_equal(source.read(reference=True), reference)


def test_packed_matlab_complex_values_are_views_not_full_stack_copies():
    from nanotomo_data import complex_values

    for real_dtype in ("<f4", ">f8"):
        packed = np.zeros((3, 4, 5), dtype=[("real", real_dtype), ("imag", real_dtype)])
        packed["real"] = 1.5
        packed["imag"] = -2.25
        values = complex_values(packed)
        assert np.shares_memory(packed, values)
        np.testing.assert_array_equal(values, np.full(packed.shape, 1.5 - 2.25j))
