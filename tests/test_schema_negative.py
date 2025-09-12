import sys
import h5py
import numpy as np
import pytest

from tomojax.data.io_hdf5 import validate_nxtomo


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def test_validator_catches_missing_groups(tmp_path):
    p = tmp_path / "bad.nxs"
    with h5py.File(str(p), "w") as f:
        e = f.create_group("entry")
        inst = e.create_group("instrument")
        det = inst.create_group("detector")
        det.create_dataset("data", data=np.zeros((2, 4, 4), dtype=np.float32))
        # Intentionally omit sample/transformations/rotation_angle
    rep = validate_nxtomo(str(p))
    assert rep["issues"] and any("rotation_angle" in s for s in rep["issues"]) 
