import h5py
import numpy as np
import pytest

from tomojax.data.io_hdf5 import validate_nxtomo


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


def test_validator_reports_unreadable_hdf5_file(tmp_path):
    path = tmp_path / "not_hdf5.nxs"
    path.write_text("not an hdf5 file", encoding="utf-8")

    report = validate_nxtomo(str(path))

    assert report["issues"]
    assert any("Unable to read HDF5 file" in issue for issue in report["issues"])
