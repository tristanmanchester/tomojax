import sys

import h5py
import numpy as np
import pytest

from tomojax.cli import validate as validate_cli


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def _write_minimal_nxtomo(path):
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["definition"] = "NXtomo"

        detector = entry.create_group("instrument").create_group("detector")
        detector.create_dataset("data", data=np.zeros((2, 4, 4), dtype=np.float32))
        detector.create_dataset("image_key", data=np.zeros((2,), dtype=np.int32))

        sample = entry.create_group("sample")
        sample.create_dataset("name", data="sample")
        rotation_angle = sample.create_group("transformations").create_dataset(
            "rotation_angle",
            data=np.array([0.0, 90.0], dtype=np.float32),
        )
        rotation_angle.attrs["units"] = "degree"


def test_validate_cli_reports_valid_file(tmp_path, capsys):
    path = tmp_path / "valid.nxs"
    _write_minimal_nxtomo(path)

    status = validate_cli.main([str(path)])

    captured = capsys.readouterr()
    assert status == 0
    assert captured.out == f"OK: {path}\n"


def test_validate_cli_reports_invalid_file(tmp_path, capsys):
    path = tmp_path / "invalid.nxs"
    with h5py.File(path, "w") as f:
        f.create_group("not_entry")

    status = validate_cli.main([str(path)])

    captured = capsys.readouterr()
    assert status == 1
    assert f"INVALID: {path} (1 issue)" in captured.out
    assert "- Missing /entry" in captured.out


def test_validate_cli_reports_missing_file(tmp_path, capsys):
    path = tmp_path / "missing.nxs"

    status = validate_cli.main([str(path)])

    captured = capsys.readouterr()
    assert status != 0
    assert "ERROR: file not found" in captured.err
