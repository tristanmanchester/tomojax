from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tomojax.io import RealLaminographyInput, load_real_laminography_input


def _write_real_lamino_fixture(path: Path) -> tuple[np.ndarray, np.ndarray]:
    projections = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    thetas = np.asarray([10.0, 20.0], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/imaging/data", data=projections)
        handle.create_dataset("entry/imaging_sum/smaract_zrot", data=thetas)
    return projections, thetas


def test_load_real_laminography_input_base(tmp_path: Path) -> None:
    path = tmp_path / "real_lamino.nxs"
    projections, thetas = _write_real_lamino_fixture(path)

    loaded = load_real_laminography_input(path)

    assert isinstance(loaded, RealLaminographyInput)
    assert loaded.projections.dtype == np.float32
    assert loaded.thetas_deg.dtype == np.float32
    np.testing.assert_array_equal(loaded.projections, projections)
    np.testing.assert_array_equal(loaded.thetas_deg, thetas)


def test_load_real_laminography_input_transposes_detector(tmp_path: Path) -> None:
    path = tmp_path / "real_lamino.nxs"
    projections, _thetas = _write_real_lamino_fixture(path)

    loaded = load_real_laminography_input(path, transpose_detector=True)

    np.testing.assert_array_equal(loaded.projections, np.transpose(projections, (0, 2, 1)))


def test_load_real_laminography_input_flips_u(tmp_path: Path) -> None:
    path = tmp_path / "real_lamino.nxs"
    projections, _thetas = _write_real_lamino_fixture(path)

    loaded = load_real_laminography_input(path, flip_u=True)

    np.testing.assert_array_equal(loaded.projections, projections[:, :, ::-1])


def test_load_real_laminography_input_flips_v(tmp_path: Path) -> None:
    path = tmp_path / "real_lamino.nxs"
    projections, _thetas = _write_real_lamino_fixture(path)

    loaded = load_real_laminography_input(path, flip_v=True)

    np.testing.assert_array_equal(loaded.projections, projections[:, ::-1, :])


def test_load_real_laminography_input_rejects_angle_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad_real_lamino.nxs"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/imaging/data", data=np.zeros((2, 3, 4), dtype=np.float32))
        handle.create_dataset(
            "entry/imaging_sum/smaract_zrot",
            data=np.zeros((3,), dtype=np.float32),
        )

    with pytest.raises(ValueError, match="projection count 2 does not match angle count 3"):
        load_real_laminography_input(path)
