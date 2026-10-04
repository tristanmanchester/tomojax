from __future__ import annotations

import numpy as np
import pytest

from tomojax.io.api import load_angles

pytestmark = pytest.mark.surface


@pytest.mark.parametrize("suffix", [".npy", ".csv", ".txt"])
def test_angle_sidecar_preserves_acquisition_order(tmp_path, suffix):
    path = tmp_path / f"angles{suffix}"
    expected = [90.5, -0.25, 42.125]
    if suffix == ".npy":
        np.save(path, expected)
    else:
        path.write_text("# acquisition order\nangle,exposure\n90.5,1\n\n-0.25,2\n42.125,3\n")
    actual = load_angles(path)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


@pytest.mark.parametrize("values", [[], [0, np.nan], [np.inf, 1], [0, -np.inf]])
@pytest.mark.parametrize("suffix", [".npy", ".csv"])
def test_angle_sidecar_rejects_empty_or_nonfinite_acquisition(tmp_path, values, suffix):
    path = tmp_path / f"angles{suffix}"
    if suffix == ".npy":
        np.save(path, values)
    else:
        path.write_text("angle\n" + "\n".join(map(str, values)))
    with pytest.raises(ValueError, match="at least one angle and only finite"):
        load_angles(path)


def test_angle_sidecar_does_not_drop_a_corrupted_view(tmp_path):
    path = tmp_path / "angles.csv"
    path.write_text("angle\n0\nnot-a-number\n90\n")
    with pytest.raises(ValueError, match="line 3"):
        load_angles(path)


def test_angle_sidecar_requires_a_vector(tmp_path):
    path = tmp_path / "angles.npy"
    np.save(path, np.array([[0, 90]]))
    with pytest.raises(ValueError, match="one-dimensional"):
        load_angles(path)
