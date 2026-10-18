"""The external adapter preserves physical ray endpoints, not a fitted circular orbit."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tomojax.geometry import Detector, Grid


def test_tigre_cone_adapter_preserves_shifted_tilted_anisotropic_rays(monkeypatch):
    pytest.importorskip("tigre")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "bench"))
    adapter = importlib.import_module("tigre_cone")
    rotation = Rotation.from_euler("xyz", [[0, 0, 0], [0.1, -0.2, 1.2], [-0.2, 0.1, 2]]).as_matrix()
    source = np.einsum("nij,j->ni", rotation, [80, 0.3, 1.2])
    centre = np.einsum("nij,j->ni", rotation, [-45, -0.5, -0.9])
    u, v = rotation[:, :, 1] * 0.7, rotation[:, :, 2] * 1.1
    grid = Grid(31, 29, 35, 0.8, 1.2, 0.6, vol_center=(0.4, -0.6, 0.2))
    detector = Detector(43, 37, 0.7, 1.1)
    geo, angles = adapter.geometry(np.c_[source, centre, u, v], grid=grid, detector=detector)
    recovered = Rotation.from_euler("ZYZ", angles).as_matrix()
    shift = np.array(grid.vol_center) - geo.offOrigin[:, ::-1]
    np.testing.assert_allclose(recovered[:, :, 0] * geo.DSO[:, None] + shift, source, atol=1e-5)
    local = np.c_[geo.DSO - geo.DSD, geo.offDetector[:, 1], geo.offDetector[:, 0]]
    np.testing.assert_allclose(np.einsum("nij,nj->ni", recovered, local) + shift, centre, atol=1e-5)
    np.testing.assert_allclose(recovered[:, :, 1] * geo.dDetector[1], u, atol=2e-7)
    np.testing.assert_allclose(recovered[:, :, 2] * geo.dDetector[0], v, atol=2e-7)
    np.testing.assert_array_equal(geo.dVoxel, [0.6, 1.2, 0.8])
