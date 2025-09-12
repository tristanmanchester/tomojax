import sys
import numpy as np
import pytest

from tomojax.data.simulate import SimConfig, simulate, simulate_to_file
from tomojax.data.datasets import validate_dataset


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def test_simulate_deterministic_seed():
    cfg = SimConfig(nx=16, ny=16, nz=16, nu=16, nv=16, n_views=8, phantom="blobs", noise="gaussian", noise_level=0.01, seed=42)
    a = simulate(cfg)
    b = simulate(cfg)
    assert np.allclose(np.asarray(a["projections"]), np.asarray(b["projections"]))


def test_simulate_nxs_roundtrip(tmp_path):
    cfg = SimConfig(nx=16, ny=16, nz=16, nu=16, nv=16, n_views=8, phantom="cube", noise="none", seed=1)
    out = tmp_path / "sim.nxs"
    simulate_to_file(cfg, str(out))
    rep = validate_dataset(str(out))
    assert rep["issues"] == []
