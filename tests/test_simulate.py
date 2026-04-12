import sys
import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.data import simulate as simulate_mod
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


def test_simulate_slow_path_forwards_default_gather_dtype(monkeypatch):
    seen = []

    def spy_forward_project_view(geom, grid, det, vol, *, view_index, gather_dtype="fp32", **kwargs):
        seen.append((view_index, gather_dtype))
        return jnp.zeros((det.nv, det.nu), dtype=jnp.float32)

    monkeypatch.setattr(simulate_mod, "default_gather_dtype", lambda: "bf16")
    monkeypatch.setattr(simulate_mod, "forward_project_view", spy_forward_project_view)

    cfg = SimConfig(
        nx=8,
        ny=8,
        nz=8,
        nu=6,
        nv=4,
        n_views=3,
        phantom="cube",
        noise="none",
        seed=0,
    )
    out = simulate(cfg)

    assert out["projections"].shape == (cfg.n_views, cfg.nv, cfg.nu)
    assert seen == [(0, "bf16"), (1, "bf16"), (2, "bf16")]
