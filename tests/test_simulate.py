from contextlib import nullcontext
import os
import sys
import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.cli import simulate as simulate_cli
from tomojax.data import simulate as simulate_mod
from tomojax.data.simulate import SimConfig, simulate, simulate_to_file
from tomojax.data.datasets import validate_dataset


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def test_simulate_deterministic_seed():
    cfg = SimConfig(
        nx=16,
        ny=16,
        nz=16,
        nu=16,
        nv=16,
        n_views=8,
        phantom="blobs",
        noise="gaussian",
        noise_level=0.01,
        seed=42,
    )
    a = simulate(cfg)
    b = simulate(cfg)
    assert np.allclose(np.asarray(a["projections"]), np.asarray(b["projections"]))


def test_simulate_nxs_roundtrip(tmp_path):
    cfg = SimConfig(
        nx=16, ny=16, nz=16, nu=16, nv=16, n_views=8, phantom="cube", noise="none", seed=1
    )
    out = tmp_path / "sim.nxs"
    simulate_to_file(cfg, str(out))
    rep = validate_dataset(str(out))
    assert rep["issues"] == []


def test_simulate_slow_path_forwards_default_gather_dtype(monkeypatch):
    seen = []

    def spy_forward_project_view(
        geom, grid, det, vol, *, view_index, gather_dtype="fp32", **kwargs
    ):
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


def test_simulate_cli_builds_config_and_calls_simulate_to_file(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    monkeypatch.setattr(simulate_cli, "setup_logging", lambda: captured.setdefault("setup", True))
    monkeypatch.setattr(simulate_cli, "log_jax_env", lambda: captured.setdefault("env", True))

    def fake_transfer_guard_context(mode: str):
        captured["guard"] = mode
        return nullcontext()

    monkeypatch.setattr(
        simulate_cli,
        "transfer_guard_context",
        fake_transfer_guard_context,
    )

    def fake_simulate_to_file(cfg: SimConfig, out_path: str) -> str:
        captured["cfg"] = cfg
        captured["out_path"] = out_path
        return out_path

    monkeypatch.setattr(simulate_cli, "simulate_to_file", fake_simulate_to_file)
    monkeypatch.setattr(
        "sys.argv",
        [
            "simulate",
            "--out",
            str(tmp_path / "sim.nxs"),
            "--nx",
            "8",
            "--ny",
            "9",
            "--nz",
            "10",
            "--nu",
            "11",
            "--nv",
            "12",
            "--n-views",
            "5",
            "--geometry",
            "lamino",
            "--tilt-deg",
            "25",
            "--tilt-about",
            "z",
            "--phantom",
            "cube",
            "--noise",
            "gaussian",
            "--noise-level",
            "0.1",
            "--seed",
            "7",
            "--transfer-guard",
            "log",
            "--progress",
        ],
    )
    monkeypatch.delenv("TOMOJAX_PROGRESS", raising=False)

    simulate_cli.main()

    cfg = captured["cfg"]
    assert isinstance(cfg, SimConfig)
    assert cfg.geometry == "lamino"
    assert cfg.tilt_deg == 25.0
    assert cfg.tilt_about == "z"
    assert cfg.phantom == "cube"
    assert cfg.noise == "gaussian"
    assert cfg.noise_level == 0.1
    assert cfg.seed == 7
    assert captured["guard"] == "log"
    assert captured["out_path"] == str(tmp_path / "sim.nxs")
    assert captured["setup"] is True
    assert captured["env"] is True
    assert os.environ["TOMOJAX_PROGRESS"] == "1"
