from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import tomojax.cli.convert as convert_cli
import tomojax.cli.runtime_checks as runtime_checks_cli
import tomojax.cli.simulate as simulate_cli


def test_convert_main_delegates_to_converter(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_convert(in_path: str, out_path: str) -> None:
        calls.append((in_path, out_path))

    monkeypatch.setattr(convert_cli, "convert", fake_convert)
    monkeypatch.setattr(
        sys,
        "argv",
        ["tomojax-convert", "--in", "input.npz", "--out", "output.nxs"],
    )

    convert_cli.main()

    assert calls == [("input.npz", "output.nxs")]


def test_runtime_checks_cpu_main_sets_cpu_backend_and_prints_devices(monkeypatch, capsys):
    fake_jax = SimpleNamespace(
        default_backend=lambda: "cpu",
        devices=lambda: ["cpu:0"],
    )

    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)

    runtime_checks_cli.test_cpu_main()

    captured = capsys.readouterr()
    assert "JAX backend: cpu" in captured.out
    assert "Devices: ['cpu:0']" in captured.out
    assert os.environ["JAX_PLATFORM_NAME"] == "cpu"


def test_simulate_main_builds_config_and_runs_writer(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    @contextmanager
    def fake_transfer_guard(mode: str):
        captured["transfer_guard"] = mode
        yield

    def fake_setup_logging() -> None:
        captured["setup_logging"] = True

    def fake_log_jax_env() -> None:
        captured["log_jax_env"] = True

    def fake_simulate_to_file(cfg, out_path: str) -> str:
        captured["cfg"] = cfg
        captured["out_path"] = out_path
        return out_path

    monkeypatch.setattr(simulate_cli, "setup_logging", fake_setup_logging)
    monkeypatch.setattr(simulate_cli, "log_jax_env", fake_log_jax_env)
    monkeypatch.setattr(simulate_cli, "transfer_guard_context", fake_transfer_guard)
    monkeypatch.setattr(simulate_cli, "simulate_to_file", fake_simulate_to_file)
    monkeypatch.delenv("TOMOJAX_PROGRESS", raising=False)

    out_path = tmp_path / "simulated.nxs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tomojax-simulate",
            "--out",
            str(out_path),
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
            "13",
            "--geometry",
            "lamino",
            "--tilt-deg",
            "25",
            "--tilt-about",
            "z",
            "--phantom",
            "cube",
            "--single-size",
            "0.25",
            "--single-value",
            "2.5",
            "--noise",
            "gaussian",
            "--noise-level",
            "0.1",
            "--seed",
            "7",
            "--progress",
            "--transfer-guard",
            "log",
        ],
    )

    simulate_cli.main()

    cfg = captured["cfg"]
    assert captured["setup_logging"] is True
    assert captured["log_jax_env"] is True
    assert captured["transfer_guard"] == "log"
    assert captured["out_path"] == str(out_path)
    assert cfg.nx == 8
    assert cfg.ny == 9
    assert cfg.nz == 10
    assert cfg.nu == 11
    assert cfg.nv == 12
    assert cfg.n_views == 13
    assert cfg.geometry == "lamino"
    assert cfg.tilt_deg == 25.0
    assert cfg.tilt_about == "z"
    assert cfg.phantom == "cube"
    assert cfg.single_size == 0.25
    assert cfg.single_value == 2.5
    assert cfg.noise == "gaussian"
    assert cfg.noise_level == 0.1
    assert cfg.seed == 7
    assert Path(captured["out_path"]).name == "simulated.nxs"
    assert os.environ["TOMOJAX_PROGRESS"] == "1"
