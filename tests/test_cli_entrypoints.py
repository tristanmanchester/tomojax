from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import jax.numpy as jnp
import numpy as np
import pytest

import tomojax.cli.align as align_cli
import tomojax.cli.convert as convert_cli
import tomojax.cli.recon as recon_cli
import tomojax.cli.runtime_checks as runtime_checks_cli
import tomojax.cli.simulate as simulate_cli
from tomojax.data.io_hdf5 import LoadedNXTomo


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


def test_recon_help_documents_quicklook_aliases(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tomojax-recon", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        recon_cli.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--quicklook" in captured.out
    assert "--save-preview" in captured.out


def test_recon_views_per_batch_parser_accepts_auto_and_integers():
    assert recon_cli._parse_views_per_batch("auto") == "auto"
    assert recon_cli._parse_views_per_batch("AUTO") == "auto"
    assert recon_cli._parse_views_per_batch("4") == 4
    assert recon_cli._parse_views_per_batch("0") == 0

    with pytest.raises(argparse.ArgumentTypeError):
        recon_cli._parse_views_per_batch("fast")


def test_recon_main_writes_manifest_sidecar(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    detector = recon_cli.Detector(
        nu=2,
        nv=1,
        du=1.0,
        dv=1.0,
        det_center=(0.0, 0.0),
    )
    grid = recon_cli.Grid(nx=2, ny=2, nz=1, vx=1.0, vy=1.0, vz=1.0)
    meta = LoadedNXTomo.from_dataset(
        {
            "projections": np.zeros((2, 1, 2), dtype=np.float32),
            "thetas_deg": np.asarray([0.0, 90.0], dtype=np.float32),
            "detector": detector.to_dict(),
            "grid": grid.to_dict(),
            "geometry_type": "parallel",
        }
    )

    @contextmanager
    def fake_transfer_guard(mode: str):
        captured["transfer_guard"] = mode
        yield

    def fake_fbp(geom, recon_grid, recon_detector, projections, **kwargs):
        captured["fbp_kwargs"] = kwargs
        return jnp.zeros((recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32)

    def fake_save_nxtomo(path, *, projections, metadata):
        captured["save_path"] = path
        captured["save_metadata"] = metadata

    monkeypatch.setattr(recon_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(recon_cli, "log_jax_env", lambda: None)
    monkeypatch.setattr(recon_cli, "load_nxtomo", lambda path: meta)
    monkeypatch.setattr(
        recon_cli,
        "build_geometry_from_meta",
        lambda geometry_meta, grid_override=None, apply_saved_alignment=False: (
            grid,
            detector,
            object(),
        ),
    )
    monkeypatch.setattr(recon_cli, "transfer_guard_context", fake_transfer_guard)
    monkeypatch.setattr(recon_cli, "fbp", fake_fbp)
    monkeypatch.setattr(recon_cli, "save_nxtomo", fake_save_nxtomo)

    out_path = tmp_path / "recon.nxs"
    manifest_path = tmp_path / "manifests" / "recon.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tomojax-recon",
            "--data",
            str(tmp_path / "input.nxs"),
            "--out",
            str(out_path),
            "--roi",
            "off",
            "--gather-dtype",
            "fp32",
            "--transfer-guard",
            "log",
            "--save-manifest",
            str(manifest_path),
        ],
    )

    recon_cli.main()

    assert captured["transfer_guard"] == "log"
    assert captured["save_path"] == str(out_path)
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["command"] == "tomojax-recon"
    assert payload["argv"][0] == "tomojax-recon"
    assert payload["cli_args"]["save_manifest"] == str(manifest_path)
    assert payload["resolved_config"]["output_path"] == str(out_path)
    assert payload["resolved_config"]["gather_dtype"] == "fp32"
    assert payload["resolved_config"]["views_per_batch"] == 1
    assert payload["resolved_config"]["views_per_batch_mode"] == "default"
    assert payload["resolved_config"]["reconstruction_grid"]["nx"] == 2
    assert payload["versions"]["tomojax"]
    assert "created_at" in payload
    assert "available" in payload["jax"]


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


def test_align_main_writes_parameter_sidecars_from_returned_params(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    detector = align_cli.Detector(
        nu=2,
        nv=1,
        du=0.5,
        dv=2.0,
        det_center=(0.0, 0.0),
    )
    grid = align_cli.Grid(nx=2, ny=2, nz=1, vx=1.0, vy=1.0, vz=1.0)
    meta = LoadedNXTomo.from_dataset(
        {
            "projections": np.zeros((2, 1, 2), dtype=np.float32),
            "thetas_deg": np.asarray([0.0, 90.0], dtype=np.float32),
            "detector": detector.to_dict(),
            "grid": grid.to_dict(),
            "geometry_type": "parallel",
        }
    )
    params5 = jnp.asarray(
        [
            [0.1, 0.2, 0.3, 1.0, 4.0],
            [-0.1, -0.2, -0.3, -2.0, -6.0],
        ],
        dtype=jnp.float32,
    )

    @contextmanager
    def fake_transfer_guard(mode: str):
        captured["transfer_guard"] = mode
        yield

    def fake_align(geom, recon_grid, recon_detector, projections, *, cfg):
        captured["align_grid"] = recon_grid
        captured["align_detector"] = recon_detector
        captured["align_projections_shape"] = tuple(projections.shape)
        x = jnp.zeros((recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32)
        info = {
            "loss": [1.0],
            "outer_stats": [],
            "stopped_by_observer": False,
            "observer_action": "continue",
            "wall_time_total": 0.0,
        }
        return x, params5, info

    def fake_save_nxtomo(path, *, projections, metadata):
        captured["save_path"] = path
        captured["save_projections_shape"] = tuple(projections.shape)
        captured["save_metadata"] = metadata

    monkeypatch.setattr(align_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(align_cli, "log_jax_env", lambda: None)
    monkeypatch.setattr(align_cli, "_init_jax_compilation_cache", lambda: None)
    monkeypatch.setattr(align_cli, "load_nxtomo", lambda path: meta)
    monkeypatch.setattr(
        align_cli,
        "build_geometry_from_meta",
        lambda geometry_meta, grid_override=None, apply_saved_alignment=False: (
            grid,
            detector,
            object(),
        ),
    )
    monkeypatch.setattr(align_cli, "transfer_guard_context", fake_transfer_guard)
    monkeypatch.setattr(align_cli, "align", fake_align)
    monkeypatch.setattr(align_cli, "save_nxtomo", fake_save_nxtomo)

    out_path = tmp_path / "aligned.nxs"
    json_path = tmp_path / "params" / "aligned.json"
    csv_path = tmp_path / "params" / "aligned.csv"
    manifest_path = tmp_path / "manifests" / "aligned.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tomojax-align",
            "--data",
            str(tmp_path / "input.nxs"),
            "--out",
            str(out_path),
            "--roi",
            "off",
            "--gather-dtype",
            "fp32",
            "--transfer-guard",
            "log",
            "--save-params-json",
            str(json_path),
            "--save-params-csv",
            str(csv_path),
            "--save-manifest",
            str(manifest_path),
        ],
    )

    align_cli.main()

    assert captured["transfer_guard"] == "log"
    assert captured["save_path"] == str(out_path)
    assert captured["save_projections_shape"] == (2, 1, 2)
    saved_meta = captured["save_metadata"]
    np.testing.assert_allclose(np.asarray(saved_meta.align_params), np.asarray(params5))
    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["views"]) == 2
    assert payload["views"][0]["dx_px"] == pytest.approx(2.0)
    assert payload["views"][1]["dz_px"] == pytest.approx(-3.0)

    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 2
    assert rows[0]["view_index"] == "0"
    assert float(rows[0]["dx_px"]) == pytest.approx(2.0)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["command"] == "tomojax-align"
    assert manifest["argv"][0] == "tomojax-align"
    assert manifest["cli_args"]["save_manifest"] == str(manifest_path)
    assert manifest["resolved_config"]["output_path"] == str(out_path)
    assert manifest["resolved_config"]["gather_dtype"] == "fp32"
    assert manifest["resolved_config"]["levels"] is None
    assert manifest["resolved_config"]["reconstruction_grid"]["nx"] == 2
    assert manifest["resolved_config"]["alignment_params_shape"] == [2, 5]
    assert manifest["versions"]["tomojax"]
    assert "created_at" in manifest
    assert "available" in manifest["jax"]
