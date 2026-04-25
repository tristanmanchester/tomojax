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
import tomojax.cli.inspect as inspect_cli
import tomojax.cli.preprocess as preprocess_cli
import tomojax.cli.recon as recon_cli
import tomojax.cli.runtime_checks as runtime_checks_cli
import tomojax.cli.simulate as simulate_cli
from tomojax.data.io_hdf5 import LoadedNXTomo
from tomojax.core.geometry import Detector, Grid


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


def test_align_compilation_cache_uses_current_jax_config_api(monkeypatch, tmp_path):
    updates: list[tuple[str, str | int]] = []

    def update(name: str, value: str | int) -> None:
        updates.append((name, value))

    def initialize_cache(_: str) -> None:
        raise AssertionError("deprecated initialize_cache must not be used")

    fake_jax = SimpleNamespace(config=SimpleNamespace(update=update))
    fake_experimental = SimpleNamespace(
        compilation_cache=SimpleNamespace(initialize_cache=initialize_cache)
    )
    cache_dir = tmp_path / "jax-cache"

    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.experimental", fake_experimental)
    monkeypatch.setenv("TOMOJAX_JAX_CACHE_DIR", str(cache_dir))

    align_cli._init_jax_compilation_cache()

    assert cache_dir.is_dir()
    assert updates == [
        ("jax_compilation_cache_dir", str(cache_dir)),
        ("jax_persistent_cache_min_entry_size_bytes", -1),
        ("jax_persistent_cache_min_compile_time_secs", 0),
        (
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        ),
    ]


def test_align_compilation_cache_defaults_under_xdg_cache_home(monkeypatch, tmp_path):
    updates: list[tuple[str, str | int]] = []
    fake_jax = SimpleNamespace(
        config=SimpleNamespace(update=lambda name, value: updates.append((name, value)))
    )
    expected_cache_dir = tmp_path / "xdg" / "tomojax" / "jax_cache"

    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.delenv("TOMOJAX_JAX_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    align_cli._init_jax_compilation_cache()

    assert expected_cache_dir.is_dir()
    assert updates[0] == ("jax_compilation_cache_dir", str(expected_cache_dir))


def test_recon_help_documents_quicklook_aliases(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tomojax-recon", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        recon_cli.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--quicklook" in captured.out
    assert "--save-preview" in captured.out


def test_inspect_help_documents_json_and_quicklook(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tomojax-inspect", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        inspect_cli.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--json" in captured.out
    assert "--quicklook" in captured.out


def test_preprocess_help_documents_safeguards(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tomojax-preprocess", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        preprocess_cli.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--log" in captured.out
    assert "--epsilon" in captured.out
    assert "--clip-min" in captured.out
    assert "output" in captured.out


def test_align_help_documents_bounds_example(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tomojax-align", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        align_cli.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--bounds" in captured.out
    assert "dx=-20:20,dz=-20:20,alpha=-0.05:0.05" in captured.out
    assert "--pose-model" in captured.out
    assert "--recon-algo" in captured.out
    assert "--views-per-batch" in captured.out
    assert "--spdhg-seed" in captured.out
    assert "--no-recon-positivity" in captured.out
    assert "--checkpoint" in captured.out
    assert "--checkpoint-every" in captured.out
    assert "--resume" in captured.out
    assert "--optimise-geometry" in captured.out
    assert "det_u_px" in captured.out
    assert "detector_roll_deg" in captured.out
    assert "lbfgs" in captured.out
    assert "--lbfgs-memory-size" in captured.out


def test_align_geometry_dof_parser_accepts_detector_and_axis_aliases():
    assert align_cli._parse_geometry_dofs_arg("det_u_px,detector_roll_deg") == (
        "det_u_px",
        "detector_roll_deg",
    )
    assert align_cli._parse_geometry_dofs_arg("tilt_deg") == ("tilt_deg",)
    with pytest.raises(argparse.ArgumentTypeError):
        align_cli._parse_geometry_dofs_arg("dx")


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
    config_path = tmp_path / "recon.toml"
    config_path.write_text(
        "\n".join(
            [
                f'data = "{tmp_path / "input.nxs"}"',
                f'out = "{out_path}"',
                'roi = "off"',
                'gather_dtype = "bf16"',
                'transfer_guard = "off"',
                f'save_manifest = "{manifest_path}"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tomojax-recon",
            "--config",
            str(config_path),
            "--gather-dtype",
            "fp32",
            "--transfer-guard",
            "log",
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
    assert payload["cli_args"]["config"] == str(config_path)
    assert payload["cli_args"]["save_manifest"] == str(manifest_path)
    assert payload["resolved_config"]["output_path"] == str(out_path)
    assert payload["resolved_config"]["config_path"] == str(config_path)
    assert payload["resolved_config"]["config_file_values"]["gather_dtype"] == "bf16"
    assert "gather_dtype" in payload["resolved_config"]["explicit_cli_keys"]
    assert payload["resolved_config"]["effective_options"]["gather_dtype"] == "fp32"
    assert payload["resolved_config"]["gather_dtype"] == "fp32"
    assert payload["resolved_config"]["views_per_batch"] == 1
    assert payload["resolved_config"]["views_per_batch_mode"] == "default"
    assert payload["resolved_config"]["reconstruction_grid"]["nx"] == 2
    assert payload["versions"]["tomojax"]
    assert "created_at" in payload
    assert "available" in payload["jax"]


def test_recon_main_passes_fista_constraints_and_records_manifest(monkeypatch, tmp_path):
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

    def fake_fista_tv(geom, recon_grid, recon_detector, projections, *, config, det_grid=None):
        captured["fista_grid"] = recon_grid
        captured["fista_detector"] = recon_detector
        captured["fista_projections_shape"] = tuple(projections.shape)
        captured["fista_config"] = config
        captured["fista_det_grid"] = det_grid
        volume = jnp.zeros((recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32)
        return volume, {"loss": [0.0]}

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
    monkeypatch.setattr(recon_cli, "fista_tv", fake_fista_tv)
    monkeypatch.setattr(recon_cli, "save_nxtomo", fake_save_nxtomo)

    out_path = tmp_path / "fista.nxs"
    manifest_path = tmp_path / "manifests" / "fista.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tomojax-recon",
            "--data",
            str(tmp_path / "input.nxs"),
            "--algo",
            "fista",
            "--iters",
            "3",
            "--lambda-tv",
            "0.0",
            "--L",
            "1.0",
            "--roi",
            "off",
            "--gather-dtype",
            "fp32",
            "--positivity",
            "--lower-bound",
            "0",
            "--upper-bound",
            "1",
            "--out",
            str(out_path),
            "--save-manifest",
            str(manifest_path),
        ],
    )

    recon_cli.main()

    cfg = captured["fista_config"]
    assert cfg.positivity is True
    assert cfg.lower_bound == pytest.approx(0.0)
    assert cfg.upper_bound == pytest.approx(1.0)
    assert cfg.iters == 3
    assert cfg.L == pytest.approx(1.0)
    assert captured["save_path"] == str(out_path)
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    algorithm_config = payload["resolved_config"]["algorithm_config"]
    assert algorithm_config["positivity"] is True
    assert algorithm_config["lower_bound"] == pytest.approx(0.0)
    assert algorithm_config["upper_bound"] == pytest.approx(1.0)


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
            "--gaussian-sigma",
            "0.2",
            "--dropped-view-fraction",
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
    assert cfg.artefacts is not None
    assert cfg.artefacts.gaussian_sigma == 0.2
    assert cfg.artefacts.dropped_view_fraction == 0.1
    assert cfg.artefacts.poisson_scale == 0.0
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

    def fake_align(
        geom,
        recon_grid,
        recon_detector,
        projections,
        *,
        cfg,
        resume_state=None,
        checkpoint_callback=None,
    ):
        captured["align_grid"] = recon_grid
        captured["align_detector"] = recon_detector
        captured["align_projections_shape"] = tuple(projections.shape)
        captured["align_cfg"] = cfg
        captured["align_resume_state"] = resume_state
        captured["align_checkpoint_callback"] = checkpoint_callback
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
    config_path = tmp_path / "align.toml"
    config_path.write_text(
        "\n".join(
            [
                f'data = "{tmp_path / "input.nxs"}"',
                f'out = "{out_path}"',
                'roi = "off"',
                'gather_dtype = "bf16"',
                'transfer_guard = "off"',
                "bounds = { dx = [-20, 20], alpha = [-0.05, 0.05] }",
                'pose_model = "spline"',
                "knot_spacing = 5",
                "degree = 2",
                'recon_algo = "spdhg"',
                "views_per_batch = 8",
                "spdhg_seed = 7",
                "recon_positivity = false",
                "lbfgs_memory_size = 9",
                f'save_params_json = "{json_path}"',
                f'save_params_csv = "{csv_path}"',
                f'save_manifest = "{manifest_path}"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tomojax-align",
            "--config",
            str(config_path),
            "--gather-dtype",
            "fp32",
            "--views-per-batch",
            "2",
            "--transfer-guard",
            "log",
        ],
    )

    align_cli.main()

    assert captured["transfer_guard"] == "log"
    assert captured["save_path"] == str(out_path)
    assert captured["save_projections_shape"] == (2, 1, 2)
    assert captured["align_cfg"].bounds == (
        ("alpha", -0.05, 0.05),
        ("dx", -20.0, 20.0),
    )
    assert captured["align_cfg"].pose_model == "spline"
    assert captured["align_cfg"].knot_spacing == 5
    assert captured["align_cfg"].degree == 2
    assert captured["align_cfg"].recon_algo == "spdhg"
    assert captured["align_cfg"].views_per_batch == 2
    assert captured["align_cfg"].spdhg_seed == 7
    assert captured["align_cfg"].recon_positivity is False
    assert captured["align_cfg"].lbfgs_memory_size == 9
    assert captured["align_resume_state"] is None
    assert captured["align_checkpoint_callback"] is None
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
    assert manifest["cli_args"]["config"] == str(config_path)
    assert manifest["cli_args"]["save_manifest"] == str(manifest_path)
    assert manifest["resolved_config"]["output_path"] == str(out_path)
    assert manifest["resolved_config"]["config_path"] == str(config_path)
    assert manifest["resolved_config"]["config_file_values"]["gather_dtype"] == "bf16"
    assert manifest["resolved_config"]["config_file_values"]["bounds"] == [
        ["alpha", -0.05, 0.05],
        ["dx", -20.0, 20.0],
    ]
    assert manifest["resolved_config"]["config_file_values"]["pose_model"] == "spline"
    assert manifest["resolved_config"]["align_config"]["pose_model"] == "spline"
    assert "gather_dtype" in manifest["resolved_config"]["explicit_cli_keys"]
    assert manifest["resolved_config"]["effective_options"]["gather_dtype"] == "fp32"
    assert manifest["resolved_config"]["gather_dtype"] == "fp32"
    assert manifest["resolved_config"]["levels"] is None
    assert manifest["resolved_config"]["checkpoint_path"] is None
    assert manifest["resolved_config"]["checkpoint_every"] is None
    assert manifest["resolved_config"]["resume_path"] is None
    assert manifest["resolved_config"]["reconstruction_grid"]["nx"] == 2
    assert manifest["resolved_config"]["alignment_params_shape"] == [2, 5]
    assert manifest["versions"]["tomojax"]
    assert "created_at" in manifest
    assert "available" in manifest["jax"]
