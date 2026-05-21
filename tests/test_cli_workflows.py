from __future__ import annotations

import importlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.cli.main import main
import tomojax.cli.recon as recon_cli
import tomojax.cli.simulate as simulate_cli
from tomojax.io import load_dataset, save_dataset

from ._helpers import (
    make_projection_dataset,
    write_angle_csv,
    write_projection_dataset,
    write_tiff_stack,
)


def test_inspect_and_validate_cli_on_product_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "scan.nxs"
    report = tmp_path / "inspect.json"
    write_projection_dataset(path)

    assert main(["validate", str(path)]) == 0
    assert main(["inspect", str(path), "--json", str(report)]) == 0

    captured = capsys.readouterr()
    assert f"OK: {path}" in captured.out
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["projection"]["shape"] == [2, 2, 4]
    assert payload["angles"]["coverage_deg"] == 90.0


def test_ingest_cli_writes_standard_dataset_from_tiffs(tmp_path: Path) -> None:
    stack = tmp_path / "stack"
    angles = tmp_path / "angles.csv"
    out_path = tmp_path / "ingested.nxs"
    write_tiff_stack(stack, [1.0, 2.0], shape=(2, 4))
    write_angle_csv(angles, [0.0, 90.0])

    assert (
        main(
            [
                "ingest",
                str(stack),
                "--angles",
                str(angles),
                "--out",
                str(out_path),
                "--du",
                "0.5",
                "--dv",
                "0.75",
                "--grid",
                "4",
                "4",
                "2",
            ]
        )
        == 0
    )

    dataset = load_dataset(out_path)
    assert dataset.projections.shape == (2, 2, 4)
    assert dataset.detector is not None
    assert dataset.detector.du == pytest.approx(0.5)
    assert dataset.detector.dv == pytest.approx(0.75)
    assert dataset.grid is not None
    assert dataset.grid.nz == 2


def test_preprocess_cli_handles_tiff_stack_workflow(tmp_path: Path) -> None:
    projections = tmp_path / "projections"
    flats = tmp_path / "flats"
    darks = tmp_path / "darks"
    angles = tmp_path / "angles.csv"
    out_path = tmp_path / "corrected.nxs"
    write_tiff_stack(projections, [5.0, 9.0])
    write_tiff_stack(flats, [11.0])
    write_tiff_stack(darks, [1.0])
    write_angle_csv(angles, [0.0, 90.0])

    assert (
        main(
            [
                "preprocess",
                str(projections),
                str(out_path),
                "--format",
                "tiff-stack",
                "--flats",
                str(flats),
                "--darks",
                str(darks),
                "--angles",
                str(angles),
            ]
        )
        == 0
    )

    dataset = load_dataset(out_path)
    np.testing.assert_allclose(dataset.projections[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6)


def test_convert_cli_roundtrips_nxtomo_to_npz(tmp_path: Path) -> None:
    nxs_path = tmp_path / "scan.nxs"
    npz_path = tmp_path / "scan.npz"
    write_projection_dataset(nxs_path)

    assert main(["convert", "--in", str(nxs_path), "--out", str(npz_path)]) == 0

    dataset = load_dataset(npz_path)
    assert dataset.projections.shape == (2, 2, 4)
    np.testing.assert_allclose(dataset.angles_deg, [0.0, 90.0])


def test_recon_cli_routes_tiny_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    write_projection_dataset(scan)
    captured: dict[str, object] = {}

    def fake_run(command: object, config_metadata: dict[str, object]) -> None:
        captured["algo"] = command.algo
        captured["data"] = command.data
        captured["out"] = command.out
        assert config_metadata["config_path"] is None
        dataset = load_dataset(command.data)
        dataset.volume = np.zeros((4, 4, 2), dtype=np.float32)
        save_dataset(command.out, dataset)

    monkeypatch.setattr(recon_cli, "_run_reconstruction", fake_run)

    assert (
        main(
            [
                "recon",
                "--data",
                str(scan),
                "--out",
                str(recon),
                "--algo",
                "fbp",
                "--roi",
                "off",
                "--grid",
                "4",
                "4",
                "2",
            ]
        )
        == 0
    )

    assert captured == {"algo": "fbp", "data": str(scan), "out": str(recon)}
    loaded = load_dataset(recon)
    assert loaded.volume is not None
    assert loaded.volume.shape == (4, 4, 2)


def test_simulate_cli_routes_loadable_synthetic_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_path = tmp_path / "synthetic.nxs"
    captured: dict[str, object] = {}

    def fake_simulate_to_file(config: object, out: str) -> str:
        captured["shape"] = (config.nx, config.ny, config.nz, config.nu, config.nv, config.n_views)
        dataset = make_projection_dataset(
            projections=np.zeros(
                (int(config.n_views), int(config.nv), int(config.nu)), dtype=np.float32
            ),
            angles_deg=np.linspace(
                0.0, 180.0, int(config.n_views), endpoint=False, dtype=np.float32
            ),
        )
        dataset.volume = np.zeros(
            (int(config.nx), int(config.ny), int(config.nz)), dtype=np.float32
        )
        save_dataset(out, dataset)
        return out

    monkeypatch.setattr(simulate_cli, "simulate_to_file", fake_simulate_to_file)

    assert (
        main(
            [
                "simulate",
                "--out",
                str(out_path),
                "--nx",
                "2",
                "--ny",
                "2",
                "--nz",
                "2",
                "--nu",
                "2",
                "--nv",
                "2",
                "--n-views",
                "8",
                "--phantom",
                "sphere",
                "--no-single-rotate",
            ]
        )
        == 0
    )

    assert captured["shape"] == (2, 2, 2, 2, 2, 8)
    loaded = load_dataset(out_path)
    assert loaded.projections.shape == (8, 2, 2)
    assert loaded.volume is not None


def test_align_cli_mode_cor_writes_alignment_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    align_cli_main = importlib.import_module("tomojax.cli._align_main")
    align_cli_plan = importlib.import_module("tomojax.cli._align_plan")
    scan = tmp_path / "scan.nxs"
    aligned = tmp_path / "aligned.nxs"
    manifest = tmp_path / "align.json"
    write_projection_dataset(scan)
    calls: list[tuple[tuple[int, int, int], list[int], str]] = []

    def fake_align_multires(
        geom,
        recon_grid,
        recon_detector,
        projections,
        *,
        factors,
        cfg,
        resume_state=None,
        checkpoint_callback=None,
    ):
        del geom, recon_detector, resume_state, checkpoint_callback
        calls.append((tuple(projections.shape), list(factors), cfg.schedule))
        x = jnp.zeros((recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32)
        params5 = jnp.zeros((int(projections.shape[0]), 5), dtype=jnp.float32)
        return x, params5, {"loss": [0.0], "outer_stats": [], "active_dofs": ["det_u"]}

    monkeypatch.setattr(align_cli_main, "setup_logging", lambda: None)
    monkeypatch.setattr(align_cli_main, "log_jax_env", lambda: None)
    monkeypatch.setattr(align_cli_main, "init_jax_compilation_cache", lambda: None)
    monkeypatch.setattr(align_cli_plan, "align_multires", fake_align_multires)

    assert (
        main(
            [
                "align",
                "--data",
                str(scan),
                "--out",
                str(aligned),
                "--mode",
                "cor",
                "--roi",
                "off",
                "--grid",
                "4",
                "4",
                "2",
                "--outer-iters",
                "1",
                "--recon-iters",
                "1",
                "--views-per-batch",
                "1",
                "--save-manifest",
                str(manifest),
            ]
        )
        == 0
    )

    assert calls == [((2, 2, 4), [1], "cor")]
    loaded = load_dataset(aligned)
    assert loaded.volume is not None
    assert loaded.volume.shape == (4, 4, 2)
    assert loaded.align_params is not None
    assert loaded.align_params.shape == (2, 5)
    assert manifest.stat().st_size > 0


def test_align_cli_geometry_dofs_route_to_multires_without_explicit_levels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    align_cli_main = importlib.import_module("tomojax.cli._align_main")
    align_cli_plan = importlib.import_module("tomojax.cli._align_plan")
    scan = tmp_path / "scan.nxs"
    aligned = tmp_path / "aligned.nxs"
    write_projection_dataset(scan)
    calls: list[tuple[list[int], tuple[str, ...], tuple[str, ...]]] = []

    def fake_align_multires(
        geom,
        recon_grid,
        recon_detector,
        projections,
        *,
        factors,
        cfg,
        resume_state=None,
        checkpoint_callback=None,
    ):
        del geom, recon_detector, projections, resume_state, checkpoint_callback
        calls.append((list(factors), tuple(cfg.optimise_dofs or ()), tuple(cfg.freeze_dofs or ())))
        x = jnp.zeros((recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32)
        params5 = jnp.zeros((2, 5), dtype=jnp.float32)
        return (
            x,
            params5,
            {"loss": [0.0], "outer_stats": [], "active_geometry_dofs": ["det_u_px"]},
        )

    monkeypatch.setattr(align_cli_main, "setup_logging", lambda: None)
    monkeypatch.setattr(align_cli_main, "log_jax_env", lambda: None)
    monkeypatch.setattr(align_cli_main, "init_jax_compilation_cache", lambda: None)
    monkeypatch.setattr(align_cli_plan, "align_multires", fake_align_multires)

    assert (
        main(
            [
                "align",
                "--data",
                str(scan),
                "--out",
                str(aligned),
                "--optimise-dofs",
                "det_u_px",
                "--roi",
                "off",
                "--grid",
                "4",
                "4",
                "2",
                "--outer-iters",
                "1",
                "--recon-iters",
                "1",
                "--views-per-batch",
                "1",
            ]
        )
        == 0
    )

    assert calls == [([1], ("det_u_px",), ())]
