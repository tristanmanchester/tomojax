from __future__ import annotations

import importlib
import json
from pathlib import Path

import imageio.v3 as iio
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.cli.main import main
import tomojax.cli.recon as recon_cli
import tomojax.cli.simulate as simulate_cli
from tomojax.geometry import Detector
from tomojax.io import load_dataset, save_dataset, save_projection_payload
from tomojax.recon.quicklook import scale_to_uint8

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


@pytest.mark.parametrize(
    ("omitted_flag", "expected"),
    [
        ("--flats", "requires --flats and --darks"),
        ("--darks", "requires --flats and --darks"),
        ("--angles", "requires --angles"),
    ],
)
def test_preprocess_cli_tiff_stack_requires_sidecars(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    omitted_flag: str,
    expected: str,
) -> None:
    projections = tmp_path / "projections"
    flats = tmp_path / "flats"
    darks = tmp_path / "darks"
    angles = tmp_path / "angles.csv"
    out_path = tmp_path / "corrected.nxs"
    write_tiff_stack(projections, [5.0, 9.0])
    write_tiff_stack(flats, [11.0])
    write_tiff_stack(darks, [1.0])
    write_angle_csv(angles, [0.0, 90.0])

    args = [
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
    index = args.index(omitted_flag)
    del args[index : index + 2]

    assert main(args) == 2
    captured = capsys.readouterr()
    assert expected in captured.err
    assert not out_path.exists()


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


def test_main_formats_expected_subcommand_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    write_projection_dataset(scan)

    def fail_expected(command: object, config_metadata: dict[str, object]) -> None:
        del command, config_metadata
        raise ValueError("bad detector metadata")

    monkeypatch.setattr(recon_cli, "_run_reconstruction", fail_expected)

    assert main(["recon", "--data", str(scan), "--out", str(recon)]) == 1
    captured = capsys.readouterr()
    assert captured.err == "ERROR: recon: bad detector metadata\n"


def test_main_does_not_swallow_programmer_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    write_projection_dataset(scan)

    def fail_programmer(command: object, config_metadata: dict[str, object]) -> None:
        del command, config_metadata
        raise TypeError("wrong internal call shape")

    monkeypatch.setattr(recon_cli, "_run_reconstruction", fail_programmer)

    with pytest.raises(TypeError, match="wrong internal call shape"):
        main(["recon", "--data", str(scan), "--out", str(recon)])


def test_recon_cli_executes_fbp_and_writes_volume_metadata(tmp_path: Path) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    manifest = tmp_path / "recon-manifest.json"
    write_projection_dataset(scan)

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
                "--views-per-batch",
                "1",
                "--no-checkpoint-projector",
                "--save-manifest",
                str(manifest),
            ]
        )
        == 0
    )

    loaded = load_dataset(recon)
    assert loaded.volume is not None
    assert loaded.volume.shape == (4, 4, 2)
    assert np.isfinite(loaded.volume).all()
    assert loaded.grid is not None
    assert loaded.grid.nx == 4
    assert loaded.grid.ny == 4
    assert loaded.grid.nz == 2
    assert loaded.detector is not None

    metadata = loaded.copy_metadata()
    assert metadata.frame == "sample"
    assert metadata.volume_axes_order == "zyx"
    assert loaded.geometry_metadata["detector_center_override"]["source"] == "metadata"

    resolved = json.loads(manifest.read_text(encoding="utf-8"))["resolved_config"]
    assert resolved["algorithm"] == "fbp"
    assert resolved["algorithm_config"]["filter"] == "ramp"
    assert resolved["reconstruction_grid"]["nx"] == 4
    assert resolved["reconstruction_grid"]["ny"] == 4
    assert resolved["reconstruction_grid"]["nz"] == 2
    assert resolved["roi"] == {
        "requested": "off",
        "is_parallel": True,
        "grid_changed": False,
    }
    assert resolved["volume_shape"] == [4, 4, 2]


def test_recon_cli_accepts_detector_center_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    write_projection_dataset(scan)
    captured: dict[str, object] = {}

    def fake_run(command: object, config_metadata: dict[str, object]) -> None:
        del config_metadata
        captured["det_u_px"] = command.det_u_px
        captured["det_v_px"] = command.det_v_px
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
                "--det-u-px",
                "6",
                "--det-v-px",
                "-1.5",
            ]
        )
        == 0
    )

    assert captured == {"det_u_px": 6.0, "det_v_px": -1.5}


def test_recon_cli_rejects_nonfinite_detector_center_override(tmp_path: Path) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    write_projection_dataset(scan)

    with pytest.raises(SystemExit) as exc:
        main(["recon", "--data", str(scan), "--out", str(recon), "--det-u-px", "nan"])

    assert exc.value.code == 2


def test_recon_detector_center_override_records_effective_pixels() -> None:
    # check-public-imports: allow-private
    from tomojax.cli._recon_plan import _apply_detector_center_override

    detector = Detector(nu=8, nv=8, du=0.5, dv=2.0, det_center=(1.0, -2.0))
    geometry_meta: dict[str, object] = {"detector": detector.to_dict()}

    updated, provenance = _apply_detector_center_override(
        detector,
        geometry_meta,
        det_u_px=6.0,
        det_v_px=None,
    )

    assert updated.det_center == pytest.approx((3.0, -2.0))
    assert provenance["source"] == "cli_override"
    assert provenance["requested_px"] == {"det_u_px": 6.0, "det_v_px": None}
    assert provenance["effective_px"] == {"det_u_px": 6.0, "det_v_px": -1.0}
    assert provenance["effective_world"] == {"det_u": 3.0, "det_v": -2.0}
    assert geometry_meta["detector"]["det_center"] == [3.0, -2.0]


def test_slices_cli_extracts_labelled_planes_without_full_cli_recon(tmp_path: Path) -> None:
    path = tmp_path / "recon.nxs"
    out_dir = tmp_path / "slices"
    dataset = make_projection_dataset()
    dataset.volume = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
    save_dataset(path, dataset)

    assert main(["slices", "--data", str(path), "--out", str(out_dir), "--prefix", "demo"]) == 0

    assert (out_dir / "demo_z0001.png").is_file()
    assert (out_dir / "demo_y0002.png").is_file()
    assert (out_dir / "demo_x0002.png").is_file()
    summary = json.loads((out_dir / "demo_slices.json").read_text(encoding="utf-8"))
    assert summary["saved_axes"] == "zyx"
    assert summary["slices"]["z"]["display_axes"] == "yx"


def test_slices_cli_extracts_requested_planes_with_xyz_disk_axes(tmp_path: Path) -> None:
    path = tmp_path / "recon.nxs"
    out_dir = tmp_path / "slices"
    volume = np.zeros((3, 4, 5), dtype=np.float32)
    for x in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for z in range(volume.shape[2]):
                volume[x, y, z] = (100 * x) + (10 * y) + z
    dataset = make_projection_dataset()
    dataset.volume = volume
    metadata = dataset.to_nxtomo_metadata()
    metadata.volume_axes_order = "xyz"
    save_projection_payload(path, projections=dataset.projections, metadata=metadata)

    assert (
        main(
            [
                "slices",
                "--data",
                str(path),
                "--out",
                str(out_dir),
                "--prefix",
                "demo",
                "--z",
                "2",
                "--y",
                "1",
                "--x",
                "2",
                "--lower-percentile",
                "0",
                "--upper-percentile",
                "100",
            ]
        )
        == 0
    )

    z_image = iio.imread(out_dir / "demo_z0002.png")
    y_image = iio.imread(out_dir / "demo_y0001.png")
    x_image = iio.imread(out_dir / "demo_x0002.png")
    np.testing.assert_array_equal(
        z_image,
        scale_to_uint8(volume[:, :, 2].T, lower_percentile=0, upper_percentile=100),
    )
    np.testing.assert_array_equal(
        y_image,
        scale_to_uint8(volume[:, 1, :].T, lower_percentile=0, upper_percentile=100),
    )
    np.testing.assert_array_equal(
        x_image,
        scale_to_uint8(volume[2, :, :].T, lower_percentile=0, upper_percentile=100),
    )


def test_slices_cli_requires_force_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "recon.nxs"
    out_dir = tmp_path / "slices"
    dataset = make_projection_dataset()
    dataset.volume = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
    save_dataset(path, dataset)

    assert main(["slices", "--data", str(path), "--out", str(out_dir), "--prefix", "demo"]) == 0
    with pytest.raises(SystemExit) as exc:
        main(["slices", "--data", str(path), "--out", str(out_dir), "--prefix", "demo"])

    assert exc.value.code == 2
    assert (
        main(["slices", "--data", str(path), "--out", str(out_dir), "--prefix", "demo", "--force"])
        == 0
    )


def test_slices_cli_reports_missing_file_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["slices", "--data", str(tmp_path / "missing.nxs"), "--out", str(tmp_path)])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "could not read" in captured.err
    assert "Traceback" not in captured.err


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
    align_cli_main = importlib.import_module("tomojax.cli.align.main")
    align_cli_plan = importlib.import_module("tomojax.cli.align.plan")
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
        config,
        resume_state=None,
        checkpoint_callback=None,
    ):
        del geom, recon_detector, resume_state, checkpoint_callback
        calls.append((tuple(projections.shape), list(factors), config.schedule))
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
    align_cli_main = importlib.import_module("tomojax.cli.align.main")
    align_cli_plan = importlib.import_module("tomojax.cli.align.plan")
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
        config,
        resume_state=None,
        checkpoint_callback=None,
    ):
        del geom, recon_detector, projections, resume_state, checkpoint_callback
        calls.append(
            (list(factors), tuple(config.optimise_dofs or ()), tuple(config.freeze_dofs or ()))
        )
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


def test_align_cli_cor_then_pose_defaults_to_full_pyramid_and_accepts_normal_quality(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    align_cli_main = importlib.import_module("tomojax.cli.align.main")
    align_cli_plan = importlib.import_module("tomojax.cli.align.plan")
    scan = tmp_path / "scan.nxs"
    aligned = tmp_path / "aligned.nxs"
    write_projection_dataset(scan)
    calls: list[tuple[list[int], str, str]] = []

    def fake_align_multires(
        geom,
        recon_grid,
        recon_detector,
        projections,
        *,
        factors,
        config,
        resume_state=None,
        checkpoint_callback=None,
    ):
        del geom, recon_detector, resume_state, checkpoint_callback
        calls.append((list(factors), config.schedule, config.align_profile))
        x = jnp.zeros((recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32)
        params5 = jnp.zeros((int(projections.shape[0]), 5), dtype=jnp.float32)
        return x, params5, {"loss": [0.0], "outer_stats": [], "active_dofs": []}

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
                "cor_then_pose",
                "--quality",
                "normal",
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
            ]
        )
        == 0
    )

    assert calls == [([4, 2, 1], "cor_then_pose", "tortoise")]


@pytest.mark.parametrize(
    ("mode", "expected_schedule", "expected_levels"),
    [
        ("auto", "setup_safe", [4, 2, 1]),
        ("max", "setup_safe", [4, 2, 1]),
        ("cor_then_pose", "cor_then_pose", [4, 2, 1]),
    ],
)
def test_align_cli_print_plan_json_reports_effective_public_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    expected_schedule: str,
    expected_levels: list[int],
) -> None:
    align_cli_main = importlib.import_module("tomojax.cli.align.main")
    scan = tmp_path / "scan.nxs"
    aligned = tmp_path / "aligned.nxs"
    write_projection_dataset(scan)

    monkeypatch.setattr(align_cli_main, "setup_logging", lambda: None)
    monkeypatch.setattr(align_cli_main, "log_jax_env", lambda: None)
    monkeypatch.setattr(align_cli_main, "init_jax_compilation_cache", lambda: None)

    assert (
        main(
            [
                "align",
                "--data",
                str(scan),
                "--out",
                str(aligned),
                "--mode",
                mode,
                "--quality",
                "normal",
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
                "--loss-schedule",
                "4:phasecorr,2:ssim,1:l2_otsu",
                "--print-plan-json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["schedule"] == expected_schedule
    assert payload["levels"] == expected_levels
    assert payload["output_path"] == str(aligned)
    assert [stage["stage_name"] for stage in payload["stages"]][:2] == (
        ["cor", "pose_polish"] if mode == "cor_then_pose" else ["cor", "detector_roll"]
    )
