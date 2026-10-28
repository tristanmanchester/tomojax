from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import h5py
import imageio.v3 as iio
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.cli.main import main
import tomojax.cli.simulate as simulate_cli
from tomojax.geometry import stack_view_poses
from tomojax.geometry.api import CalibrationState, CalibrationVariable
from tomojax.io import build_geometry_from_dataset_metadata, load_dataset, save_dataset
from tomojax.io.api import save_nxtomo
from tomojax.recon.quicklook import scale_to_uint8

from ._helpers import (
    make_projection_dataset,
    write_angle_csv,
    write_projection_dataset,
    write_tiff_stack,
)

pytestmark = pytest.mark.surface


def test_preprocess_rejects_unverified_vendor_hardening(capsys):
    with pytest.raises(SystemExit) as error:
        main(["preprocess", "raw.nxs", "-o", "scan.nxs", "--beam-hardening", "recorded"])
    assert error.value.code == 2
    assert "unrecognized arguments: --beam-hardening recorded" in capsys.readouterr().err


def test_inspect_cli_describes_and_checks_a_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "scan.nxs"
    write_projection_dataset(path)

    assert main(["inspect", str(path)]) == 0
    assert "Valid: yes" in capsys.readouterr().out
    assert main(["inspect", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["issues"] == []
    assert payload["projection"]["shape"] == [2, 2, 4]
    assert payload["angles"]["coverage_deg"] == 90.0
    assert payload["volume"]["found"] is False


def test_inspect_raw_acquisition_reports_required_cli_inputs(tmp_path, capsys):
    path = tmp_path / "raw.h5"
    with h5py.File(path, "w") as file:
        file["exchange/data"] = np.ones((2, 2, 4), np.uint16)
        file["exchange/theta"] = [0, 90]
    assert main(["inspect", str(path), "--json"]) == 1
    issues = json.loads(capsys.readouterr().out)["issues"]
    assert len(issues) == 3
    assert "detector" in issues[0] and "--config" in issues[0]
    assert "--flats" in issues[1] and "--darks" in issues[2]
    with h5py.File(path, "a") as file:
        file["exchange/data_white"] = np.ones((1, 2, 4))
        file["exchange/data_dark"] = np.zeros((1, 2, 4))
    assert main(["inspect", str(path), "--json"]) == 1
    assert len(json.loads(capsys.readouterr().out)["issues"]) == 1


@pytest.mark.parametrize("units", ["degrees", "rad"])
def test_inspection_angle_range_uses_degrees_of_sample_frames_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], units: str
) -> None:
    path = tmp_path / "scan.nxs"
    angles = np.asarray([30.0, 999.0, 60.0, -400.0])
    write_projection_dataset(path, projections=np.ones((4, 2, 4), np.float32), angles=angles)
    with h5py.File(path, "a") as handle:
        handle["entry/instrument/detector/image_key"][...] = [0, 3, 0, 3]
        stored = handle["entry/sample/transformations/rotation_angle"]
        stored[...] = np.radians(angles) if units == "rad" else angles
        stored.attrs["units"] = units
    assert main(["inspect", str(path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["issues"] == []
    assert report["angles"]["count"] == 4
    assert report["angles"]["min_deg"] == pytest.approx(30.0, abs=1e-5)
    assert report["angles"]["max_deg"] == pytest.approx(60.0, abs=1e-5)


def test_inspect_cli_exits_1_for_an_invalid_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "scan.nxs"
    write_projection_dataset(path)
    with h5py.File(path, "a") as handle:
        del handle["/entry/sample/transformations/rotation_angle"]

    assert main(["inspect", str(path)]) == 1
    assert "Issues (" in capsys.readouterr().out


def test_inspect_cli_previews_a_reconstruction(tmp_path: Path) -> None:
    path = tmp_path / "recon.nxs"
    dataset = make_projection_dataset()
    dataset.volume = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
    save_dataset(path, dataset)
    previews = tmp_path / "previews"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tomojax.cli.main import main; "
            "assert main(sys.argv[1:]) == 0; assert 'jax' not in sys.modules",
            "inspect",
            str(path),
            "--preview",
            str(previews),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    names = sorted(p.name for p in previews.iterdir())
    assert names == ["projection.png", "slice_x.png", "slice_y.png", "slice_z.png"]
    with pytest.raises(SystemExit) as exc_info:
        main(["inspect", str(path), "--preview", str(previews)])
    assert exc_info.value.code == 2
    assert main(["inspect", str(path), "--preview", str(previews), "--force"]) == 0


def test_import_cli_writes_standard_dataset_from_tiffs(tmp_path: Path) -> None:
    stack = tmp_path / "stack"
    angles = tmp_path / "angles.csv"
    out_path = tmp_path / "imported.nxs"
    write_tiff_stack(stack, [1.0, 2.0], shape=(2, 4))
    write_angle_csv(angles, [0.0, 90.0])

    config = tmp_path / "calibration.toml"
    config.write_text('detector_center = [0.12, -0.3]\ngeometry = "parallel"\n')
    args = [
        "import",
        str(stack),
        "--angles",
        str(angles),
        "-o",
        str(out_path),
        "--config",
        str(config),
    ]
    assert main([*args, "--pixel-size", "0.5", "0.75", "--name", "rock"]) == 0

    dataset = load_dataset(out_path)
    assert dataset.projections.shape == (2, 2, 4)
    assert dataset.detector is not None
    assert dataset.detector.du == pytest.approx(0.5)
    assert dataset.detector.dv == pytest.approx(0.75)
    assert dataset.detector.center == (0.12, -0.3)
    assert dataset.sample_name == "rock"

    with pytest.raises(SystemExit) as exc_info:
        main(args)
    assert exc_info.value.code == 2
    with pytest.raises(SystemExit):
        main([*args, "--force"])  # missing pixel size is not unit calibration
    assert main([*args, "--force", "--pixel-size", "1"]) == 0
    assert load_dataset(out_path).detector.du == pytest.approx(1.0)  # pyright: ignore[reportOptionalMemberAccess]


def test_import_requires_explicit_detector_center_model_and_lamino_tilt(tmp_path, capsys):
    write_tiff_stack(tmp_path / "views", [1.0, 2.0], shape=(2, 4))
    write_angle_csv(tmp_path / "angles.csv", [0, 90])
    config, output = tmp_path / "calibration.toml", tmp_path / "scan.nxs"
    args = [
        "import",
        str(tmp_path / "views"),
        "--angles",
        str(tmp_path / "angles.csv"),
        "--pixel-size",
        ".5",
        "--config",
        str(config),
        "-o",
        str(output),
    ]
    for text, required in (
        ("", "--geometry"),
        ('geometry = "parallel"', "detector_center"),
        ('geometry = "lamino"\ndetector_center = [0, 0]', "tilt_deg"),
    ):
        config.write_text(text)
        with pytest.raises(SystemExit):
            main(args)
        assert required in capsys.readouterr().err and not output.exists()
    config.write_text(config.read_text() + "\ntilt_deg = 47.0\n")
    assert main(args) == 0
    assert load_dataset(output).geometry_inputs()["tilt_deg"] == 47


def test_import_cli_needs_angles_for_tiffs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stack = tmp_path / "stack"
    write_tiff_stack(stack, [1.0, 2.0], shape=(2, 4))
    with pytest.raises(SystemExit) as exc_info:
        main(["import", str(stack), "-o", str(tmp_path / "scan.nxs")])
    assert exc_info.value.code == 2
    assert "TIFF stacks need --angles" in capsys.readouterr().err


def test_import_cli_converts_nxtomo_to_npz(tmp_path: Path) -> None:
    nxs_path = tmp_path / "scan.nxs"
    npz_path = tmp_path / "scan.npz"
    write_projection_dataset(nxs_path)

    assert main(["import", str(nxs_path), "-o", str(npz_path)]) == 0

    dataset = load_dataset(npz_path)
    assert dataset.projections.shape == (2, 2, 4)
    np.testing.assert_allclose(dataset.angles, [0.0, 90.0])


def _fake_reconstruct(
    monkeypatch: pytest.MonkeyPatch, error: Exception | None = None
) -> dict[str, object]:
    """Make ``tj.reconstruct`` return zeros (or raise ``error``); record its call."""
    import tomojax

    calls: dict[str, object] = {}

    def fake(scan, method, *, grid, config, **keywords):
        if error is not None:
            raise error
        calls.update(method=method, grid=grid, config=config, **keywords)
        volume = np.zeros((grid.nx, grid.ny, grid.nz), np.float32)
        return tomojax.Reconstruction(volume, scan, method, {"config": config})

    monkeypatch.setattr(tomojax, "reconstruct", fake)
    return calls


def test_recon_cli_routes_tiny_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    write_projection_dataset(scan)
    calls = _fake_reconstruct(monkeypatch)

    args = ["recon", str(scan), "-o", str(recon), "--method", "cgls", "--roi", "off"]
    assert main([*args, "--grid", "4", "4", "2", "--iterations", "7", "--warm-start"]) == 0

    assert calls.pop("method") == "cgls"
    assert (calls.pop("grid").nx, calls.pop("config").iterations) == (4, 50)
    # Options not given are left to the configuration.
    assert calls == {
        "filter": None,
        "iterations": 7,
        "tv_weight": None,
        "nonnegative": None,
        "warm_start": True,
        "seed": None,
    }
    loaded = load_dataset(recon)
    assert loaded.volume is not None
    assert loaded.volume.shape == (4, 4, 2)


def test_main_formats_expected_subcommand_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    write_projection_dataset(scan)
    _ = _fake_reconstruct(monkeypatch, ValueError("bad detector metadata"))

    assert main(["recon", str(scan), "-o", str(recon)]) == 1
    captured = capsys.readouterr()
    assert captured.err == "tomojax recon: error: bad detector metadata\n"


def test_main_does_not_swallow_programmer_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    write_projection_dataset(scan)
    _ = _fake_reconstruct(monkeypatch, TypeError("wrong internal call shape"))

    with pytest.raises(TypeError, match="wrong internal call shape"):
        main(["recon", str(scan), "-o", str(recon)])


def test_recon_cli_executes_fbp_and_writes_volume_metadata(tmp_path: Path) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    manifest = tmp_path / "recon-manifest.json"
    config = tmp_path / "recon.toml"
    write_projection_dataset(scan)
    _ = config.write_text("views_per_batch = 1\nscale = 0.5\n", encoding="utf-8")

    args = [
        "recon",
        str(scan),
        "-o",
        str(recon),
        "--method",
        "fbp",
        "--roi",
        "off",
        "--grid",
        "4",
        "4",
        "2",
    ]
    assert main([*args, "--config", str(config), "--manifest", str(manifest)]) == 0

    loaded = load_dataset(recon)
    assert loaded.volume is not None
    assert loaded.volume.shape == (4, 4, 2)
    assert np.isfinite(loaded.volume).all()
    assert loaded.grid is not None
    assert (loaded.grid.nx, loaded.grid.ny, loaded.grid.nz) == (4, 4, 2)
    assert loaded.detector is not None
    assert loaded.copy_metadata().volume_axes_order == "zyx"
    assert loaded.geometry_metadata["reconstruction_method"] == "fbp"

    resolved = json.loads(manifest.read_text(encoding="utf-8"))["resolved_config"]
    assert resolved["method"] == "fbp"
    assert resolved["settings"] == {"views_per_batch": 1, "scale": 0.5}
    settings = resolved["reconstruction"]["config"]
    assert (settings["filter"], settings["views_per_batch"], settings["scale"]) == ("ramp", 1, 0.5)
    assert resolved["reconstruction_grid"]["nx"] == 4
    assert resolved["roi"] == {
        "requested": "off",
        "grid_changed": False,
        "cylindrical_output_mask": False,
    }
    assert resolved["volume_shape"] == [4, 4, 2]


@pytest.mark.parametrize("warm_start", [False, True])
def test_recon_cli_runs_cgls_like_the_python_solver(tmp_path: Path, *, warm_start: bool) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    manifest = tmp_path / "recon-manifest.json"
    write_projection_dataset(scan)
    args = [
        "--roi",
        "off",
        "--grid",
        "4",
        "4",
        "2",
        "--iterations",
        "6",
        *(["--warm-start"] if warm_start else []),
    ]
    command = ["recon", str(scan), "-o", str(recon), "--method", "cgls"]
    assert main([*command, *args, "--manifest", str(manifest)]) == 0
    loaded = load_dataset(recon)
    assert loaded.volume is not None
    assert loaded.volume.shape == (4, 4, 2)
    resolved = json.loads(manifest.read_text(encoding="utf-8"))["resolved_config"]
    assert resolved["method"] == "cgls"
    assert resolved["reconstruction"]["warm_start"] == warm_start
    assert resolved["reconstruction"]["config"]["iterations"] == 6
    assert 1 <= resolved["reconstruction"]["effective_iterations"] <= 6


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_recon_cli_auto_roi_does_not_depend_on_physical_units(tmp_path: Path, scale: float) -> None:
    import tomojax as tj

    grid = tj.Grid(11, 11, 5, scale, scale, scale)
    detector = tj.Detector(9, 3, scale, scale)
    geometry = tj.ParallelGeometry(grid, detector, np.array([0.0, 90.0]))
    scan_path, recon_path = tmp_path / "scan.nxs", tmp_path / "recon.nxs"
    tj.save(scan_path, tj.Scan(np.zeros((2, 3, 9), np.float32), geometry))

    args = ["recon", str(scan_path), "-o", str(recon_path), "--method", "cgls"]
    assert main([*args, "--iterations", "0"]) == 0

    actual = tj.load_reconstruction(recon_path)
    assert (actual.grid.nx, actual.grid.ny, actual.grid.nz) == (9, 9, 3)
    assert (actual.grid.vx, actual.grid.vy, actual.grid.vz) == (scale, scale, scale)
    np.testing.assert_array_equal(actual.volume, np.zeros((9, 9, 3), np.float32))


def test_recon_cli_cylinder_constrains_fista_to_the_field_of_view(tmp_path: Path) -> None:
    import numpy as np

    import tomojax as tj
    from tomojax.recon import FistaConfig

    scan_path, recon_path = tmp_path / "scan.nxs", tmp_path / "recon.nxs"
    write_projection_dataset(scan_path)
    command = ["recon", str(scan_path), "-o", str(recon_path), "--method", "fista"]
    assert main([*command, "--roi", "cyl", "--iterations", "4"]) == 0

    scan = tj.load(scan_path)
    # The CLI's grid for --roi cyl, and the cylinder as FISTA's support.
    grid = tj.load_reconstruction(recon_path).grid
    # check-public-imports: allow-private
    from tomojax.cli._reconstruction_region import cylinder_support

    # The support constrains FISTA's updates, and the saved volume is zero outside.
    support = cylinder_support(grid, scan.detector)
    config = FistaConfig(support=support)
    expected = tj.reconstruct(scan, "fista", grid=grid, config=config, iterations=4).volume
    actual = tj.load_reconstruction(recon_path).volume
    np.testing.assert_allclose(actual, np.asarray(expected) * support, rtol=1e-5, atol=1e-6)


def test_recon_cli_config_settings_are_fields_of_the_methods_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scan = tmp_path / "scan.nxs"
    config = tmp_path / "recon.toml"
    write_projection_dataset(scan)
    calls = _fake_reconstruct(monkeypatch)
    _ = config.write_text(
        'regulariser = "huber_tv"\nhuber_delta = 0.05\ntv_weight = 0.01\n', encoding="utf-8"
    )

    args = ["recon", str(scan), "-o", str(tmp_path / "out.nxs"), "--method", "fista"]
    assert main([*args, "--config", str(config), "--roi", "off"]) == 0

    cfg = calls["config"]
    assert (cfg.regulariser, cfg.huber_delta, cfg.iterations) == ("huber_tv", 0.05, 50)
    # A key that is an option is passed as the keyword, which tj.reconstruct applies.
    assert calls["tv_weight"] == 0.01


@pytest.mark.parametrize(
    ("toml", "options", "message"),
    [
        (
            "tv_prox_iterations = 3\n",
            ["--method", "fbp"],
            "method 'fbp' does not take tv_prox_iterations",
        ),
        ("", ["--method", "fbp", "--iterations", "5"], "method 'fbp' does not take iterations"),
        ("", ["--method", "fbp", "--warm-start"], "method 'fbp' does not take warm_start"),
        ("tau = 0.1\n", ["--method", "fista"], "method 'fista' does not take tau"),
    ],
)
def test_recon_cli_refuses_settings_the_method_does_not_take(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    toml: str,
    options: list[str],
    message: str,
) -> None:
    scan = tmp_path / "scan.nxs"
    config = tmp_path / "recon.toml"
    write_projection_dataset(scan)
    _ = config.write_text(toml, encoding="utf-8")

    args = ["recon", str(scan), "-o", str(tmp_path / "out.nxs"), "--config", str(config)]
    with pytest.raises(SystemExit) as exc:
        main([*args, *options])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert message in err
    if toml:
        assert "has " in err and "views_per_batch" in err  # the valid ones


def test_inspect_cli_previews_central_slices_of_xyz_disk_volumes(tmp_path: Path) -> None:
    path = tmp_path / "recon.nxs"
    previews = tmp_path / "previews"
    volume = np.zeros((3, 4, 5), dtype=np.float32)
    for x in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for z in range(volume.shape[2]):
                volume[x, y, z] = (100 * x) + (10 * y) + z
    dataset = make_projection_dataset()
    dataset.volume = volume
    metadata = dataset.to_nxtomo_metadata()
    metadata.volume_axes_order = "xyz"
    save_nxtomo(path, projections=dataset.projections, metadata=metadata)

    assert main(["inspect", str(path), "--preview", str(previews)]) == 0

    # Central slices, displayed as (y, x), (z, x) and (z, y).
    for name, plane in (
        ("slice_z", volume[:, :, 2].T),
        ("slice_y", volume[:, 2, :].T),
        ("slice_x", volume[1, :, :].T),
    ):
        np.testing.assert_array_equal(iio.imread(previews / f"{name}.png"), scale_to_uint8(plane))


def test_cli_reports_a_missing_input_as_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for command in (["inspect"], ["recon", "-o", str(tmp_path / "r.nxs")]):
        with pytest.raises(SystemExit) as exc:
            main([*command, str(tmp_path / "missing.nxs")])
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "input not found" in captured.err
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
            angles=np.linspace(0.0, 180.0, int(config.n_views), endpoint=False, dtype=np.float32),
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
                "-o",
                str(out_path),
                "--grid",
                "2",
                "2",
                "2",
                "--detector",
                "2",
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


def _fake_alignment(
    monkeypatch: pytest.MonkeyPatch,
    poses: np.ndarray | None = None,
    info: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Make ``align_multires`` return ``poses`` (none moved by default); record its calls."""
    calls: list[dict[str, object]] = []

    def fake_align_multires(geometry, grid, detector, projections, *, factors, config, **kwargs):
        del geometry, detector, kwargs
        calls.append(
            {"shape": tuple(projections.shape), "factors": list(factors), "config": config}
        )
        volume = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
        params = np.zeros((int(projections.shape[0]), 6), np.float32) if poses is None else poses
        return volume, jnp.asarray(params), {"loss": [0.0], "outer_stats": [], **(info or {})}

    monkeypatch.setattr("tomojax.alignment.api.align_multires", fake_align_multires)
    return calls


def test_align_cli_mode_cor_writes_alignment_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scan = tmp_path / "scan.nxs"
    aligned = tmp_path / "aligned.nxs"
    manifest = tmp_path / "align.json"
    write_projection_dataset(scan)
    calls = _fake_alignment(monkeypatch)

    args = ["align", str(scan), "-o", str(aligned), "--mode", "cor", "--roi", "off"]
    assert main([*args, "--grid", "4", "4", "2", "--manifest", str(manifest)]) == 0

    assert [(c["shape"], c["factors"], c["config"].schedule) for c in calls] == [
        ((2, 2, 4), [1], "cor")
    ]
    loaded = load_dataset(aligned)
    assert loaded.volume is not None
    assert loaded.volume.shape == (4, 4, 2)
    record = json.loads(manifest.read_text())["resolved_config"]
    assert record["alignment"]["mode"] == "cor"
    assert record["alignment"]["config"]["schedule"] == "cor"
    assert record["reconstruction_grid"]["nx"] == 4


def test_align_cli_config_settings_replace_fields_of_the_modes_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scan = tmp_path / "scan.nxs"
    config = tmp_path / "align.toml"
    write_projection_dataset(scan)
    _ = config.write_text('optimise_dofs = ["det_u_px"]\nouter_iterations = 3\n', encoding="utf-8")
    calls = _fake_alignment(monkeypatch)

    args = ["align", str(scan), "-o", str(tmp_path / "out.nxs"), "--mode", "pose", "--roi", "off"]
    assert main([*args, "--config", str(config)]) == 0

    ((call,),) = [calls]
    cfg = call["config"]
    assert call["factors"] == [1]
    assert (cfg.optimise_dofs, cfg.schedule, cfg.outer_iterations) == (("det_u_px",), None, 3)
    # The rest is pose mode's configuration.
    assert (cfg.gn_coupling, cfg.ray_integrator, cfg.pose_translation_frame) == (
        "joint",
        "joseph",
        "detector",
    )


def test_align_cli_cor_then_pose_saves_the_detector_centre_and_motion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scan = tmp_path / "scan.nxs"
    aligned = tmp_path / "aligned.nxs"
    angles = np.linspace(0.0, 360.0, 8, endpoint=False, dtype=np.float32)
    du = write_projection_dataset(
        scan,
        projections=np.ones((8, 2, 4), np.float32),
        angles=angles,
        geometry_type="lamino",
        geometry_metadata={"tilt_deg": 30.0, "tilt_about": "x"},
    ).detector.du
    offset_px, motion = 0.3, 0.05 * np.sin(np.deg2rad(3 * angles))
    # What align_multires returns: motion in the poses, the offset in the state.
    poses = np.zeros((8, 6), np.float32)
    poses[:, 3] = motion
    det_u = CalibrationVariable(
        name="det_u_px",
        value=offset_px,
        unit="native_detector_px",
        status="estimated",
        frame="detector",
        gauge="detector_ray_grid_center",
    )
    state = CalibrationState(detector=(det_u,)).to_dict()
    calls = _fake_alignment(monkeypatch, poses, {"geometry_calibration_state": state})

    args = ["align", str(scan), "-o", str(aligned), "--mode", "cor-then-pose"]
    assert main([*args, "--roi", "off"]) == 0

    ((call,),) = [calls]
    config = call["config"]
    assert call["factors"] == [1]
    assert config.schedule == "cor_then_pose"
    assert config.pose_translation_frame == "detector"
    assert config.gn_coupling == "joint"
    saved = load_dataset(aligned)
    assert saved.detector.center[0] == pytest.approx(offset_px * du, abs=1e-6)
    assert saved.align_gauge["pose_translation_frame"] == "detector"
    assert saved.align_params[:, 3] == pytest.approx(list(motion), abs=1e-6)
    # Only the detector centre changes; the scan geometry is kept.
    nominal = [
        build_geometry_from_dataset_metadata(load_dataset(path).geometry_inputs())[2]
        for path in (scan, aligned)
    ]
    np.testing.assert_allclose(
        stack_view_poses(nominal[1], 8), stack_view_poses(nominal[0], 8), atol=1e-6
    )


def test_align_cli_cor_then_pose_needs_detector_frame_translations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scan = tmp_path / "scan.nxs"
    config = tmp_path / "align.toml"
    write_projection_dataset(scan)
    _ = config.write_text('pose_translation_frame = "object"\n', encoding="utf-8")
    args = ["align", str(scan), "-o", str(tmp_path / "out.nxs"), "--mode", "cor-then-pose"]
    with pytest.raises(SystemExit) as exc:
        _ = main([*args, "--config", str(config)])
    assert exc.value.code == 2
    assert "pose_translation_frame='detector'" in capsys.readouterr().err


def test_aligning_a_file_with_poses_keeps_them_composed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tomojax as tj

    scan = tmp_path / "scan.nxs"
    angles = np.linspace(0.0, 180.0, 8, endpoint=False, dtype=np.float32)
    dataset = make_projection_dataset(projections=np.ones((8, 2, 4), np.float32), angles=angles)
    earlier = np.zeros((8, 6), np.float32)  # an earlier alignment's poses
    earlier[:, 2], earlier[:, 3] = 0.02, 0.3 * np.cos(np.deg2rad(angles))
    dataset.align_params, dataset.align_gauge = earlier, {"pose_translation_frame": "detector"}
    save_dataset(scan, dataset)
    correction = np.zeros((8, 6), np.float32)
    correction[:, 2], correction[:, 3] = 0.01, 0.1 * np.sin(np.deg2rad(angles))
    _ = _fake_alignment(monkeypatch, correction)

    out, fresh = tmp_path / "aligned.nxs", tmp_path / "fresh.nxs"
    assert main(["align", str(scan), "-o", str(out), "--mode", "pose", "--roi", "off"]) == 0
    assert (
        main(["align", str(scan), "-o", str(fresh), "--mode", "pose", "--roi", "off", "--no-poses"])
        == 0
    )

    # The command line corrects on top of the saved poses, as tomojax.align does.
    python = tj.align(tj.load(scan), mode="pose")
    composed = tj.load(out).poses
    assert composed is not None
    np.testing.assert_allclose(composed, python.poses, atol=1e-6)
    assert not np.allclose(composed, earlier, atol=1e-3)
    assert not np.allclose(composed, correction, atol=1e-3)
    # --no-poses starts from the nominal geometry.
    np.testing.assert_allclose(tj.load(fresh).poses, correction, atol=1e-6)


@pytest.mark.parametrize("quality", ["fast", "reference"])
def test_align_cli_dry_run_reports_the_effective_public_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quality: str
) -> None:
    scan = tmp_path / "scan.nxs"
    aligned = tmp_path / "aligned.nxs"
    config = tmp_path / "align.toml"
    write_projection_dataset(scan)
    _ = config.write_text(
        'loss = "4:phasecorr,2:ssim,1:l2_otsu"\nbackend = "cuda"\n', encoding="utf-8"
    )
    args = ["align", str(scan), "-o", str(aligned), "--mode", "full", "--quality", quality]
    assert main([*args, "--roi", "off", "--config", str(config), "--dry-run"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schedule"]["name"] == "setup_safe"
    assert payload["levels"] == [4, 2, 1]
    assert payload["output_path"] == str(aligned)
    assert payload["config"]["quality"] == quality
    assert payload["config"]["backend"] == "cuda"
    assert "phasecorr" in payload["loss"]
    stages = payload["schedule"]["stages"]
    assert [stage["stage_name"] for stage in stages][:2] == ["setup", "pose_polish"]
    assert not aligned.exists()


def _pose_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    config: str = "",
    *,
    geometry_type="parallel",
) -> object:
    scan = tmp_path / "scan.nxs"
    settings = tmp_path / "align.toml"
    beam = {"source_to_axis": 10.0, "source_to_detector": 20.0}
    write_projection_dataset(
        scan, geometry_type=geometry_type, geometry_metadata={"cone_beam": beam}
    )
    _ = settings.write_text(config, encoding="utf-8")
    args = ["align", str(scan), "-o", str(tmp_path / "out.nxs"), "--mode", "pose"]
    assert main([*args, "--roi", "off", "--config", str(settings), "--dry-run"]) == 0
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("geometry_type", ["parallel", "cone"])
def test_align_cli_pose_mode_defaults_to_the_coupled_solver(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], geometry_type: str
) -> None:
    plan = _pose_plan(tmp_path, capsys, geometry_type=geometry_type)
    config = plan["config"]
    assert plan["pose_solver"] == "coupled"
    assert plan["loss"] == "L2LossSpec()"
    assert (config["ray_integrator"], config["tv_weight"], config["gather_dtype"]) == (
        "joseph",
        0.0,
        "fp32",
    )
    assert config["outer_iterations"] == 30
    assert config["gn_joint_iterations"] == (10 if geometry_type == "cone" else 40)
    stages = plan["schedule"]["stages"]
    assert {stage["objective_kind"] for stage in stages} == {"joint_volume_pose"}


def test_align_cli_settings_select_the_alternating_pose_solver(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _pose_plan(tmp_path, capsys, 'gn_coupling = "fixed_volume"\nloss = "l2_otsu"\n')
    assert plan["loss"] == "L2OtsuLossSpec(temp=0.5)"
    stages = plan["schedule"]["stages"]
    assert {stage["objective_kind"] for stage in stages} == {"fixed_volume"}


def test_align_cli_rejects_settings_the_coupled_solver_cannot_use(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        _ = _pose_plan(tmp_path, capsys, 'pose_model = "spline"\n')
    assert exc.value.code == 2
    assert "joint GN requires" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["align", "recon"])
def test_cli_defaults_defer_policy_and_report_limitations(
    command: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import tomojax as tj

    path, output = tmp_path / "scan.nxs", tmp_path / "result.nxs"
    write_projection_dataset(path)
    automatic = {"status": "limited", "reasons": ["Rotation axis metadata is missing."]}

    def automatic_result(scan, *args, **kwargs):
        from tomojax.recon import FistaConfig

        assert kwargs["grid"] is None and kwargs["config"] is None
        assert (args[0] if command == "recon" else kwargs["mode"]) == "auto"
        volume = np.zeros((scan.grid.nx, scan.grid.ny, scan.grid.nz), np.float32)
        if command == "recon":
            return tj.Reconstruction(volume, scan, "fbp", {"automatic": automatic})
        info = {"automatic": automatic, "config": FistaConfig()}
        return tj.Alignment(volume, scan, np.zeros((len(scan.angles), 6)), info)

    monkeypatch.setattr(tj, "reconstruct" if command == "recon" else "align", automatic_result)
    manifest = tmp_path / "manifest.json"
    assert main([command, str(path), "-o", str(output), "--manifest", str(manifest)]) == 0
    assert load_dataset(output).volume is not None
    assert "Rotation axis metadata is missing" in caplog.text
    record = json.loads(manifest.read_text())["resolved_config"]
    assert record["reconstruction" if command == "recon" else "alignment"]["automatic"] == automatic


def test_cli_automatic_dry_run_and_expert_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, output = tmp_path / "scan.nxs", tmp_path / "result.nxs"
    write_projection_dataset(path)
    assert main(["align", str(path), "-o", str(output), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "data_dependent"
    assert not output.exists()
    config = tmp_path / "expert.toml"
    for command, key, option in [
        ("align", "outer_iterations", "mode"),
        ("recon", "views_per_batch", "method"),
    ]:
        config.write_text(f"{key} = 2\n")
        with pytest.raises(SystemExit) as error:
            main([command, str(path), "-o", str(output), "--config", str(config)])
        assert error.value.code == 2
        assert f"explicit --{option}" in capsys.readouterr().err


@pytest.mark.parametrize("layout", ["nxtomo", "exchange", "gda"])
def test_inspect_and_preview_share_raw_acquisition_discovery(tmp_path, capsys, layout):
    path = tmp_path / "raw.h5"
    sample = np.arange(8, dtype=np.uint16).reshape(2, 4) + 10
    with h5py.File(path, "w") as handle:
        if layout == "exchange":
            data_path, key_path, angles_path = "/exchange/data", None, "/exchange/theta"
            frames, angles = [sample, sample * 2], [0, np.pi]
            handle.create_dataset("exchange/data_white", data=[sample * 10])
            handle.create_dataset("exchange/data_dark", data=[sample * 0])
        else:
            data_path, key_path, angles_path = (
                (
                    "/entry/instrument/detector/data",
                    "/entry/instrument/detector/image_key",
                    "/entry/sample/rotation_angle",
                )
                if layout == "nxtomo"
                else (
                    "/entry/imaging/data",
                    "/entry/instrument/EtherCAT/image_key",
                    "/entry/imaging/smaract_zrot_value_set",
                )
            )
            frames, angles = [sample, sample * 2, sample * 10, sample * 0], [0, 180, 0, 0]
            handle.create_dataset(key_path, data=[0, 0, 1, 2])
        handle.create_dataset(data_path, data=frames)
        handle.create_dataset(angles_path, data=angles).attrs["units"] = (
            "radians" if layout == "exchange" else "mm" if layout == "gda" else "degree"
        )
    previews = tmp_path / "previews"
    assert main(["inspect", str(path), "--json", "--preview", str(previews)]) == (
        layout == "exchange"
    )
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] == (layout != "exchange")
    assert len(report["issues"]) == (layout == "exchange")
    assert report["projection"]["path"] == data_path
    assert report["angles"]["path"] == angles_path
    assert report["angles"]["coverage_deg"] == pytest.approx(180)
    assert report["flats_darks"]["sample_count"] == 2
    assert report["flats_darks"]["flat_count"] == report["flats_darks"]["dark_count"] == 1
    assert report["flats_darks"]["image_key_path"] == key_path
    np.testing.assert_array_equal(
        iio.imread(previews / "projection.png"), scale_to_uint8(sample * 2)
    )


@pytest.mark.parametrize("roi", ["auto", "off", "cyl"])
def test_cli_grid_resamples_the_cropped_scan_field_of_view(tmp_path, roi):
    import tomojax as tj
    from tomojax.geometry import grid_volume_origin

    grid, detector = tj.Grid(12, 12, 10, 1, 1, 1), tj.Detector(12, 10, 1, 1)
    scan = tj.Scan(
        np.ones((20, 10, 12), np.float32),
        tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 20, endpoint=False)),
    ).cropped(slice(6, 8), slice(None))
    src, out = tmp_path / "scan.nxs", tmp_path / "recon.nxs"
    tj.save(src, scan)
    assert main(["recon", str(src), "-o", str(out), "--grid", "6", "6", "2", "--roi", roi]) == 0
    result = tj.load_reconstruction(out)
    assert (result.grid.vx, result.grid.vy, result.grid.vz) == (2, 2, 5 if roi == "off" else 1)
    assert grid_volume_origin(result.grid)[2] == (-2.5 if roi == "off" else 1.5)
    if roi == "cyl":
        assert not np.asarray(result.volume)[0, 0, :].any()
