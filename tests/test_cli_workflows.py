from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from tomojax.cli.main import main
from tomojax.io import load_dataset, save_dataset

from ._helpers import (
    make_projection_dataset,
    write_projection_dataset,
    write_raw_nxtomo,
)

pytestmark = pytest.mark.surface


def test_recon_cli_defaults_defer_policy_and_report_limitations(monkeypatch, tmp_path, caplog):
    import tomojax as tj
    from tomojax._workflow import operations as _workflow

    path, output, manifest = tmp_path / "scan.nxs", tmp_path / "result.nxs", tmp_path / "run.json"
    write_projection_dataset(path)
    automatic = {"status": "limited", "reasons": ["Rotation axis metadata is missing."]}

    def automatic_result(plan, *, out=None, checkpoint=None):
        assert plan.info["automatic"]["method"] == plan.method  # the policy chose it
        scan = plan.scan
        volume = np.zeros((scan.grid.nz, scan.grid.ny, scan.grid.nx), np.float32)
        return tj.Reconstruction(volume, scan.geometry, "fbp", {"automatic": automatic}, scan)

    monkeypatch.setattr(_workflow, "reconstruct_resolved", automatic_result)
    assert main(["reconstruct", str(path), "-o", str(output), "--manifest", str(manifest)]) == 0
    assert load_dataset(output).volume is not None
    assert "Rotation axis metadata is missing" in caplog.text
    record = json.loads(manifest.read_text())["resolved_config"]
    assert record["reconstruction"]["automatic"] == automatic


def test_recon_cli_requires_a_method_for_expert_settings(tmp_path, capsys):
    path, output, config = tmp_path / "scan.nxs", tmp_path / "result.nxs", tmp_path / "expert.toml"
    write_projection_dataset(path)
    config.write_text("views_per_batch = 2\n")
    with pytest.raises(SystemExit) as error:
        main(["reconstruct", str(path), "-o", str(output), "--config", str(config)])
    assert error.value.code == 2
    assert "explicit --method" in capsys.readouterr().err


def test_preprocess_rejects_unverified_vendor_hardening(capsys):
    with pytest.raises(SystemExit) as error:
        main(["correct", "raw.nxs", "-o", "scan.nxs", "--beam-hardening", "recorded"])
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
    assert payload["angles"]["coverage"] == 90.0
    assert payload["volume"]["found"] is False


def test_inspect_raw_acquisition_reports_required_cli_inputs(tmp_path, capsys):
    path = tmp_path / "raw.h5"
    with h5py.File(path, "w") as file:
        file["exchange/data"] = np.ones((2, 2, 4), np.uint16)
        file["exchange/theta"] = [0, 90]
    assert main(["inspect", str(path), "--json"]) == 1
    issues = json.loads(capsys.readouterr().out)["issues"]
    assert len(issues) == 2  # no pixel size is not an issue: it loads in pixel units
    assert "--flats" in issues[0] and "--darks" in issues[1]
    with h5py.File(path, "a") as file:
        file["exchange/data_white"] = np.ones((1, 2, 4))
        file["exchange/data_dark"] = np.zeros((1, 2, 4))
    assert main(["inspect", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["issues"] == []


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
    assert report["angles"]["min"] == pytest.approx(30.0, abs=1e-5)
    assert report["angles"]["max"] == pytest.approx(60.0, abs=1e-5)


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
    dataset.volume = np.arange(4 * 4 * 2, dtype=np.float32).reshape(2, 4, 4)
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
            "--preview-dir",
            str(previews),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    names = sorted(p.name for p in previews.iterdir())
    assert names == ["projection.png", "slice_x_2.png", "slice_y_2.png", "slice_z_1.png"]
    with pytest.raises(SystemExit) as exc_info:
        main(["inspect", str(path), "--preview-dir", str(previews)])
    assert exc_info.value.code == 2
    assert main(["inspect", str(path), "--preview-dir", str(previews), "--force"]) == 0


def test_load_command_is_retired(capsys):
    with pytest.raises(SystemExit) as error:
        main(["load", "raw.nxs", "-o", "scan.nxs"])
    assert error.value.code == 2
    assert "unknown command 'load'" in capsys.readouterr().err


def test_correct_declares_a_geometry_the_file_does_not_record(tmp_path, capsys):
    # tomojax correct supplies what the file lacks, the geometry included,
    # and resolves it inside the loader before file geometry is required.
    import tomojax as tj

    write_raw_nxtomo(tmp_path / "raw.nxs")
    output = tmp_path / "scan.nxs"
    args = ["correct", str(tmp_path / "raw.nxs"), "-o", str(output), "--pixel-size", "0.5"]
    with pytest.raises(SystemExit) as exc:
        main([*args, "--geometry", "lamino"])
    assert exc.value.code == 2
    assert "needs --tilt" in capsys.readouterr().err and not output.exists()
    with pytest.raises(SystemExit) as exc:
        main([*args, "--tilt", "47"])
    assert exc.value.code == 2
    assert "requires --geometry" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(
            ["correct", str(tmp_path / "raw.nxs"), "-o", str(output), "--pixel-size", "1", "2", "3"]
        )
    assert exc.value.code == 2
    assert "one value, or two" in capsys.readouterr().err
    lamino = ["--geometry", "lamino", "--tilt", "47", "--detector-roll", "1.25"]
    assert main([*args, *lamino, "--detector-center", "0.1", "0.0"]) == 0
    loaded = tj.load(output)
    geometry = loaded.geometry
    assert isinstance(geometry, tj.LaminographyGeometry) and geometry.tilt == 47
    assert geometry.detector_roll == 1.25 and geometry.detector.center == (0.1, 0.0)
    assert geometry.detector.du == 0.5
    assert loaded.source is not None
    assert loaded.source.geometry_metadata.get("detector_center_status") == "provided"


def test_import_cli_converts_nxtomo_to_npz(tmp_path: Path) -> None:
    nxs_path = tmp_path / "scan.nxs"
    npz_path = tmp_path / "scan.npz"
    write_projection_dataset(nxs_path)

    assert main(["correct", str(nxs_path), "-o", str(npz_path)]) == 0

    dataset = load_dataset(npz_path)
    assert dataset.projections.shape == (2, 2, 4)
    np.testing.assert_allclose(dataset.angles, [0.0, 90.0])


def _fake_reconstruct(
    monkeypatch: pytest.MonkeyPatch, error: Exception | None = None
) -> dict[str, object]:
    """Record what the CLI asks the resolver for, and solve nothing (or raise ``error``)."""
    import tomojax
    from tomojax._workflow import operations as _workflow, planning as _planning

    calls: dict[str, object] = {}
    resolve = _planning.resolve_reconstruction

    def recorded(scan, method="auto", **keywords):
        calls.update(method=method, **keywords)
        return resolve(scan, method, **keywords)

    def fake(plan, *, out=None, checkpoint=None):
        if error is not None:
            raise error
        volume = np.zeros(plan.scan.grid.shape, np.float32)
        info = {"config": plan.config, **plan.info}
        return tomojax.Reconstruction(volume, plan.scan.geometry, plan.method, info, plan.scan)

    monkeypatch.setattr(_planning, "resolve_reconstruction", recorded)
    monkeypatch.setattr(_workflow, "reconstruct_resolved", fake)
    return calls


def test_recon_cli_routes_tiny_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    write_projection_dataset(scan)
    calls = _fake_reconstruct(monkeypatch)

    args = ["reconstruct", str(scan), "-o", str(recon), "--method", "cgls"]
    assert main([*args, "--grid", "4", "4", "2", "--iterations", "7", "--warm-start"]) == 0

    assert calls.pop("method") == "cgls"
    assert (calls.pop("shape"), calls.pop("config").iterations) == ((2, 4, 4), None)
    # Options not given are left to the configuration.
    assert calls == {
        "region": None,
        "filter": None,
        "iterations": 7,
        "tv_weight": None,
        "nonnegative": None,
        "warm_start": True,
        "seed": None,
    }
    loaded = load_dataset(recon)
    assert loaded.volume is not None
    assert loaded.volume.shape == (2, 4, 4)


def test_main_formats_expected_subcommand_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    write_projection_dataset(scan)
    _ = _fake_reconstruct(monkeypatch, ValueError("bad detector metadata"))

    assert main(["reconstruct", str(scan), "-o", str(recon)]) == 1
    captured = capsys.readouterr()
    assert captured.err == "tomojax reconstruct: error: bad detector metadata\n"


def test_main_does_not_swallow_programmer_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    write_projection_dataset(scan)
    _ = _fake_reconstruct(monkeypatch, TypeError("wrong internal call shape"))

    with pytest.raises(TypeError, match="wrong internal call shape"):
        main(["reconstruct", str(scan), "-o", str(recon)])


def test_recon_cli_executes_fbp_and_writes_volume_metadata(tmp_path: Path) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    manifest = tmp_path / "recon-manifest.json"
    config = tmp_path / "recon.toml"
    write_projection_dataset(scan)
    _ = config.write_text('scale = 0.5\nexterior = "zero"\n', encoding="utf-8")

    args = [
        "reconstruct",
        str(scan),
        "-o",
        str(recon),
        "--method",
        "fbp",
        "--grid",
        "4",
        "4",
        "2",
    ]
    assert main([*args, "--config", str(config), "--manifest", str(manifest)]) == 0

    loaded = load_dataset(recon)
    assert loaded.volume is not None
    assert loaded.volume.shape == (2, 4, 4)
    assert np.isfinite(loaded.volume).all()
    assert loaded.grid is not None
    assert (loaded.grid.nx, loaded.grid.ny, loaded.grid.nz) == (4, 4, 2)
    assert loaded.detector is not None
    with h5py.File(recon) as handle:
        assert handle["entry/processing/tomojax"].attrs["volume_axes_order"] == "zyx"
    assert loaded.geometry_metadata["reconstruction_method"] == "fbp"

    resolved = json.loads(manifest.read_text(encoding="utf-8"))["resolved_config"]
    assert resolved["method"] == "fbp"
    assert resolved["settings"] == {"scale": 0.5, "exterior": "zero"}
    assert loaded.geometry_metadata["reconstruction_info"]["config"]["exterior"] == "zero"
    settings = resolved["reconstruction"]["config"]
    assert (settings["filter"], settings["scale"]) == ("ramp", 0.5)
    assert resolved["reconstruction_grid"]["nx"] == 4
    assert resolved["region"] == {
        "requested": None,
        "grid_changed": False,
        "cylindrical_output_mask": False,
    }
    assert resolved["volume_shape"] == [2, 4, 4]


@pytest.mark.parametrize("warm_start", [False, True])
def test_recon_cli_runs_cgls_like_the_python_solver(tmp_path: Path, *, warm_start: bool) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    manifest = tmp_path / "recon-manifest.json"
    write_projection_dataset(scan)
    args = [
        "--grid",
        "4",
        "4",
        "2",
        "--iterations",
        "6",
        *(["--warm-start"] if warm_start else []),
    ]
    command = ["reconstruct", str(scan), "-o", str(recon), "--method", "cgls"]
    assert main([*command, *args, "--manifest", str(manifest)]) == 0
    loaded = load_dataset(recon)
    assert loaded.volume is not None
    assert loaded.volume.shape == (2, 4, 4)
    resolved = json.loads(manifest.read_text(encoding="utf-8"))["resolved_config"]
    assert resolved["method"] == "cgls"
    assert resolved["reconstruction"]["warm_start"] == warm_start
    assert resolved["reconstruction"]["config"]["iterations"] == 6
    assert 1 <= resolved["reconstruction"]["effective_iterations"] <= 6
    # The saved image is the Python solver's on the same scan, grid and start.
    import tomojax as tj

    source = tj.load(scan)
    grid = source.grid.resampled(shape=(2, 4, 4))
    expected = tj.reconstruct(source, "cgls", grid=grid, iterations=6, warm_start=warm_start)
    np.testing.assert_allclose(loaded.volume, np.asarray(expected.volume), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_recon_cli_auto_roi_does_not_depend_on_physical_units(tmp_path: Path, scale: float) -> None:
    import tomojax as tj

    grid = tj.Grid(11, 11, 5, scale, scale, scale)
    detector = tj.Detector(9, 3, scale, scale)
    geometry = tj.ParallelGeometry(grid, detector, np.array([0.0, 90.0]))
    scan_path, recon_path = tmp_path / "scan.nxs", tmp_path / "recon.nxs"
    tj.save(scan_path, tj.Scan(np.zeros((2, 3, 9), np.float32), geometry))

    args = [
        "reconstruct",
        str(scan_path),
        "-o",
        str(recon_path),
        "--method",
        "cgls",
        "--region",
        "auto",
    ]
    assert main([*args, "--iterations", "0"]) == 0

    actual = tj.load_reconstruction(recon_path)
    assert (actual.grid.nx, actual.grid.ny, actual.grid.nz) == (9, 9, 3)
    assert (actual.grid.vx, actual.grid.vy, actual.grid.vz) == (scale, scale, scale)
    np.testing.assert_array_equal(actual.volume, np.zeros((3, 9, 9), np.float32))


def test_recon_cli_cylinder_masks_fista_after_solving(tmp_path: Path) -> None:
    import numpy as np

    import tomojax as tj

    scan_path, recon_path = tmp_path / "scan.nxs", tmp_path / "recon.nxs"
    write_projection_dataset(scan_path)
    command = ["reconstruct", str(scan_path), "-o", str(recon_path), "--method", "fista"]
    assert main([*command, "--region", "cyl", "--iterations", "4"]) == 0

    scan = tj.load(scan_path)
    grid = tj.load_reconstruction(recon_path).grid
    # check-public-imports: allow-private
    from tomojax._workflow.field_of_view import cylinder_support, region_grid

    # --region cyl solves unconstrained on the cylinder grid and masks outside afterwards.
    _, cylinder = region_grid(scan, "cyl")
    assert cylinder is not None
    mask = cylinder_support(grid, cylinder)
    expected = tj.reconstruct(scan, "fista", grid=grid, iterations=4).volume
    actual = tj.load_reconstruction(recon_path).volume
    np.testing.assert_allclose(actual, np.where(mask, expected, 0), rtol=1e-5, atol=1e-6)
    # Python region="cyl" matches CLI --region cyl identically.
    py_cyl = tj.reconstruct(scan, "fista", region="cyl", iterations=4).volume
    np.testing.assert_allclose(actual, np.asarray(py_cyl), rtol=1e-5, atol=1e-6)


def test_recon_cli_config_settings_are_fields_of_the_methods_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scan = tmp_path / "scan.nxs"
    config = tmp_path / "recon.toml"
    write_projection_dataset(scan)
    calls = _fake_reconstruct(monkeypatch)
    _ = config.write_text(
        'regularizer = "huber_tv"\nhuber_delta = 0.05\ntv_weight = 0.01\n', encoding="utf-8"
    )

    args = ["reconstruct", str(scan), "-o", str(tmp_path / "out.nxs"), "--method", "fista"]
    assert main([*args, "--config", str(config)]) == 0

    cfg = calls["config"]
    assert (cfg.regularizer, cfg.huber_delta, cfg.iterations) == ("huber_tv", 0.05, 200)
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

    args = ["reconstruct", str(scan), "-o", str(tmp_path / "out.nxs"), "--config", str(config)]
    with pytest.raises(SystemExit) as exc:
        main([*args, *options])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert message in err
    if toml:
        assert "has " in err and "backend" in err  # the valid ones


def test_inspect_cli_previews_central_slices_of_xyz_disk_volumes(tmp_path: Path) -> None:
    path = tmp_path / "recon.nxs"
    previews = tmp_path / "previews"
    volume = np.zeros((3, 4, 5), dtype=np.float32)
    for x in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for z in range(volume.shape[2]):
                volume[x, y, z] = (100 * x) + (10 * y) + z
    dataset = make_projection_dataset()
    dataset.volume = volume.transpose(2, 1, 0)
    save_dataset(path, dataset)
    with h5py.File(path, "r+") as file:
        group = file["entry/processing/tomojax"]
        del group["volume"]
        group.create_dataset("volume", data=volume)
        group.attrs["volume_axes_order"] = "xyz"

    from tomojax.cli import _previews  # check-public-imports: allow-private

    assert main(["inspect", str(path), "--preview-dir", str(previews)]) == 0
    # Central slices, displayed as (y, x), (z, x) and (z, y).
    for axis, index, plane in (
        ("z", 2, volume[:, :, 2].T),
        ("y", 2, volume[:, 2, :].T),
        ("x", 1, volume[1, :, :].T),
    ):
        assert (previews / f"slice_{axis}_{index}.png").exists()
        np.testing.assert_array_equal(_previews.saved_plane(path, axis, index), plane)


def test_cli_reports_a_missing_input_as_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = str(tmp_path / "missing.nxs")
    with pytest.raises(SystemExit) as exc:
        main(["inspect", missing])
    assert exc.value.code == 2
    assert "input not found" in capsys.readouterr().err
    assert main(["reconstruct", missing, "-o", str(tmp_path / "r.nxs"), "--progress", "json"]) == 1
    error = json.loads(capsys.readouterr().out)
    assert error["event"] == "error"
    assert error["kind"] == "input"
    assert missing in error["message"]


def test_simulate_cli_routes_loadable_synthetic_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_path = tmp_path / "synthetic.nxs"
    captured: dict[str, object] = {}

    import tomojax as tj
    from tomojax import datasets

    simulate = datasets.simulate

    def recording(config):
        captured["shape"] = (config.nx, config.ny, config.nz, config.nu, config.nv, config.n_views)
        captured["expert"] = (config.single_rotate, config.artifacts.zinger_fraction)
        return simulate(config)

    monkeypatch.setattr(datasets, "simulate", recording)
    # Phantom details and detector artifacts are config-file settings.
    settings = tmp_path / "simulate.toml"
    _ = settings.write_text("single_rotate = false\nzinger_fraction = 0.25\n", encoding="utf-8")

    assert (
        main(
            [
                "simulate",
                "--config",
                str(settings),
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
                "--translation-sigma",
                "0.25",
            ]
        )
        == 0
    )

    assert captured["shape"] == (2, 2, 2, 2, 2, 8)
    assert captured["expert"] == (False, 0.25)
    loaded = load_dataset(out_path)
    assert loaded.projections.shape == (8, 2, 2)
    assert loaded.volume is not None
    sim = tj.load_simulation(out_path)
    assert sim.scan.poses is None
    assert sim.true_poses is not None and np.any(sim.true_poses)


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
    assert main(["inspect", str(path), "--json", "--preview-dir", str(previews)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] and report["issues"] == []
    assert report["projection"]["path"] == data_path
    assert report["angles"]["path"] == angles_path
    assert report["angles"]["coverage"] == pytest.approx(180)
    assert report["flats_darks"]["sample_count"] == 2
    assert report["flats_darks"]["flat_count"] == report["flats_darks"]["dark_count"] == 1
    assert report["flats_darks"]["image_key_path"] == key_path
    from tomojax.cli import _previews  # check-public-imports: allow-private

    assert (previews / "projection.png").exists()
    np.testing.assert_array_equal(_previews.projection_image(path), sample * 2)


@pytest.mark.parametrize("roi", [None, "auto", "cyl"])
def test_cli_grid_resamples_the_cropped_scan_field_of_view(tmp_path, roi):
    import tomojax as tj
    from tomojax.geometry import grid_volume_origin

    # The field (radius 6.5) reaches a voxel center in each of the 12 columns.
    grid, detector = tj.Grid(12, 12, 10, 1, 1, 1), tj.Detector(14, 10, 1, 1)
    scan = tj.Scan(
        np.ones((20, 10, 14), np.float32),
        tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 20, endpoint=False)),
    ).cropped(slice(6, 8), slice(None))
    src, out = tmp_path / "scan.nxs", tmp_path / "recon.nxs"
    tj.save(src, scan)
    region_flags = [] if roi is None else ["--region", roi]
    assert (
        main(["reconstruct", str(src), "-o", str(out), "--grid", "6", "6", "2", *region_flags]) == 0
    )
    result = tj.load_reconstruction(out)
    assert (result.grid.vx, result.grid.vy, result.grid.vz) == (2, 2, 5 if roi is None else 1)
    assert grid_volume_origin(result.grid)[2] == (-2.5 if roi is None else 1.5)
    if roi == "cyl":
        assert not np.asarray(result.volume)[:, 0, 0].any()


@pytest.mark.parametrize("auxiliary", ["--preview", "--manifest"])
def test_recon_refuses_to_replace_an_existing_auxiliary_output(tmp_path: Path, auxiliary):
    scan = tmp_path / "scan.nxs"
    write_projection_dataset(scan)
    existing = tmp_path / ("preview.png" if auxiliary == "--preview" else "run.json")
    existing.write_text("keep")
    command = [
        "reconstruct",
        str(scan),
        "-o",
        str(tmp_path / "recon.nxs"),
        auxiliary,
        str(existing),
    ]
    with pytest.raises(SystemExit) as exit_info:
        main(command)
    assert exit_info.value.code == 2 and existing.read_text() == "keep"


def test_raw_export_refuses_to_replace_an_existing_sidecar(tmp_path: Path):
    from tomojax.geometry import Detector, Grid
    from tomojax.io import ProjectionDataset

    dataset = ProjectionDataset(
        projections=np.zeros((2, 4, 6), np.float32),
        angles=np.asarray([0.0, 90.0], np.float32),
        volume=np.ones((4, 5, 6), np.float32),
        detector=Detector(6, 4, 1.0, 1.0),
        grid=Grid(6, 5, 4, 0.5, 0.5, 0.25),
    )
    save_dataset(tmp_path / "recon.nxs", dataset)
    (tmp_path / "volume.json").write_text("keep")
    with pytest.raises(SystemExit) as exit_info:
        main(["export", str(tmp_path / "recon.nxs"), "-o", str(tmp_path / "volume.raw")])
    assert exit_info.value.code == 2 and (tmp_path / "volume.json").read_text() == "keep"


def test_correct_cli_supports_config_slices_and_open_crop(tmp_path: Path, capsys) -> None:
    import tomojax as tj

    raw = tmp_path / "raw.nxs"
    write_raw_nxtomo(raw)
    cfg = tmp_path / "correct.toml"
    cfg.write_text("zingers_size = 3\n", encoding="utf-8")
    out = tmp_path / "scan.nxs"

    with pytest.raises(SystemExit) as info:
        main(["correct", "--config-keys"])
    assert info.value.code == 0
    keys_out = capsys.readouterr().out
    assert "zingers_size" in keys_out and "paganin_delta_beta" in keys_out

    assert (
        main(
            [
                "correct",
                str(raw),
                "-o",
                str(out),
                "--config",
                str(cfg),
                "--views",
                "::2",
                "--crop",
                "0:2,:1",
                "--zingers",
            ]
        )
        == 0
    )
    loaded = tj.load(out)
    assert loaded.projections.shape == (1, 2, 1)

    for bad_flag in (["--views", "bad"], ["--crop", "bad"], ["--paganin", "100,,10,1"]):
        with pytest.raises(SystemExit) as exc:
            main(["correct", str(raw), "-o", str(tmp_path / "bad.nxs"), *bad_flag])
        assert exc.value.code == 2


def test_correct_cli_no_apply_poses_preserves_nominal_geometry(tmp_path: Path) -> None:
    import tomojax as tj

    src = tmp_path / "posed.nxs"
    out = tmp_path / "nominal.nxs"
    write_raw_nxtomo(src)
    with h5py.File(src, "a") as handle:
        geom = handle.require_group("entry/geometry")
        geom.attrs["align_params_json"] = json.dumps(
            [[1.0, 2.0, 0.0, 0.0, 0.0], [0.5, -1.0, 0.0, 0.0, 0.0]]
        )
    assert main(["correct", str(src), "-o", str(out), "--no-apply-poses"]) == 0
    loaded = tj.load(out)
    assert loaded.poses is None


def test_cli_rejects_duplicate_and_self_overwrite_outputs_even_with_force(
    tmp_path: Path, capsys
) -> None:
    scan = tmp_path / "scan.nxs"
    write_projection_dataset(scan)
    with pytest.raises(SystemExit) as exc:
        main(["reconstruct", str(scan), "-o", str(scan), "--force"])
    assert exc.value.code == 2
    assert "output path must differ from input path" in capsys.readouterr().err

    same = tmp_path / "out.nxs"
    with pytest.raises(SystemExit) as exc:
        main(["reconstruct", str(scan), "-o", str(same), "--manifest", str(same), "--force"])
    assert exc.value.code == 2
    assert "duplicate output" in capsys.readouterr().err


def test_cli_config_validation_rejects_bad_types_and_forbidden_keys(tmp_path: Path, capsys) -> None:
    scan = tmp_path / "scan.nxs"
    write_projection_dataset(scan)

    for toml_text, args, fragment in (
        (
            "force = true\n",
            ["reconstruct", str(scan), "-o", str(tmp_path / "a.nxs")],
            "unknown config key",
        ),
        (
            'out = "x.nxs"\n',
            ["reconstruct", str(scan), "-o", str(tmp_path / "b.nxs")],
            "unknown config key",
        ),
        (
            "iterations = 8.9\n",
            ["reconstruct", str(scan), "-o", str(tmp_path / "c.nxs"), "--method", "cgls"],
            "integer",
        ),
        (
            "levels = 2\n",
            ["align", str(scan), "-o", str(tmp_path / "d.nxs")],
            "levels",
        ),
    ):
        cfg = tmp_path / "bad.toml"
        cfg.write_text(toml_text, encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            main([*args, "--config", str(cfg)])
        assert exc.value.code == 2
        assert fragment in capsys.readouterr().err


def test_reconstruct_help_defaults_config_keys_and_boolean_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    import tomllib

    from tomojax.recon import CGLSConfig, FISTAConfig, MLEMConfig, SIRTConfig, SPDHGConfig

    with pytest.raises(SystemExit) as help_exit:
        main(["reconstruct", "--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "cgls/sirt up to 50" in help_text
    assert f"mlem {MLEMConfig().iterations}" in help_text
    assert CGLSConfig().iterations is None
    assert SIRTConfig().iterations is None
    assert f"fista up to {FISTAConfig().iterations}" in " ".join(help_text.split())
    assert SPDHGConfig().iterations is None
    assert "spdhg one block per update" in " ".join(help_text.split())

    with pytest.raises(SystemExit) as keys_exit:
        main(["reconstruct", "--method", "fista", "--config-keys"])
    assert keys_exit.value.code == 0
    keys_text = capsys.readouterr().out
    parsed = tomllib.loads(keys_text)
    assert "regularizer" in parsed and "tau" not in parsed

    scan = tmp_path / "scan.nxs"
    write_projection_dataset(scan)
    cfg = tmp_path / "recon.toml"
    cfg.write_text("nonnegative = true\nwarm_start = true\n", encoding="utf-8")
    calls = _fake_reconstruct(monkeypatch)
    assert (
        main(
            [
                "reconstruct",
                str(scan),
                "-o",
                str(tmp_path / "r.nxs"),
                "--method",
                "fista",
                "--config",
                str(cfg),
                "--no-nonnegative",
                "--no-warm-start",
            ]
        )
        == 0
    )
    assert calls["nonnegative"] is False and calls["warm_start"] is False

    for bad_auto in (["--iterations", "5"], ["--no-warm-start"]):
        with pytest.raises(SystemExit) as auto_exc:
            main(["reconstruct", str(scan), "-o", str(tmp_path / "a.nxs"), *bad_auto])
        assert auto_exc.value.code == 2
        assert "explicit --method" in capsys.readouterr().err


def test_simulate_cli_grid_derives_view_count_like_python(tmp_path: Path) -> None:
    import tomojax as tj

    out_path = tmp_path / "asym.nxs"
    assert (
        main(["simulate", "-o", str(out_path), "--grid", "6", "5", "4", "--phantom", "sphere"]) == 0
    )
    loaded = tj.load(out_path)
    py_scan, _ = tj.simulate("sphere", grid=tj.Grid(6, 5, 4, 1, 1, 1))
    assert len(loaded.angles) == len(py_scan.angles) == 12
    assert loaded.grid.shape == (4, 5, 6)


def test_align_cli_opposite_views_validation_dry_run_and_manifest(tmp_path: Path, capsys) -> None:
    import tomojax as tj

    scan = tmp_path / "scan.nxs"
    grid = tj.Grid(8, 8, 4, 1.0, 1.0, 1.0)
    detector = tj.Detector(10, 4, 1.0, 1.0)
    angles = np.array([0.0, 90.0, 180.0, 270.0], np.float32)
    geom = tj.ParallelGeometry(grid, detector, angles)
    x = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    proj = np.zeros((4, 4, 10), np.float32)
    proj[0] = np.exp(-((x - 0.2) ** 2) / 0.15)
    proj[1] = np.exp(-((x + 0.1) ** 2) / 0.2)
    proj[2] = np.exp(-((-x - 0.2) ** 2) / 0.15)
    proj[3] = np.exp(-((-x + 0.1) ** 2) / 0.2)
    tj.save(scan, tj.Scan(proj, geom))

    def document(*options):
        assert main(["align", str(scan), "--dry-run", "--progress", "json", *options]) == 0
        result = json.loads(capsys.readouterr().out.splitlines()[-1])
        assert result["event"] == "result" and result["outputs"] == []
        return result["info"]["summary"]

    plan = document()
    assert plan["output_path"] is None and plan["method"] == "joint"
    assert plan["fit"] == ["translations"]
    center = document("--method", "opposite-views")
    assert center["type"] == "opposite_views_plan" and center["fit"] == ["center"]
    assert plan["bounded_working_grid"] is True
    assert plan["type"] == "joint_plan" and plan["config"]["levels"]
    assert document("--region", "auto")["bounded_working_grid"] is False
    assert main(["align", str(scan), "--dry-run"]) == 0  # people get text
    assert capsys.readouterr().out.startswith("method")

    for extra in (
        ["--fit", "translations"],
        ["--fit", "center", "--grid", "4", "4", "2"],
        ["--fit", "center", "--iterations", "2"],
        ["--fit", "center", "--region", "auto"],
    ):
        with pytest.raises(SystemExit) as exc:
            main(["align", str(scan), "--dry-run", "--method", "opposite-views", *extra])
        assert exc.value.code == 2

    out = tmp_path / "aligned.nxs"
    manifest = tmp_path / "align.json"
    assert (
        main(
            [
                "align",
                str(scan),
                "-o",
                str(out),
                "--method",
                "opposite-views",
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    assert "Shared corrections  center" in capsys.readouterr().out
    resolved = json.loads(manifest.read_text(encoding="utf-8"))["resolved_config"]
    assert resolved["method"] == "opposite-views" and resolved["fit"] == ["center"]
    assert "explicit_cli_keys" in resolved and "effective_options" in resolved


def test_align_cli_prints_fitted_motion_and_stop_reason(tmp_path, capsys):
    import tomojax as tj

    path, out = tmp_path / "motion.nxs", tmp_path / "aligned.nxs"
    sim = tj.simulate("shepp", size=12, n_views=20, translation_sigma=0.3)
    tj.save(path, sim)
    assert (
        main(
            [
                "align",
                str(path),
                "-o",
                str(out),
                "--fit",
                "translations",
                "rotations",
                "--iterations",
                "1",
                "--progress",
                "none",
            ]
        )
        == 0
    )
    summary, warning = capsys.readouterr()
    assert "weakly determined" in warning
    assert "relying on them. Review" in warning
    assert "https://tomojax.dev/align-a-scan/#check-the-result" in warning
    report = tj.load_alignment(out).info["report"]
    dx = report["poses"]["x"]
    assert "Per-view shifts (RMS/max, native px)" in summary
    assert f"x {dx['rms']:.3g}/{dx['max']:.3g}" in summary
    assert "Per-view rotations (RMS/max, deg)" in summary
    assert "Stopped  level 1" in summary and "step limit reached (1 step)" in summary


def test_export_cli_validates_flags_cleans_stale_tiffs_and_fails_without_volume(
    tmp_path: Path, capsys
) -> None:
    from tomojax.geometry import Detector, Grid
    from tomojax.io import ProjectionDataset

    no_vol = tmp_path / "scan.nxs"
    write_projection_dataset(no_vol)
    assert main(["export", str(no_vol), "-o", str(tmp_path / "out.raw")]) == 1
    assert "no reconstructed volume" in capsys.readouterr().err

    recon = tmp_path / "recon.nxs"
    dataset = ProjectionDataset(
        projections=np.zeros((2, 4, 6), np.float32),
        angles=np.asarray([0.0, 90.0], np.float32),
        volume=np.ones((2, 4, 6), np.float32),
        detector=Detector(6, 4, 1.0, 1.0),
        grid=Grid(6, 4, 2, 0.5, 0.5, 0.25),
    )
    save_dataset(recon, dataset)

    for bad_args in (
        ["--dtype", "float32", "--range", "0", "1", "-o", str(tmp_path / "a.raw")],
        ["--dtype", "uint8", "--range", "1", "1", "-o", str(tmp_path / "b.raw")],
        ["--prefix", "img", "-o", str(tmp_path / "c.raw")],
    ):
        with pytest.raises(SystemExit) as exc:
            main(["export", str(recon), *bad_args])
        assert exc.value.code == 2

    tiffs = tmp_path / "slices"
    tiffs.mkdir()
    (tiffs / "slice_00099.tif").write_bytes(b"stale")
    assert main(["export", str(recon), "-o", str(tiffs), "--force"]) == 0
    assert not (tiffs / "slice_00099.tif").exists()
    assert sorted(p.name for p in tiffs.iterdir()) == [
        "slice.json",
        "slice_00000.tif",
        "slice_00001.tif",
    ]
    sidecar = json.loads((tiffs / "slice.json").read_text(encoding="utf-8"))
    assert "vol_origin_xyz" in sidecar and "grid" in sidecar


def test_inspect_json_prints_even_if_preview_fails(tmp_path: Path, capsys) -> None:
    path = tmp_path / "scan.nxs"
    write_projection_dataset(path)
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("file", encoding="utf-8")

    assert main(["inspect", str(path), "--json", "--preview-dir", str(blocker)]) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["valid"] is True
    assert payload["preview_error"]
    assert "preview failed" in captured.err
