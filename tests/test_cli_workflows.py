from __future__ import annotations

import json
from pathlib import Path

import h5py
import imageio.v3 as iio
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.cli.main import main
import tomojax.cli.recon as recon_cli
import tomojax.cli.simulate as simulate_cli
from tomojax.geometry import Detector, stack_view_poses
from tomojax.geometry.api import CalibrationState, CalibrationVariable
from tomojax.io import build_geometry_from_dataset_metadata, load_dataset, save_dataset
from tomojax.io.api import save_projection_payload
from tomojax.recon.quicklook import scale_to_uint8

from ._helpers import (
    make_projection_dataset,
    write_angle_csv,
    write_projection_dataset,
    write_tiff_stack,
)

pytestmark = pytest.mark.surface


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


def test_inspect_cli_exits_1_for_an_invalid_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "scan.nxs"
    write_projection_dataset(path)
    with h5py.File(path, "a") as handle:
        del handle["/entry/sample/name"]

    assert main(["inspect", str(path)]) == 1
    assert "Issues (" in capsys.readouterr().out


def test_inspect_cli_previews_a_reconstruction(tmp_path: Path) -> None:
    path = tmp_path / "recon.nxs"
    dataset = make_projection_dataset()
    dataset.volume = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
    save_dataset(path, dataset)
    previews = tmp_path / "previews"

    assert main(["inspect", str(path), "--preview", str(previews)]) == 0
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

    args = ["import", str(stack), "--angles", str(angles), "-o", str(out_path)]
    assert main([*args, "--pixel-size", "0.5", "0.75", "--name", "rock"]) == 0

    dataset = load_dataset(out_path)
    assert dataset.projections.shape == (2, 2, 4)
    assert dataset.detector is not None
    assert dataset.detector.du == pytest.approx(0.5)
    assert dataset.detector.dv == pytest.approx(0.75)
    assert dataset.sample_name == "rock"

    with pytest.raises(SystemExit) as exc_info:
        main(args)
    assert exc_info.value.code == 2
    assert main([*args, "--force"]) == 0
    assert load_dataset(out_path).detector.du == pytest.approx(1.0)  # pyright: ignore[reportOptionalMemberAccess]


def test_import_cli_needs_angles_for_tiffs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stack = tmp_path / "stack"
    write_tiff_stack(stack, [1.0, 2.0], shape=(2, 4))
    with pytest.raises(SystemExit) as exc_info:
        main(["import", str(stack), "-o", str(tmp_path / "scan.nxs")])
    assert exc_info.value.code == 2
    assert "TIFF stacks need --angles" in capsys.readouterr().err


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
                "-o",
                str(out_path),
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
        ("--flats", "needs --flats, --darks and --angles"),
        ("--darks", "needs --flats, --darks and --angles"),
        ("--angles", "needs --flats, --darks and --angles"),
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
        "-o",
        str(out_path),
        "--flats",
        str(flats),
        "--darks",
        str(darks),
        "--angles",
        str(angles),
    ]
    index = args.index(omitted_flag)
    del args[index : index + 2]

    with pytest.raises(SystemExit) as exc_info:
        main(args)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert expected in captured.err
    assert not out_path.exists()


def test_import_cli_converts_nxtomo_to_npz(tmp_path: Path) -> None:
    nxs_path = tmp_path / "scan.nxs"
    npz_path = tmp_path / "scan.npz"
    write_projection_dataset(nxs_path)

    assert main(["import", str(nxs_path), "-o", str(npz_path)]) == 0

    dataset = load_dataset(npz_path)
    assert dataset.projections.shape == (2, 2, 4)
    np.testing.assert_allclose(dataset.angles, [0.0, 90.0])


def test_recon_cli_routes_tiny_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    write_projection_dataset(scan)
    captured: dict[str, object] = {}

    def fake_run(command: object, config_metadata: dict[str, object]) -> None:
        captured["method"] = command.method
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
        )
        == 0
    )

    assert captured == {"method": "fbp", "data": str(scan), "out": str(recon)}
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

    assert main(["recon", str(scan), "-o", str(recon)]) == 1
    captured = capsys.readouterr()
    assert captured.err == "tomojax recon: error: bad detector metadata\n"


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
        main(["recon", str(scan), "-o", str(recon)])


def test_recon_cli_executes_fbp_and_writes_volume_metadata(tmp_path: Path) -> None:
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    manifest = tmp_path / "recon-manifest.json"
    write_projection_dataset(scan)

    assert (
        main(
            [
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
                "--views-per-batch",
                "1",
                "--no-checkpoint-projector",
                "--manifest",
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
    assert resolved["method"] == "fbp"
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
    assert resolved["algorithm_config"]["warm_start"] == warm_start
    assert 1 <= resolved["algorithm_config"]["effective_iterations"] <= 6


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
                str(scan),
                "-o",
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
        main(["recon", str(scan), "-o", str(recon), "--det-u-px", "nan"])

    assert exc.value.code == 2


def test_recon_detector_center_override_records_effective_pixels() -> None:
    # check-public-imports: allow-private
    from tomojax.cli._recon_plan import _apply_detector_center_override

    detector = Detector(nu=8, nv=8, du=0.5, dv=2.0, center=(1.0, -2.0))
    geometry_meta: dict[str, object] = {"detector": detector.to_dict()}

    updated, provenance = _apply_detector_center_override(
        detector,
        geometry_meta,
        det_u_px=6.0,
        det_v_px=None,
    )

    assert updated.center == pytest.approx((3.0, -2.0))
    assert provenance["source"] == "cli_override"
    assert provenance["requested_px"] == {"det_u_px": 6.0, "det_v_px": None}
    assert provenance["effective_px"] == {"det_u_px": 6.0, "det_v_px": -1.0}
    assert provenance["effective_world"] == {"det_u": 3.0, "det_v": -2.0}
    assert geometry_meta["detector"]["det_center"] == [3.0, -2.0]


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
    save_projection_payload(path, projections=dataset.projections, metadata=metadata)

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
                "--views",
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

    args = ["align", str(scan), "-o", str(tmp_path / "out.nxs"), "--roi", "off"]
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
    assert main(["align", str(scan), "-o", str(out), "--roi", "off"]) == 0
    assert main(["align", str(scan), "-o", str(fresh), "--roi", "off", "--no-poses"]) == 0

    # The command line corrects on top of the saved poses, as tomojax.align does.
    python = tj.align(tj.load(scan))
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
    _ = config.write_text('loss = "4:phasecorr,2:ssim,1:l2_otsu"\n', encoding="utf-8")
    args = ["align", str(scan), "-o", str(aligned), "--mode", "full", "--quality", quality]
    assert main([*args, "--roi", "off", "--config", str(config), "--dry-run"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schedule"]["name"] == "setup_safe"
    assert payload["levels"] == [4, 2, 1]
    assert payload["output_path"] == str(aligned)
    assert payload["config"]["quality"] == quality
    assert "phasecorr" in payload["loss"]
    stages = payload["schedule"]["stages"]
    assert [stage["stage_name"] for stage in stages][:2] == ["cor", "detector_roll"]
    assert not aligned.exists()


def _pose_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str], config: str = "") -> object:
    scan = tmp_path / "scan.nxs"
    settings = tmp_path / "align.toml"
    write_projection_dataset(scan)
    _ = settings.write_text(config, encoding="utf-8")
    args = ["align", str(scan), "-o", str(tmp_path / "out.nxs"), "--mode", "pose"]
    assert main([*args, "--roi", "off", "--config", str(settings), "--dry-run"]) == 0
    return json.loads(capsys.readouterr().out)


def test_align_cli_pose_mode_defaults_to_the_coupled_solver(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _pose_plan(tmp_path, capsys)
    config = plan["config"]
    assert plan["pose_solver"] == "coupled"
    assert plan["loss"] == "L2LossSpec()"
    assert (config["ray_integrator"], config["tv_weight"], config["gather_dtype"]) == (
        "joseph",
        0.0,
        "fp32",
    )
    assert config["outer_iterations"] == 30
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
