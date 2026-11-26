"""``inspect --preview-dir``: slice previews of scans, and the center sweep."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

import tomojax as tj
from tomojax import interactive
from tomojax.cli.main import main

pytestmark = pytest.mark.surface

_ANGLES = np.linspace(0.0, 180.0, 48, endpoint=False)


def _phantom(grid):
    z, y, x = np.mgrid[: grid.nz, : grid.ny, : grid.nx]
    center = np.array([grid.nz, grid.ny, grid.nx]) / 2 - 0.5
    inside = (x - center[2]) ** 2 + (y - center[1]) ** 2 < (0.3 * grid.nx) ** 2
    return np.where(inside & (abs(z - center[0]) < grid.nz / 3), 1.0, 0.0).astype(np.float32)


def _scan(detector=None, grid=None, recorded=0.0):
    """A parallel scan simulated at the detector middle, recorded ``recorded`` pixels off."""
    grid = grid or tj.Grid(16, 16, 8, 1.0, 1.0, 1.0)
    detector = detector or tj.Detector(16, 8, 1.0, 1.0)
    geometry = tj.ParallelGeometry(grid, detector, _ANGLES)
    data = np.asarray(tj.project(geometry, _phantom(grid)))
    shifted = replace(
        detector, center=(detector.center[0] + recorded * detector.du, detector.center[1])
    )
    return tj.Scan(data, replace(geometry, detector=shifted))


def _inspect_json(capsys, *argv):
    assert main(["inspect", *argv, "--json"]) in (0, 1)
    return json.loads(capsys.readouterr().out)


def test_a_sweep_writes_every_candidate_in_order_and_a_sheet(tmp_path, capsys):
    path, directory = tmp_path / "scan.nxs", tmp_path / "sweep"
    tj.save(path, _scan(recorded=3.0))
    report = _inspect_json(
        capsys, str(path), "--preview-dir", str(directory), "--centers", "-3:3:1.5"
    )
    sweep = report["sweep"]
    assert [entry["center_px"] for entry in sweep] == [-3.0, -1.5, 0.0, 1.5, 3.0]
    assert [entry["ordinal"] for entry in sweep] == [0, 1, 2, 3, 4]
    assert [entry["path"].rsplit("/", 1)[1] for entry in sweep][2] == "center_02_+0.00px.png"
    assert all((directory / entry["path"].rsplit("/", 1)[1]).exists() for entry in sweep)
    assert (directory / "center_sweep.png").exists()
    assert report["sweep_apply"]["recorded_px"] == pytest.approx(3.0)


def test_the_sweep_plane_at_the_true_center_is_the_plane_of_the_corrected_scan():
    off, true = _scan(recorded=3.0), _scan()
    (plane,) = interactive.center_sweep(off, "z", 4, [0.0], method="fbp")
    np.testing.assert_array_equal(plane, interactive.reconstruct_slice(true, "z", 4, method="fbp"))


def test_the_printed_command_applies_the_chosen_center_in_length_units(tmp_path, capsys):
    detector = tj.Detector(16, 8, 0.25, 0.5, (0.0, 0.75))
    grid = tj.Grid(16, 16, 8, 0.25, 0.25, 0.5, vol_center=(0.0, 0.0, 0.75))
    path, directory = tmp_path / "scan.nxs", tmp_path / "sweep"
    tj.save(path, _scan(detector, grid))
    report = _inspect_json(capsys, str(path), "--preview-dir", str(directory), "--centers", "-2,2")
    apply = report["sweep_apply"]
    chosen = report["sweep"][1]
    u = chosen["center_length"]
    assert u == pytest.approx(2 * 0.25) and apply["v"] == 0.75
    command = apply["command"].replace("U", repr(u)).split()
    out = tmp_path / "centered.nxs"
    assert command[:2] == ["tomojax", "correct"]
    assert main([*command[1:3], "-o", str(out), *command[5:], "--progress", "none"]) == 0
    centered = tj.load(out)
    assert centered.detector.center == pytest.approx((u, 0.75))
    (swept,) = interactive.center_sweep(tj.load(path), "z", 4, [2.0])
    np.testing.assert_allclose(interactive.reconstruct_slice(centered, "z", 4), swept, atol=1e-6)


def test_repeated_candidates_and_slices_keep_their_order_and_never_collide(tmp_path, capsys):
    path, directory = tmp_path / "scan.nxs", tmp_path / "pv"
    tj.save(path, _scan())
    argv = [str(path), "--preview-dir", str(directory), "--centers", "1,1,1.001"]
    report = _inspect_json(capsys, *argv, "--slice", "z=2", "--slice", "z=5", "--slice", "z=2")
    names = [entry["path"].rsplit("/", 1)[1] for entry in report["sweep"]]
    assert names == ["center_00_+1.00px.png", "center_01_+1.00px.png", "center_02_+1.00px.png"]
    requests = [(r["axis"], r["index"]) for r in report["slice_requests"]]
    assert requests == [("z", 2), ("z", 5), ("z", 2)]
    assert report["slice_requests"][0]["path"] == report["slice_requests"][2]["path"]
    assert sorted(n for n in report["previews"] if n.startswith("slice")) == [
        "slice_z_2.png",
        "slice_z_5.png",
    ]
    # The stream's result keeps the report and points at the full listing.
    assert (
        main(
            [
                "inspect",
                *argv,
                "--force",
                "--progress",
                "json",
                "--slice",
                "z=2",
                "--slice",
                "z=5",
                "--slice",
                "z=2",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    summary = result["info"]["summary"]
    listing = json.loads((directory / "inspect-previews.json").read_text())
    assert summary["valid"] is True and summary["preview_listing"].endswith("inspect-previews.json")
    assert [(r["axis"], r["index"]) for r in listing["slice_requests"]] == requests


def test_a_plane_needing_the_whole_volume_validates_the_whole_volume(monkeypatch):
    grid = tj.Grid(6, 6, 33, 1, 1, 1)
    detector = tj.Detector(8, 33, 1, 1)
    data = np.random.default_rng(19).uniform(size=(9, 33, 8)).astype(np.float32)
    data[3, 32, 5] = np.nan  # outside the axial slab of plane 0
    scan = tj.Scan(data, tj.ParallelGeometry(grid, detector, np.arange(9) * 20.0))
    monkeypatch.setattr("tomojax.backends.device_free_memory_bytes", lambda **kw: 1)
    (plane,) = interactive.reconstruct_slices(scan, [("z", 0)], method="fbp")
    assert np.isfinite(plane).all()
    with pytest.raises(tj.InvalidParameter):
        interactive.reconstruct_slices(scan, [("z", 0), ("y", 3)], method="fbp")


def test_planes_that_need_the_volume_share_one_solve(monkeypatch):
    calls = []
    solve = interactive.reconstruct_resolved

    def counted(plan, **kwargs):
        calls.append(plan.method)
        return solve(plan, **kwargs)

    monkeypatch.setattr(interactive, "reconstruct_resolved", counted)
    requests = [("y", 2), ("x", 3), ("z", 4)]
    images = interactive.reconstruct_slices(_scan(), requests, method="sirt", iterations=2)
    assert calls == ["sirt"] and [i.shape for i in images] == [(8, 16), (8, 16), (16, 16)]


def test_a_thousand_candidates_list_to_a_file_not_the_summary(monkeypatch, tmp_path, capsys):
    path, directory = tmp_path / "scan.nxs", tmp_path / "pv"
    tj.save(path, _scan())

    monkeypatch.setattr(interactive, "resolve_sweep", lambda *a, **k: ())
    monkeypatch.setattr(
        interactive, "sweep_planes", lambda *a: (np.zeros((4, 4), np.float32) for _ in range(1000))
    )
    argv = ["inspect", str(path), "--preview-dir", str(directory), "--centers", "0:999:1"]
    assert main([*argv, "--progress", "json"]) == 0
    record = json.loads(capsys.readouterr().out.splitlines()[-1])
    listing = json.loads((directory / "inspect-previews.json").read_text())
    assert len(listing["sweep"]) == 1000 and record["info"]["summary"]["valid"] is True
    assert "sweep" not in record["info"]["summary"]


@pytest.mark.parametrize("geometry", ["lamino", "cone"])
def test_laminography_and_cone_scans_sweep(tmp_path, capsys, geometry):
    path, directory = tmp_path / "scan.nxs", tmp_path / "sweep"
    extra = ["--tilt", "30"] if geometry == "lamino" else []
    argv = ["simulate", "-o", str(path), "--size", "12", "--geometry", geometry, *extra]
    assert main([*argv, "--progress", "none"]) == 0
    capsys.readouterr()
    report = _inspect_json(
        capsys, str(path), "--preview-dir", str(directory), "--centers", "-1,0,1"
    )
    assert len(report["sweep"]) == 3 and (directory / "center_sweep.png").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["--centers", "-3:3:1"],
        ["--centers", "-3,0,3"],
        ["--centers=-3:3:1"],
    ],
)
def test_signed_centers_reach_the_parser_whole(tmp_path, capsys, argv):
    path = tmp_path / "scan.nxs"
    tj.save(path, _scan())
    report = _inspect_json(capsys, str(path), "--preview-dir", str(tmp_path / "pv"), *argv)
    assert report["sweep"][0]["center_px"] == -3.0


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--centers", "1:0:1"], "STOP at least START"),
        (["--centers", "a,b"], "START:STOP:STEP or a comma list"),
        (["--slice", "w=3"], "AXIS=INDEX"),
    ],
)
def test_malformed_values_are_usage_errors_with_hints(tmp_path, capsys, argv, message):
    with pytest.raises(SystemExit) as error:
        main(["inspect", "scan.nxs", "--preview-dir", str(tmp_path), *argv])
    assert error.value.code == 2
    err = capsys.readouterr().err
    assert message in err and "hint:" in err


def test_sweeps_need_a_folder_and_a_scan(tmp_path, capsys):
    from ._helpers import write_raw_nxtomo

    with pytest.raises(SystemExit) as error:
        main(["inspect", "scan.nxs", "--centers", "0,1"])
    assert error.value.code == 2 and "--preview-dir" in capsys.readouterr().err
    raw = tmp_path / "raw.nxs"
    write_raw_nxtomo(raw)
    with pytest.raises(SystemExit) as error:
        main(["inspect", str(raw), "--preview-dir", str(tmp_path / "pv"), "--centers", "0,1"])
    assert error.value.code == 2
    err = capsys.readouterr().err
    assert "holds raw frames" in err and "tomojax correct" in err


def test_a_scan_previews_its_reconstructed_central_slice(tmp_path, capsys):
    path, directory = tmp_path / "scan.nxs", tmp_path / "pv"
    tj.save(path, _scan())
    report = _inspect_json(capsys, str(path), "--preview-dir", str(directory))
    assert sorted(report["previews"]) == ["projection.png", "slice_z_4.png"]
    assert report["valid"] is True


def test_a_parallel_scan_previews_and_sweeps_without_a_full_solve(tmp_path, capsys, monkeypatch):
    def full_solve(plan, **kwargs):
        raise AssertionError(f"{plan.method} reconstructed the whole volume for one plane")

    monkeypatch.setattr(interactive, "reconstruct_resolved", full_solve)
    path, directory = tmp_path / "scan.nxs", tmp_path / "pv"
    tj.save(path, _scan())
    argv = ["--preview-dir", str(directory), "--centers", "-1,0,1", "--slice", "z=4"]
    report = _inspect_json(capsys, str(path), *argv)
    assert "slice_z_4.png" in report["previews"] and len(report["sweep"]) == 3


def test_a_bad_slice_writes_nothing_and_a_rerun_needs_force(tmp_path, capsys):
    scan, directory = tmp_path / "scan.nxs", tmp_path / "pv"
    tj.save(scan, _scan())
    assert main(["inspect", str(scan), "--preview-dir", str(directory), "--slice", "z=99"]) == 1
    assert not list(directory.glob("*.png"))
    assert main(["inspect", str(scan), "--preview-dir", str(directory), "--slice", "z=4"]) == 0
    recon = tmp_path / "recon.nxs"
    assert main(["reconstruct", str(scan), "-o", str(recon), "--progress", "none"]) == 0
    saved = tmp_path / "saved"
    assert main(["inspect", str(recon), "--preview-dir", str(saved)]) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit) as error:
        main(["inspect", str(recon), "--preview-dir", str(saved)])
    assert error.value.code == 2 and "slice_z_4.png" in capsys.readouterr().err


def test_the_report_gives_the_grid_as_nx_ny_nz(tmp_path, capsys):
    path = tmp_path / "scan.nxs"
    tj.save(path, _scan())
    assert main(["inspect", str(path)]) == 0
    assert "for 16 x 16 x 8 voxels" in capsys.readouterr().out
