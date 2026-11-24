"""``inspect --preview-dir``: slice previews of scans, and the centre sweep."""

from __future__ import annotations

from dataclasses import replace
import gc
import json
import weakref

import numpy as np
import pytest

import tomojax as tj
from tomojax import interactive
from tomojax.cli.main import main

pytestmark = pytest.mark.surface

_ANGLES = np.linspace(0.0, 180.0, 48, endpoint=False)


def _phantom(grid):
    z, y, x = np.mgrid[: grid.nz, : grid.ny, : grid.nx]
    centre = np.array([grid.nz, grid.ny, grid.nx]) / 2 - 0.5
    inside = (x - centre[2]) ** 2 + (y - centre[1]) ** 2 < (0.3 * grid.nx) ** 2
    return np.where(inside & (abs(z - centre[0]) < grid.nz / 3), 1.0, 0.0).astype(np.float32)


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
        capsys, str(path), "--preview-dir", str(directory), "--centres", "-3:3:1.5"
    )
    sweep = report["sweep"]
    assert [entry["centre_px"] for entry in sweep] == [-3.0, -1.5, 0.0, 1.5, 3.0]
    assert [entry["ordinal"] for entry in sweep] == [0, 1, 2, 3, 4]
    assert [entry["path"].rsplit("/", 1)[1] for entry in sweep][2] == "centre_02_+0.00px.png"
    assert all((directory / entry["path"].rsplit("/", 1)[1]).exists() for entry in sweep)
    assert (directory / "centre_sweep.png").exists()
    assert report["sweep_apply"]["recorded_px"] == pytest.approx(3.0)


def test_the_sweep_plane_at_the_true_centre_is_the_plane_of_the_corrected_scan():
    off, true = _scan(recorded=3.0), _scan()
    sweep = interactive.resolve_sweep(off, "z", 4, [0.0], method="fbp")
    (plane,) = interactive.sweep_planes(sweep)
    np.testing.assert_array_equal(plane, interactive.reconstruct_slice(true, "z", 4, method="fbp"))


def test_the_printed_command_applies_the_chosen_centre_in_length_units(tmp_path, capsys):
    detector = tj.Detector(16, 8, 0.25, 0.5, (0.0, 0.75))
    grid = tj.Grid(16, 16, 8, 0.25, 0.25, 0.5, vol_center=(0.0, 0.0, 0.75))
    path, directory = tmp_path / "scan.nxs", tmp_path / "sweep"
    tj.save(path, _scan(detector, grid))
    report = _inspect_json(capsys, str(path), "--preview-dir", str(directory), "--centres", "-2,2")
    apply = report["sweep_apply"]
    chosen = report["sweep"][1]
    u = chosen["centre_length"]
    assert u == pytest.approx(2 * 0.25) and apply["v"] == 0.75
    command = apply["command"].replace("U", repr(u)).split()
    out = tmp_path / "centred.nxs"
    assert command[:2] == ["tomojax", "correct"]
    assert main([*command[1:3], "-o", str(out), *command[5:], "--progress", "none"]) == 0
    centred = tj.load(out)
    assert centred.detector.center == pytest.approx((u, 0.75))
    sweep = interactive.resolve_sweep(tj.load(path), "z", 4, [2.0])
    swept = next(interactive.sweep_planes(sweep))
    np.testing.assert_allclose(interactive.reconstruct_slice(centred, "z", 4), swept, atol=1e-6)


def test_repeated_candidates_and_slices_keep_their_order_and_never_collide(tmp_path, capsys):
    path, directory = tmp_path / "scan.nxs", tmp_path / "pv"
    tj.save(path, _scan())
    argv = [str(path), "--preview-dir", str(directory), "--centres", "1,1,1.001"]
    report = _inspect_json(capsys, *argv, "--slice", "z=2", "--slice", "z=5", "--slice", "z=2")
    names = [entry["path"].rsplit("/", 1)[1] for entry in report["sweep"]]
    assert names == ["centre_00_+1.00px.png", "centre_01_+1.00px.png", "centre_02_+1.00px.png"]
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
    assert interactive.plan_planes(scan, [("z", 0)], method="fbp").full is None
    with pytest.raises(tj.InvalidParameter):
        interactive.plan_planes(scan, [("z", 0), ("y", 3)], method="fbp")


def test_planning_solves_nothing_and_planes_share_one_solve(monkeypatch):
    calls = []
    solve = interactive.reconstruct_resolved

    def counted(plan, **kwargs):
        calls.append(plan.method)
        return solve(plan, **kwargs)

    monkeypatch.setattr(interactive, "reconstruct_resolved", counted)
    plan = interactive.plan_planes(
        _scan(), [("y", 2), ("x", 3), ("z", 4)], method="sirt", iterations=2
    )
    assert calls == [] and plan.full is not None
    images = interactive.planes_of(plan)
    assert calls == ["sirt"] and [i.shape for i in images] == [(8, 16), (8, 16), (16, 16)]


def test_the_route_chosen_in_planning_is_the_route_run(monkeypatch):
    import importlib

    fbp_host = importlib.import_module("tomojax.recon.fbp_host")

    grid = tj.Grid(6, 6, 33, 0.3, 0.3, 0.3)
    detector = tj.Detector(8, 33, 0.3, 0.3)
    data = np.random.default_rng(3).uniform(size=(9, 33, 8)).astype(np.float32)
    scan = tj.Scan(data, tj.ParallelGeometry(grid, detector, np.arange(9) * 20.0))
    monkeypatch.setattr("tomojax.backends.device_free_memory_bytes", lambda **kw: 1)
    plan = interactive.plan_planes(scan, [("z", 5)], method="fbp")
    assert plan.plans[0].route.kind == "slab"
    # Memory frees up before the run: the plane still comes from its planned host slab.
    monkeypatch.setattr("tomojax.backends.device_free_memory_bytes", lambda **kw: 2**40)
    slabs, whole = [], []
    host = fbp_host.fbp_host
    monkeypatch.setattr(fbp_host, "fbp_host", lambda *a, **k: slabs.append(1) or host(*a, **k))
    monkeypatch.setattr(interactive, "reconstruct_resolved", lambda *a, **k: whole.append(1))
    (image,) = interactive.planes_of(plan)
    assert slabs == [1] and whole == [] and image.shape == (6, 6)


def test_a_sweep_keeps_one_candidate_geometry_alive(monkeypatch):
    from tomojax.geometry import ConeSegments

    detector = tj.Detector(16, 8, 1.0, 1.0)
    grid = tj.Grid(16, 16, 8, 1.0, 1.0, 1.0)
    turn = np.linspace(0.0, 360.0, 48, endpoint=False)
    parts = tuple(
        tj.ConeGeometry(grid, detector, list(turn[i::4]), tj.ConeBeam(60.0, 120.0))
        for i in range(4)
    )
    scan = tj.Scan(np.ones((48, 8, 16), np.float32), ConeSegments(parts))
    alive = []

    def plane(plan, axis, index):
        alive[:] = [ref for ref in alive if ref() is not None]
        alive.append(weakref.ref(plan.scan.geometry))
        return np.zeros((16, 16), np.float32)

    monkeypatch.setattr(interactive, "resolved_slice", plane)
    sweep = interactive.resolve_sweep(scan, "z", 4, list(np.linspace(-2, 2, 9)), method="fbp")
    peak = 0
    for _ in interactive.sweep_planes(sweep):
        gc.collect()
        peak = max(peak, sum(ref() is not None for ref in alive))
    assert peak <= 2  # the template's own geometry and the candidate being made


def test_a_thousand_candidates_list_to_a_file_not_the_summary(monkeypatch, tmp_path, capsys):
    path, directory = tmp_path / "scan.nxs", tmp_path / "pv"
    tj.save(path, _scan())

    class Stub:
        full_solves = 0

    monkeypatch.setattr(interactive, "resolve_sweep", lambda *a, **k: Stub())
    monkeypatch.setattr(
        interactive, "sweep_planes", lambda _p: (np.zeros((4, 4), np.float32) for _ in range(1000))
    )
    argv = ["inspect", str(path), "--preview-dir", str(directory), "--centres", "0:999:1"]
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
        capsys, str(path), "--preview-dir", str(directory), "--centres", "-1,0,1"
    )
    assert len(report["sweep"]) == 3 and (directory / "centre_sweep.png").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["--centres", "-3:3:1"],
        ["--centres", "-3,0,3"],
        ["--centres=-3:3:1"],
    ],
)
def test_signed_centres_reach_the_parser_whole(tmp_path, capsys, argv):
    path = tmp_path / "scan.nxs"
    tj.save(path, _scan())
    report = _inspect_json(capsys, str(path), "--preview-dir", str(tmp_path / "pv"), *argv)
    assert report["sweep"][0]["centre_px"] == -3.0


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--centres", "1:0:1"], "STOP at least START"),
        (["--centres", "a,b"], "START:STOP:STEP or a comma list"),
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
        main(["inspect", "scan.nxs", "--centres", "0,1"])
    assert error.value.code == 2 and "--preview-dir" in capsys.readouterr().err
    raw = tmp_path / "raw.nxs"
    write_raw_nxtomo(raw)
    with pytest.raises(SystemExit) as error:
        main(["inspect", str(raw), "--preview-dir", str(tmp_path / "pv"), "--centres", "0,1"])
    assert error.value.code == 2
    err = capsys.readouterr().err
    assert "holds raw frames" in err and "tomojax correct" in err


def test_a_scan_previews_its_reconstructed_central_slice(tmp_path, capsys):
    path, directory = tmp_path / "scan.nxs", tmp_path / "pv"
    tj.save(path, _scan())
    report = _inspect_json(capsys, str(path), "--preview-dir", str(directory))
    assert sorted(report["previews"]) == ["projection.png", "slice_z_4.png"]
    assert report["valid"] is True
