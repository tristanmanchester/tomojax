"""Compare dry-run plans with execution and independently enumerated work."""

import json

import numpy as np
import pytest

import tomojax as tj
from tomojax._planning import resolve_reconstruction
from tomojax.cli._plans import plan_product  # check-public-imports: allow-private
from tomojax.cli.main import main
from tomojax.wire.resources import Snapshot


def scan(angles):
    grid = tj.Grid(7, 8, 5, 1, 1, 1, vol_center=(0.3, 0.2, -0.1))
    geometry = tj.ParallelGeometry(grid, tj.Detector(12, 6, 1, 1), angles)
    volume = np.zeros(grid.shape, np.float32)
    volume[1:4, 2:5, 1:4] = 1
    return tj.Scan(np.asarray(tj.project(geometry, volume)), geometry)


def test_snapshot_controls_estimate_while_run_reports_actual():
    source = scan([0, 45, 90, 135])
    grid = source.grid
    volume = 4 * 7 * 8 * 5
    work = resolve_reconstruction(source)
    for room, expected in ((10 * volume, "resident"), (2 * volume, "slabbed")):
        sample = (Snapshot(4, "cpu:0", "host", None, None, room),)
        planned = plan_product(source, work, snapshot=sample)
        assert plan_product(source, work, snapshot=sample) == planned
        assert planned.method == "fbp" and planned.iterations is None
        assert planned.placement.strategy == expected and planned.placement.snapshot == sample
        assert planned.placement.host_bytes >= source.projections.nbytes
        assert planned.grid.vol_center == grid.vol_center
    # The preflight's constrained sample has no effect on the allocator used by the run.
    out = np.empty(grid.shape, np.float32)
    actual = tj.reconstruct(source, out=out)
    assert actual.volume is out and actual.info["placement"]["strategy"] == "slabbed"
    assert actual.info["placement"]["retries"] == {}
    assert len(actual.info["placement"]["devices"]) == 1


def test_reconstruction_cli_plan_resolves_method_grid_and_budget(tmp_path, capsys):
    path = tmp_path / "scan.nxs"
    output = tmp_path / "unused.nxs"
    tj.save(path, scan(np.arange(0, 180, 10)))
    args = ["reconstruct", str(path), "--dry-run"]
    assert main([*args, "-o", str(output)]) == 0
    auto = json.loads(capsys.readouterr().out)
    assert auto["type"] == "reconstruction_plan" and auto["method"] == "fourier"
    assert not output.exists()
    assert main([*args, "--method", "cgls", "--iterations", "7", "--grid", "3", "5", "2"]) == 0
    explicit = json.loads(capsys.readouterr().out)
    assert (explicit["method"], explicit["iterations"]) == ("cgls", 7)
    assert [explicit["grid"][k] for k in ("nx", "ny", "nz")] == [3, 5, 2]
    assert explicit["reasons"] and explicit["placement"]["snapshot"]
    with pytest.raises(SystemExit) as error:
        main([*args, "--progress", "json"])
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out)["event"] == "error"


def test_alignment_plan_budgets_and_padded_levels_match_run(tmp_path, capsys):
    source = scan(np.arange(0, 360, 45))
    path = tmp_path / "scan.nxs"
    tj.save(path, source)
    config = tmp_path / "fit.toml"
    config.write_text("levels = [2, 1]\ncoarse_iterations = 2\nrefine_iterations = 0\n")
    args = [
        "align",
        str(path),
        "--fit",
        "center",
        "--iterations",
        "1",
        "--image-iterations",
        "3",
        "--config",
        str(config),
        "--grid",
        "7",
        "8",
        "5",
        "--dry-run",
    ]
    assert main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["type"] == "joint_plan" and not plan["bounded_working_grid"]
    levels = plan["levels"]
    # The solver pads two fine planes at each end for support. The coarse grid spans all nine.
    assert [(x["grid"]["nx"], x["grid"]["ny"], x["grid"]["nz"]) for x in levels] == [
        (4, 4, 5),
        (7, 8, 9),
    ]
    assert [(x["iterations"], x["image_iterations"]) for x in levels] == [(2, 3), (1, 3)]
    from tomojax.alignment import AlignConfig

    settings = AlignConfig(
        iterations=1, coarse_iterations=2, image_iterations=3, refine_iterations=0
    )
    result = tj.align(source, grid=source.grid, fit="center", config=settings)
    for planned, actual in zip(levels, result.info["levels"], strict=True):
        assert planned["grid"]["nz"] == actual["grid"]["nz"]
        assert actual["iterations"] == planned["iterations"]
        assert actual["image_iterations"] == planned["image_iterations"]
    assert result.info["placement"]["devices"]
    assert (
        main(["align", str(path), "--method", "opposite-views", "--fit", "center", "--dry-run"])
        == 0
    )
    centre = json.loads(capsys.readouterr().out)
    assert centre["type"] == "opposite_views_plan" and centre["pairs"] == 4
    assert centre["placement"]["devices"] == ["cpu:0"]


@pytest.mark.parametrize("method", ["fbp", "fourier", "cgls", "sirt", "mlem", "spdhg", "fista"])
def test_default_plan_for_each_method(method):
    source = scan(np.linspace(0, 180, 211, endpoint=False))
    snapshot = (Snapshot(0, "cpu:0", "host", None, None, 1),)
    plan = plan_product(source, resolve_reconstruction(source, method), snapshot=snapshot)
    assert plan.method == method and plan.grid.nz == 5
    if method == "fista":
        # The stream uploads at most 64 complete detector images in its default batch.
        assert plan.placement.device_bytes == 4 * (64 * 6 * 12 * 4)
