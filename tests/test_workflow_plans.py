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
    # A centre fit calibrates the geometry; it publishes no per-view table.
    assert result.scan.poses is None and type(result.scan.geometry) is tj.ParallelGeometry
    assert (
        main(["align", str(path), "--method", "opposite-views", "--fit", "center", "--dry-run"])
        == 0
    )
    centre = json.loads(capsys.readouterr().out)
    assert centre["type"] == "opposite_views_plan" and centre["pairs"] == 4
    assert centre["placement"]["devices"] == ["cpu:0"]


@pytest.mark.parametrize("method", ["fbp", "fourier", "cgls", "sirt", "mlem", "spdhg", "fista"])
def test_default_plan_for_each_method(method, monkeypatch):
    source = scan(np.linspace(0, 180, 211, endpoint=False))
    snapshot = (Snapshot(0, "cpu:0", "host", None, None, 1),)
    plan = plan_product(source, resolve_reconstruction(source, method), snapshot=snapshot)
    assert plan.method == method and plan.grid.nz == 5
    if method == "fista":
        # CPU FISTA keeps its state resident, even under a constrained memory sample.
        assert plan.placement.strategy == "resident"
        monkeypatch.setattr("tomojax.recon._backend.cuda_backend", lambda: True)
        device = (Snapshot(0, "gpu:0", "jax", None, None, 1),)
        plan = plan_product(source, resolve_reconstruction(source, method), snapshot=device)
        assert plan.placement.strategy == "slabbed"


def test_explicit_fourier_backend_needs_cupy_at_preflight(monkeypatch):
    import sys

    from tomojax.recon import FourierConfig

    source = scan([0, 45, 90, 135])
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(tj.InvalidParameter) as err:
        resolve_reconstruction(source, "fourier", config=FourierConfig(backend="cupy"))
    assert err.value.path == "config.backend"


@pytest.mark.parametrize("method", ["fbp", "sirt", "mlem", "fourier"])
@pytest.mark.parametrize("roll", [0.0, 18.0])
def test_low_memory_plan_and_run_agree_on_rolled_slab_eligibility(monkeypatch, method, roll):
    grid = tj.Grid(6, 6, 5, 1, 1, 1)
    geom = tj.ParallelGeometry(
        grid, tj.Detector(8, 5, 1, 1), np.arange(9) * 20.0, detector_roll=roll
    )
    subject = tj.Scan(np.ones((9, 5, 8), np.float32), geom)
    options = {"iterations": 1} if method in ("mlem", "sirt") else {}
    monkeypatch.setattr("tomojax.backends.device_free_memory_bytes", lambda **_: 1)
    snapshot = (Snapshot(0, "cpu:0", "host", None, None, 1),)
    planned = plan_product(
        subject, resolve_reconstruction(subject, method, **options), snapshot=snapshot
    )
    actual = tj.reconstruct(subject, method, **options)
    expected = "slabbed" if method == "fourier" or not roll else "resident"
    assert actual.info["placement"]["strategy"] == expected
    assert planned.placement.strategy == expected


@pytest.mark.parametrize(
    ("bad", "path"), [("grid", "grid"), ("angle", "source"), ("pose", "source")]
)
def test_preflight_rejects_invalid_projection_geometry(bad, path):
    class ScaledPose(tj.ParallelGeometry):
        def pose_matrix(self, index):
            matrix = np.array(super().pose_matrix(index))
            matrix[2, 2] = 1.05
            return matrix

    grid = tj.Grid(4, 4, 2, 1, 1, 1, vol_origin=(0, np.nan if bad == "grid" else 0, 0))
    kind = ScaledPose if bad == "pose" else tj.ParallelGeometry
    geometry = kind(grid, tj.Detector(4, 2, 1, 1), [0, np.nan if bad == "angle" else 90])
    source = tj.Scan(np.ones((2, 2, 4), np.float32), geometry)
    with pytest.raises(tj.InvalidParameter) as error:
        resolve_reconstruction(source, "cgls")
    assert error.value.path == path


@pytest.mark.parametrize("cone", [True, False])
@pytest.mark.parametrize("bad_row", [0, 16])
def test_host_fbp_preflight_uses_execution_row_windows(monkeypatch, cone, bad_row):
    from tomojax import backends

    monkeypatch.setattr(backends, "device_free_memory_bytes", lambda **kw: 1)
    grid = tj.Grid(3, 3, 1, 1, 1, 1)
    detector = tj.Detector(5, 33, 1, 1)
    angles = np.arange(0, 360, 30)
    geometry = (
        tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(50, 80))
        if cone
        else tj.ParallelGeometry(grid, detector, angles)
    )
    data = np.ones((12, 33, 5), np.float32)
    data[4, bad_row, 2] = np.nan
    source = tj.Scan(data, geometry)
    if bad_row == 0:
        resolve_reconstruction(source, "fbp")
        assert np.isfinite(tj.reconstruct(source, "fbp").volume).all()
    else:
        for action in (resolve_reconstruction, tj.reconstruct):
            with pytest.raises(tj.InvalidParameter) as error:
                action(source, "fbp")
            assert error.value.path == "source"


@pytest.mark.parametrize("field", ["support", "regularizer"])
def test_fista_preflight_respects_host_solver_capabilities(monkeypatch, field):
    from tomojax import recon

    grid = tj.Grid(8, 8, 4, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(7, 4, 1, 1), [0, 45, 90, 135])
    source = tj.Scan(np.zeros((4, 4, 7), np.float32), geometry)
    cfg = (
        recon.FISTAConfig(support=np.ones((1, 8, 8), bool), iterations=1, stream=False)
        if field == "support"
        else recon.FISTAConfig(regularizer="huber_tv", iterations=1, stream=False)
    )
    monkeypatch.setattr("tomojax.recon._backend.cuda_backend", lambda: True)
    monkeypatch.setattr("tomojax.backends.device_free_memory_bytes", lambda: 2**30)
    resolve_reconstruction(source, "fista", config=cfg)
    monkeypatch.setattr("tomojax.backends.device_free_memory_bytes", lambda: 1)
    for action in (resolve_reconstruction, tj.reconstruct):
        with pytest.raises(tj.InvalidParameter) as error:
            action(source, "fista", config=cfg)
        assert error.value.path == f"config.{field}"


@pytest.mark.parametrize(
    ("method", "rooms"),
    [
        ("sirt", [14000, 17279, 17280, 17281]),
        ("mlem", [8000, 11519, 11520, 11521]),
        ("fbp", [1799, 1800, 1801]),
        ("fdk", [1799, 1800, 1801]),
        ("fourier", [8000, 14000, 100000]),
        ("cgls", [5000, 5082, 5083, 14000]),
        ("fista", [8000, 13552, 13553, 14000]),
        ("spdhg", [1800, 8000, 14000]),
    ],
)
def test_plan_matches_execution_across_storage_threshold(monkeypatch, method, rooms):
    import jax

    grid = tj.Grid(6, 6, 5, 1, 1, 1)
    detector = tj.Detector(8, 5, 1, 1)
    angles = np.arange(9) * (40.0 if method == "fdk" else 20.0)
    geo = (
        tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(50, 80))
        if method == "fdk"
        else tj.ParallelGeometry(grid, detector, angles)
    )
    scan = tj.Scan(np.ones((9, 5, 8), np.float32), geo)
    name = "fbp" if method == "fdk" else method
    settings = {} if name in ("fbp", "fourier") else {"iterations": 1}
    if method in ("fista", "spdhg"):
        settings["tv_weight"] = 0.0
    if method == "cgls":
        settings = {"config": tj.recon.CGLSConfig(iterations=1, stream=True)}
    device = jax.devices()[0]
    for room in rooms:
        monkeypatch.setattr(
            "tomojax.backends.device_free_memory_bytes", lambda available=room, **kw: available
        )
        snapshot = (
            Snapshot(
                0,
                f"{device.platform}:{device.id}",
                "host" if device.platform == "cpu" else "jax",
                None,
                None,
                room,
            ),
        )
        plan = plan_product(scan, resolve_reconstruction(scan, name, **settings), snapshot=snapshot)
        actual = tj.reconstruct(scan, name, **settings).info["placement"]["strategy"]
        if method in ("sirt", "mlem"):
            assert actual == (
                "slabbed" if room < (17280 if method == "sirt" else 11520) else "resident"
            )
        assert plan.placement.strategy == actual, (method, room, actual)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("method", "rooms"),
    [
        ("cgls", [10_079, 10_080]),
        ("sirt", [15_000, 40_000]),
        ("mlem", [15_000, 40_000]),
    ],
)
def test_plan_matches_execution_across_cupy_pool_threshold(monkeypatch, method, rooms):
    grid = tj.Grid(6, 6, 5, 1, 1, 1)
    detector = tj.Detector(8, 5, 1, 1)
    angles = np.arange(9) * 40.0
    if method == "cgls":
        geometry = tj.ParallelGeometry(grid, detector, angles)
        settings = {"config": tj.recon.CGLSConfig(iterations=1, views_per_batch=1)}
    else:
        geometry = tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(50, 80))
        settings = {"iterations": 1}
    source = tj.Scan(np.ones((9, 5, 8), np.float32), geometry)
    # The JAX allocator cannot hold the state. CuPy may still hold the output.
    monkeypatch.setattr("tomojax.backends.device_free_memory_bytes", lambda **kw: 1)
    for available, expected in zip(rooms, ["slabbed", "resident"], strict=True):
        monkeypatch.setattr(
            "tomojax.core.backend_policy.cupy_available_bytes", lambda room=available: room
        )
        work = resolve_reconstruction(source, method, **settings)
        plan = plan_product(source, work)
        assert {s.scope: s.available for s in plan.placement.snapshot} == {
            "jax": 1,
            "cupy": available,
        }
        actual = tj.reconstruct(source, method, **settings).info["placement"]["strategy"]
        assert plan.placement.strategy == actual == expected
