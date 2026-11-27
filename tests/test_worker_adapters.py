"""Drive the production executor through the JSONL client interface."""

import json
from pathlib import Path
import subprocess
import sys

import h5py
import jax
import numpy as np
import pytest

import tomojax as tj

pytestmark = pytest.mark.surface


@pytest.fixture(scope="module")
def worker():
    with subprocess.Popen(
        [sys.executable, "-m", "tomojax.cli.main", "serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as proc:
        try:
            ready = json.loads(proc.stdout.readline())
            # The references below run in this process; the worker must use the same backend.
            assert {d["platform"] for d in ready["devices"]} == {jax.default_backend()}, ready
            assert {"validate", "plan", "run", "plane", "inspect", "open"} <= set(
                ready["capabilities"]
            )

            def call(command, *, events=None, **parameters):
                nonlocal request
                request += 1
                rid = str(request)
                proc.stdin.write(
                    json.dumps({"id": rid, "command": {"type": command, **parameters}}) + "\n"
                )
                proc.stdin.flush()
                while line := proc.stdout.readline():
                    row = json.loads(line)
                    assert row["request"] == rid
                    if row["type"] != "progress":
                        return row
                    if events is not None:
                        events.append(row["record"])
                pytest.fail(f"unexpected EOF from worker, status {proc.poll()}")

            request = 0
            yield call
            assert call("shutdown")["type"] == "result"
            assert proc.wait(10) == 0
        finally:
            if proc.poll() is None:
                proc.kill()


def output(reply):
    assert reply["type"] == "result", reply
    return reply["result"]


@pytest.fixture(scope="module")
def simulation(worker):
    return output(
        worker("run", operation={"type": "simulate", "phantom": "sphere", "size": 8, "n_views": 12})
    )


def test_matched_operators_and_ordered_handle_lineage(worker, simulation, tmp_path):
    sim = simulation["value"]
    projected = output(
        worker(
            "run",
            operation=dict(type="project", source=sim["scan"], volume=sim["phantom"]["storage"]),
        )
    )
    assert projected["manifest"]["inputs"] == [simulation["manifest"]] * 2
    reference = tj.simulate("sphere", size=8, n_views=12)
    actual = np.load(projected["value"]["projections"]["storage"]["path"])
    np.testing.assert_allclose(
        actual, tj.project(reference.scan.geometry, reference.phantom), atol=1e-6
    )
    file = tmp_path / "input.h5"
    with h5py.File(file, "w") as f:
        f["samples"] = actual
    back = output(
        worker(
            "run",
            operation=dict(
                type="backproject",
                source=sim["scan"],
                projections=dict(type="hdf5", path=str(file), dataset="samples"),
            ),
        )
    )
    np.testing.assert_allclose(
        np.load(back["value"]["volume"]["storage"]["path"]),
        tj.backproject(reference.scan.geometry, actual),
        atol=1e-5,
    )


def test_save_reopen_plan_edit_and_export(worker, simulation, tmp_path):
    scan = simulation["value"]["scan"]
    for operation in [
        dict(type="select", views=[0, 2, 4, 7, 11]),
        dict(type="crop", rows=[1, 7], cols=[0, 6]),
        dict(type="bin", binning=2),
        dict(type="poses", poses=None),
        dict(type="correct"),
    ]:
        operation["source"] = scan
        assert output(worker("validate", operation=operation))["value"]["type"] == "validated"
        result = output(worker("run", operation=operation))
        scan = dict(type="member", handle=result["value"]["handle"], member="self")
    assert result["value"]["projections"]["shape"] == [5, 3, 3]
    recipe = dict(
        type="reconstruct", source=scan, method="fbp", config=dict(type="FBPConfig", backend="jax")
    )
    planned = output(worker("plan", operation=recipe))["value"]
    assert planned["method"] == "fbp" and planned["placement"]["snapshot"]
    recon = output(worker("run", operation=recipe))
    root = dict(type="member", handle=recon["value"]["handle"], member="self")
    file = tmp_path / "volume.nxs"
    saved = output(worker("run", operation=dict(type="save", source=root, path=str(file))))
    assert json.loads(Path(saved["value"]["paths"][1]).read_text()) == saved["manifest"]
    assert output(worker("inspect", path=str(file)))["value"]["report"]["valid"]
    reopened = output(worker("open", kind="reconstruction", path=str(file)))["value"]
    assert reopened["scan"] is None and reopened["volume"]["storage"]["type"] == "hdf5"
    reopened_root = dict(type="member", handle=reopened["handle"], member="self")
    for name, options, field in [
        ("copy.nxs", dict(include_scan=True), "include_scan"),
        ("copy.txt", {}, "path"),
    ]:
        path = tmp_path / name
        recipe = dict(type="save", source=reopened_root, path=str(path), **options)
        for cmd in ("validate", "run"):
            error = worker(cmd, operation=recipe)["error"]
            assert (error["kind"], error["path"]) == ("invalid_parameter", field)
        assert not path.exists() and not path.with_name(path.name + ".manifest.json").exists()
    ref = dict(type="member", handle=reopened["handle"], member="volume")
    sliced = output(worker("plane", source=ref, axis="y", index=1))["value"]["plane"]
    with h5py.File(file) as f:
        np.testing.assert_array_equal(
            np.load(sliced["storage"]["path"]), f[reopened["volume"]["storage"]["dataset"]][:, 1, :]
        )
    raw = tmp_path / "result.raw"
    exported = output(worker("run", operation=dict(type="export", source=root, path=str(raw))))[
        "value"
    ]
    assert set(map(Path, exported["paths"])) == {
        raw,
        raw.with_suffix(".raw.json"),
        raw.with_name("result.raw.manifest.json"),
    }
    assert raw.stat().st_size == np.prod(recon["value"]["volume"]["shape"]) * 4


@pytest.mark.parametrize(
    ("operation", "path"),
    [
        (
            dict(type="reconstruct", method="fbp", config=dict(type="FBPConfig", filter="banana")),
            "config.filter",
        ),
        (dict(type="reconstruct", method="fbp", iterations=2), "iterations"),
        (dict(type="reconstruct", method="auto", warm_start=True), "warm_start"),
        (
            dict(
                type="reconstruct",
                method="fbp",
                config=dict(type="FBPConfig", devices=["nonexistent"]),
            ),
            "config.devices",
        ),
        (dict(type="align", fit="translations", method="opposite-views"), "fit"),
        (
            dict(
                type="correct",
                corrections=[dict(type="Rings"), dict(type="Zingers", threshold=-1)],
            ),
            "corrections[1].threshold",
        ),
        (dict(type="select", views=[2, 1]), "views"),
        (dict(type="crop", rows=[0, 0], cols=[0, 2]), "rows"),
        (dict(type="bin", binning=1000), "binning"),
    ],
)
def test_validate_and_run_return_same_field(worker, simulation, operation, path):
    operation = dict(operation, source=simulation["value"]["scan"])
    for command in ("validate", "run"):
        reply = worker(command, operation=operation)
        assert reply["type"] == "error", reply
        assert (reply["error"]["kind"], reply["error"]["path"]) == ("invalid_parameter", path)


def test_raw_frames_stream_into_reopenable_slab_volume(worker, tmp_path):
    raw = tmp_path / "raw.h5"
    counts = np.random.default_rng(7).integers(430, 980, size=(17, 5, 13), dtype=np.uint16)
    with h5py.File(raw, "w") as f:
        f["exchange/data"] = counts
        f["exchange/theta"] = np.linspace(0, 180, 17, endpoint=False)
    opening = dict(
        kind="frames",
        path=str(raw),
        flats=1000,
        darks=10,
        detector=dict(nu=13, nv=5, du=1, dv=1),
        geometry=dict(kind="parallel"),
    )
    assert (
        output(worker("validate", operation=dict(type="open", **opening)))["value"]["type"]
        == "validated"
    )
    opened = output(worker("open", **opening))["value"]
    assert opened["counts"]["domain"] == "counts"
    root = dict(type="member", handle=opened["handle"], member="self")
    recipe = dict(
        type="reconstruct_slabs",
        path=str(tmp_path / "slabs.nxs"),
        source=root,
        method="fbp",
        grid=dict(nx=12, ny=11, nz=5, vx=1, vy=1, vz=1, vol_origin=[-5.0, -5.5, -2.0]),
        config=dict(type="FBPConfig", backend="jax", exterior="zero"),
    )
    reply = output(worker("run", operation=recipe))
    result = reply["value"]["paths"]
    saved = tj.load_reconstruction(result[0])
    assert saved.grid.vol_origin == (-5.0, -5.5, -2.0)
    assert saved.info["source"] == str(raw)
    assert saved.info["config"]["exterior"] == "zero"
    actual = saved.volume
    frames = tj.load_frames(raw, detector=tj.Detector(13, 5, 1, 1), flats=1000, darks=10)
    from tomojax.recon import FBPConfig

    expected = tj.reconstruct(
        frames.corrected(),
        "fbp",
        grid=tj.Grid(**recipe["grid"]),
        config=FBPConfig(backend="jax", exterior="zero"),
    )
    assert reply["info"]["summary"]["truncation"] == saved.info["truncation"]
    assert tj.inspect(result[0])["volume"]["reconstruction_info"] == saved.info
    assert saved.info["truncation"] == pytest.approx(expected.info["truncation"])
    np.testing.assert_allclose(actual, expected.volume, rtol=2e-5, atol=3e-6)
    for command in ("validate", "run"):
        failed = worker(
            command, operation={**recipe, "force": True, "corrections": [dict(type="RejectViews")]}
        )
        assert failed["error"]["path"] == "corrections[0]"


def test_combined_scan_keeps_repeated_parents_and_alignment_exposes_scan(worker):
    recipe = dict(type="simulate", phantom="sphere", size=12, geometry="cone", n_views=24)
    cone = output(worker("run", operation=recipe))
    scan = cone["value"]["scan"]
    result = output(worker("run", operation=dict(type="combine", sources=[scan, scan])))
    assert result["value"]["projections"]["shape"][0] == 48
    assert result["manifest"]["inputs"] == [cone["manifest"], cone["manifest"]]
    # An opposite-view center fit is defined for axial parallel scans.
    parallel = output(worker("run", operation={**recipe, "geometry": "parallel", "rotation": 360}))
    # Opposite views fit the center without being told to.
    fit = dict(type="align", source=parallel["value"]["scan"], method="opposite-views")
    assert output(worker("plan", operation=fit))["value"]["pairs"] > 0
    alignment = output(worker("run", operation=fit))
    poses = alignment["value"]["poses"]
    assert poses["shape"] == [24, 6]
    assert [c["unit"] for c in poses["columns"]][:3] == ["degree"] * 3
    recon = output(
        worker(
            "run",
            operation=dict(type="reconstruct", source=alignment["value"]["scan"], method="fbp"),
        )
    )
    assert recon["manifest"]["inputs"] == [alignment["manifest"]]


@pytest.mark.parametrize(
    ("operation", "path"),
    [
        (dict(type="simulate", size=0), "size"),
        (dict(type="simulate", n_views=-1), "n_views"),
        (dict(type="simulate", poisson_scale=2, artifacts={"poisson_scale": 3}), "poisson_scale"),
        (dict(type="simulate", detector={"nu": 2, "nv": 1, "du": -0.5, "dv": 1}), "detector.du"),
        (dict(type="simulate", geometry="cone", size=2, source_to_axis=-1), "source_to_axis"),
        (
            dict(type="open", kind="scan", path="missing.h5", geometry={"kind": "lamino"}),
            "geometry.tilt",
        ),
        (
            dict(
                type="open",
                kind="scan",
                path="missing.h5",
                geometry={"kind": "cone", "source_to_axis": 2},
            ),
            "geometry.source_to_detector",
        ),
    ],
)
def test_constructor_paths_precede_io_and_dispatch(worker, operation, path):
    replies = [worker("validate", operation=operation)]
    replies.append(
        worker("open", **{k: v for k, v in operation.items() if k != "type"})
        if operation["type"] == "open"
        else worker("run", operation=operation)
    )
    for reply in replies:
        assert reply["type"] == "error", reply
        assert (reply["error"]["kind"], reply["error"]["path"]) == ("invalid_parameter", path)


def test_flat_field_settings_are_refused_on_line_integrals(worker, simulation):
    source = simulation["value"]["scan"]
    for settings, field in [
        (dict(flat_weights="position"), "flat_weights"),
        (dict(epsilon=1e-3), "epsilon"),
    ]:
        recipe = dict(type="correct", source=source, **settings)
        for command in ("validate", "run"):
            failure = worker(command, operation=recipe)["error"]
            assert (failure["kind"], failure["path"]) == ("invalid_parameter", field)


def test_mask_shape_and_subsets_depend_on_the_selected_scan(worker, simulation, tmp_path):
    source = simulation["value"]["scan"]
    path = tmp_path / "support.npy"
    np.save(path, np.ones((3, 2), np.uint8))
    for config, field in [
        (dict(type="SIRTConfig", subsets=13), "config.subsets"),
        (dict(type="SIRTConfig", support=dict(type="npy", path=str(path))), "config.support"),
    ]:
        recipe = dict(type="reconstruct", source=source, method="sirt", config=config)
        for command in ("validate", "run"):
            failure = worker(command, operation=recipe)["error"]
            assert (failure["kind"], failure["path"]) == ("invalid_parameter", field)
    recipe = dict(
        type="reconstruct",
        source=source,
        method="sirt",
        config=dict(type="SIRTConfig", subsets=12, iterations=0),
    )
    assert output(worker("validate", operation=recipe))["value"]["type"] == "validated"
    assert output(worker("run", operation=recipe))["value"]["volume"]["shape"] == [8, 8, 8]


def test_short_cone_and_single_direction_fourier_fail_during_preflight(worker, simulation):
    cone = output(
        worker(
            "run", operation=dict(type="simulate", geometry="cone", size=6, n_views=10, rotation=45)
        )
    )
    selected = output(
        worker("run", operation=dict(type="select", source=simulation["value"]["scan"], views=[3]))
    )
    for ref, method in [
        (cone["value"]["scan"], "fbp"),
        (dict(type="member", handle=selected["value"]["handle"], member="self"), "fourier"),
    ]:
        recipe = dict(type="reconstruct", method=method, source=ref)
        for command in ("validate", "plan", "run"):
            error = worker(command, operation=recipe)["error"]
            assert (error["kind"], error["path"]) == ("invalid_parameter", "method")


def test_force_preserves_an_open_source_and_its_ancestor_directory(worker, simulation, tmp_path):
    parent = tmp_path / "dataset"
    parent.mkdir()
    scan = parent / "simulation.nxs"
    root = dict(type="member", handle=simulation["value"]["handle"], member="self")
    output(worker("run", operation=dict(type="save", source=root, path=str(scan))))
    before = scan.read_bytes()
    reopened = output(worker("open", kind="simulation", path=str(scan)))["value"]
    current = dict(type="member", handle=reopened["handle"], member="self")
    for command, path in [("save", scan), ("export", parent)]:
        for outer in ("validate", "run"):
            error = worker(
                outer, operation=dict(type=command, source=current, path=str(path), force=True)
            )["error"]
            assert (error["kind"], error["path"]) == ("invalid_parameter", "path")
    assert scan.read_bytes() == before


@pytest.mark.parametrize(("original", "copy"), [("npz", "nxs"), ("nxs", "npz")])
def test_standalone_volume_keeps_its_recorded_unit_when_reopened_and_saved(
    worker, tmp_path, original, copy
):
    from tomojax.io import ProjectionDataset, save_dataset

    grid = tj.Grid(4, 3, 2, 0.5, 2, 3, vol_origin=(11, -5, 7))
    pixels = np.arange(24, dtype=np.float32).reshape(grid.shape)
    path, destination = tmp_path / f"original.{original}", tmp_path / f"copy.{copy}"
    save_dataset(
        path,
        ProjectionDataset(
            projections=None,
            angles=np.array([0, 45, 90], np.float32),
            volume=pixels,
            detector=tj.Detector(4, 2, 0.5, 3),
            grid=grid,
            geometry_metadata={
                "acquisition": {"length_unit": "um"},
                "reconstruction_method": "fbp",
            },
        ),
    )
    for file in (path, destination):
        value = output(worker("open", kind="reconstruction", path=str(file)))["value"]
        assert value["scan"] is None and value["volume"]["unit"] == "1/um"
        assert [a["unit"] for a in value["volume"]["axes"]] == ["um"] * 3
        ref = dict(type="member", handle=value["handle"], member="volume")
        plane = output(worker("plane", source=ref, axis="z", index=1))["value"]["plane"]
        assert (plane["fixed"][0]["coordinate"], plane["fixed"][0]["unit"]) == (10, "um")
        np.testing.assert_array_equal(np.load(plane["storage"]["path"]), pixels[1])
        if file == path:
            root = dict(type="member", handle=value["handle"], member="self")
            output(worker("run", operation=dict(type="save", source=root, path=str(destination))))


def test_worker_publishes_and_releases_live_planes(worker, simulation):
    events = []
    scan = simulation["value"]["scan"]
    reply = output(
        worker(
            "run",
            events=events,
            operation=dict(type="reconstruct", source=scan, method="sirt", iterations=2),
            previews=True,
        )
    )
    previews = [e for e in events if e["event"] == "preview"]
    assert previews
    for preview in previews:
        for axis, array in preview["arrays"].items():
            storage = array["storage"]
            assert array["fixed"][0]["name"] == axis
            path = Path(storage["path"])
            assert np.load(path, allow_pickle=False).shape == tuple(array["shape"])
            output(worker("release", artifact=storage["artifact"]))
            assert not path.exists()
    assert reply["value"]["type"] == "reconstruction"


def test_slice_and_sweep_publish_fixed_input_registration(worker, tmp_path):
    from dataclasses import replace

    grid = tj.Grid(25, 25, 5, 1, 1, 1)
    det = tj.Detector(25, 5, 1, 1)
    geom = tj.ParallelGeometry(grid, det, np.arange(0, 180, 15))
    data = np.random.default_rng(42).uniform(0.2, 1, (12, 5, 25)).astype(np.float32)
    path = tmp_path / "center.nxs"
    tj.save(path, tj.Scan(data, geom))
    source = dict(type="path", path=str(path))
    request = dict(source=source, axis="z", index=2, method="fbp", region="bbox")
    offsets = [0, 3, 0]
    sweep = output(worker("center_sweep", **request, centers=offsets))["value"]["planes"]
    plane = output(worker("slice", **request))["value"]["plane"]
    assert (
        sweep["fixed"]
        == plane["fixed"]
        == [dict(name="z", frame="volume", unit="scan_length", index=2, coordinate=0)]
    )
    assert sweep["axes"][1:] == plane["axes"]
    assert [a["name"] for a in plane["axes"]] == ["y", "x"]
    assert all(
        a["coordinates"] == dict(type="uniform", origin=-12, spacing=1) for a in plane["axes"]
    )
    coord = sweep["axes"][0]
    assert (coord["name"], coord["unit"], coord["frame"]) == ("center", "pixel", "detector")
    np.testing.assert_array_equal(np.load(coord["coordinates"]["values"]["path"]), offsets)
    pixels = np.load(sweep["storage"]["path"])
    np.testing.assert_array_equal(pixels[0], np.load(plane["storage"]["path"]))
    fixed = tj.reconstruct(tj.load(path), "fbp", region="bbox").grid
    for i, center in enumerate(offsets):
        shifted = tj.Scan(data, replace(geom, detector=replace(det, center=(center, 0))))
        np.testing.assert_allclose(
            pixels[i], tj.reconstruct(shifted, "fbp", grid=fixed).volume[2], rtol=1e-6, atol=2e-7
        )
    for stored in (sweep["storage"], plane["storage"], coord["coordinates"]["values"]):
        output(worker("release", artifact=stored["artifact"]))
        assert not Path(stored["path"]).exists()


def test_correction_preview_publishes_native_and_binned_pixel_centers(worker, tmp_path):
    n, nv, nu = 13, 9, 11
    angles = np.arange(n, dtype=float) * 13
    detector = dict(nu=nu, nv=nv, du=0.75, dv=1.5, center=[0.6, -1.2])
    data = np.random.default_rng(629).integers(470, 540, (n, nv, nu), dtype=np.uint16)
    data[5, 6:] = 50
    path = tmp_path / "raw.h5"
    with h5py.File(path, "w") as file:
        file["exchange/data"] = data
        file["exchange/theta"] = angles
    root = output(
        worker(
            "open",
            kind="frames",
            path=str(path),
            detector=detector,
            geometry=dict(kind="parallel"),
            flats=1000,
            darks=0,
        )
    )["value"]["handle"]
    source = dict(type="member", handle=root, member="self")
    args = dict(source=source, rows=[1, 3], binning=2, corrections=[dict(type="RejectViews")])
    value = output(worker("correction_preview", **args))["value"]
    keep = np.delete(np.arange(n), 5)
    before, after, views = (value[k] for k in ("before", "after", "views"))
    np.testing.assert_array_equal(np.load(before["storage"]["path"]), data[:, 2:6])
    logs = -np.log(data[keep, 2:6, :10].astype("f") / 1000)
    np.testing.assert_allclose(
        np.load(after["storage"]["path"]), logs.reshape(12, 2, 2, 5, 2).mean(axis=(2, 4)), rtol=3e-7
    )
    np.testing.assert_array_equal(np.load(views["storage"]["path"]), keep)
    for descriptor, ids, pitch, starts in (
        (before, np.arange(n), (1.5, 0.75), (-4.2, -3.15)),
        (after, keep, (3, 1.5), (-3.45, -2.775)),
    ):
        view, v, u = descriptor["axes"]
        assert [view["name"], v["name"], u["name"]] == ["view", "v", "u"]
        np.testing.assert_array_equal(np.load(view["coordinates"]["values"]["path"]), angles[ids])
        for axis, step, start in zip((v, u), pitch, starts, strict=True):
            assert axis["frame"] == "detector"
            assert axis["coordinates"]["spacing"] == step
            assert axis["coordinates"]["origin"] == pytest.approx(start)
    assert views["axes"] == after["axes"][:1]


def test_region_overlays_can_drive_another_reconstruction_and_close(worker, tmp_path):
    grid = tj.Grid(11, 9, 5, 1, 1, 1)
    geom = tj.LaminographyGeometry(grid, tj.Detector(11, 9, 1, 1), np.arange(0, 360, 15), tilt=36)
    scan = tj.Scan(np.random.default_rng(106).uniform(0, 1, (24, 9, 11)).astype("f"), geom)
    path = tmp_path / "region.nxs"
    tj.save(path, scan)
    source = dict(type="path", path=str(path))
    region = output(worker("region_preview", source=source))
    value = region["value"]
    ref = dict(type="member", handle=value["handle"], member="self")
    result = output(
        worker("run", operation=dict(type="reconstruct", source=source, method="fbp", region=ref))
    )
    assert result["manifest"]["inputs"] == [region["manifest"]]
    full = tj.reconstruct(tj.load(path), "fbp", grid=tj.Grid(**value["grid"])).volume
    assert result["value"]["volume"]["shape"] == list(full.shape)
    for dim, axis in enumerate("zyx"):
        desc = value["arrays"][axis]
        (fixed,) = desc["fixed"]
        assert (fixed["name"], fixed["index"]) == (axis, full.shape[dim] // 2)
        np.testing.assert_array_equal(
            np.load(desc["storage"]["path"]), np.take(np.asarray(full), fixed["index"], axis=dim)
        )
    output(worker("close", handle=value["handle"]))
    assert all(not Path(a["storage"]["path"]).exists() for a in value["arrays"].values())


def test_forced_stack_export_refuses_an_unowned_directory(worker, simulation, tmp_path):
    stack = tmp_path / "stack"
    stack.mkdir()
    sentinel = stack / "acquisition.txt"
    sentinel.write_text("original")
    root = dict(type="member", handle=simulation["value"]["handle"], member="self")
    reply = worker("run", operation=dict(type="export", source=root, path=str(stack), force=True))
    assert reply["type"] == "error"
    assert "export record" in reply["error"]["message"]
    assert list(stack.iterdir()) == [sentinel]
    assert sentinel.read_text() == "original"


def test_cone_cgls_interpolation_preflight(worker, simulation):
    cone = output(
        worker(
            "run",
            operation=dict(type="simulate", geometry="cone", size=6, n_views=12, rotation=360),
        )
    )
    for scan, interpolation, accepted in (
        (cone, "cubic", False),
        (cone, "linear", True),
        (simulation, "cubic", True),
    ):
        recipe = dict(
            type="reconstruct",
            source=scan["value"]["scan"],
            method="cgls",
            config=dict(
                type="CGLSConfig", backend="jax", interpolation=interpolation, iterations=0
            ),
        )
        for command in ("validate", "plan", "run"):
            reply = worker(command, operation=recipe)
            if accepted:
                assert reply["type"] == "result", reply
            else:
                assert reply["type"] == "error", reply
                assert (reply["error"]["kind"], reply["error"]["path"]) == (
                    "invalid_parameter",
                    "config.interpolation",
                )


@pytest.mark.parametrize(("method", "kind"), [("cgls", "CGLSConfig"), ("fista", "FISTAConfig")])
@pytest.mark.parametrize("nested_devices", [True, False])
def test_stream_device_conflict_has_the_same_field_in_preflight_and_run(
    worker, simulation, method, kind, nested_devices
):
    device = output(worker("devices"))["devices"][0]["id"]
    config = dict(type=kind, stream=True, iterations=0)
    recipe = dict(
        type="reconstruct", source=simulation["value"]["scan"], method=method, config=config
    )
    (config if nested_devices else recipe)["devices"] = [device]
    for command in ("validate", "plan", "run"):
        reply = worker(command, operation=recipe)
        assert reply["type"] == "error", reply
        assert (reply["error"]["kind"], reply["error"]["path"]) == (
            "invalid_parameter",
            "config.stream",
        )


@pytest.mark.parametrize(("method", "kind"), [("fbp", "FBPConfig"), ("sirt", "SIRTConfig")])
@pytest.mark.parametrize("geometry", ["parallel", "cone"])
def test_unavailable_cuda_backend_is_rejected_in_preflight(
    worker, simulation, method, kind, geometry
):
    if any(device["platform"] == "gpu" for device in output(worker("devices"))["devices"]):
        pytest.skip("requires CPU placement")
    scan = (
        simulation
        if geometry == "parallel"
        else output(
            worker(
                "run",
                operation=dict(type="simulate", geometry="cone", size=6, n_views=12, rotation=360),
            )
        )
    )
    recipe = dict(
        type="reconstruct",
        source=scan["value"]["scan"],
        method=method,
        config=dict(type=kind, backend="cuda"),
    )
    for command in ("validate", "plan", "run"):
        reply = worker(command, operation=recipe)
        assert reply["type"] == "error", reply
        assert (reply["error"]["kind"], reply["error"]["path"]) == (
            "invalid_parameter",
            "config.backend",
        )


def test_mlem_execution_checks_the_measurement_domain(worker, tmp_path):
    path = tmp_path / "negative.nxs"
    grid = tj.Grid(3, 3, 2, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(4, 2, 1, 1), [0, 43, 91, 137])
    measured = np.ones((4, 2, 4), np.float32)
    measured[-1, -1, -1] = -0.125
    tj.save(path, tj.Scan(measured, geometry))
    recipe = dict(type="reconstruct", method="mlem", source=dict(type="path", path=str(path)))
    for command in ("validate", "plan"):
        assert worker(command, operation=recipe)["type"] == "result"
    reply = worker("run", operation=recipe)
    assert reply["type"] == "error", reply
    assert (reply["error"]["kind"], reply["error"]["path"]) == ("invalid_parameter", "source")


def test_spdhg_seed_is_nonnegative_in_all_commands(worker, simulation):
    recipe = dict(
        type="reconstruct",
        method="spdhg",
        source=simulation["value"]["scan"],
        config=dict(type="SPDHGConfig", seed=-1, iterations=1),
    )
    for command in ("validate", "plan", "run"):
        reply = worker(command, operation=recipe)
        assert reply["type"] == "error", reply
        assert (reply["error"]["kind"], reply["error"]["path"]) == (
            "invalid_parameter",
            "config.seed",
        )


@pytest.mark.parametrize("bad_row", [2, 0])
def test_fourier_execution_checks_only_sampled_detector_rows(worker, tmp_path, bad_row):
    grid = tj.Grid(5, 5, 1, 1, 1, 1, vol_center=(0, 0, 0.25))
    geometry = tj.ParallelGeometry(grid, tj.Detector(7, 5, 1, 1), [12, 45, 90, 163])
    data = np.ones((4, 5, 7), np.float32)
    data[1, bad_row, 3] = np.nan
    path = tmp_path / "fractional-row.nxs"
    tj.save(path, tj.Scan(data, geometry))
    recipe = dict(type="reconstruct", method="fourier", source=dict(type="path", path=str(path)))
    for command in ("validate", "plan", "run"):
        reply = worker(command, operation=recipe)
        # z=0.25 lies one quarter of the way from row 2 to row 3.
        if command != "run" or bad_row == 0:
            assert reply["type"] == "result", reply
        else:
            assert reply["type"] == "error", reply
            assert (reply["error"]["kind"], reply["error"]["path"]) == (
                "invalid_parameter",
                "source",
            )


def test_fourier_dry_run_rejects_complex_projection_layout(worker, tmp_path):
    geometry = tj.ParallelGeometry(
        tj.Grid(4, 4, 2, 1, 1, 1), tj.Detector(4, 2, 1, 1), [0, 45, 90, 135]
    )
    path = tmp_path / "complex.npz"
    tj.save(path, tj.Scan(np.ones((4, 2, 4), np.complex64), geometry))
    operation = dict(type="reconstruct", method="fourier", source=dict(type="path", path=str(path)))
    for command in ("validate", "plan", "run"):
        error = worker(command, operation=operation)["error"]
        assert (error["kind"], error["path"]) == ("invalid_parameter", "source")
        assert "real NumPy array" in error["message"]


@pytest.mark.parametrize("method", ["cgls", "fista", "spdhg", "sirt"])
def test_iterative_execution_requires_finite_measurements(worker, tmp_path, method):
    grid = tj.Grid(4, 4, 2, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(4, 2, 1, 1), [0, 45, 90, 135])
    measured = np.zeros((4, 2, 4), np.float32)
    measured[2, 0, 1] = np.inf
    path = tmp_path / "nonfinite.nxs"
    tj.save(path, tj.Scan(measured, geometry))
    recipe = dict(type="reconstruct", method=method, source=dict(type="path", path=str(path)))
    for command in ("validate", "plan"):
        assert worker(command, operation=recipe)["type"] == "result"
    reply = worker("run", operation=recipe)
    assert reply["type"] == "error", reply
    assert (reply["error"]["kind"], reply["error"]["path"]) == ("invalid_parameter", "source")


@pytest.mark.parametrize(("method", "field"), [("fbp", "method"), ("cgls", "warm_start")])
def test_full_turn_fdk_requires_axis_on_detector(worker, tmp_path, method, field):
    grid = tj.Grid(4, 4, 2, 1, 1, 1)
    geometry = tj.ConeGeometry(
        grid, tj.Detector(7, 5, 1, 1, (20, 0)), np.arange(0, 360, 30), tj.ConeBeam(50, 80)
    )
    path = tmp_path / "offset.nxs"
    tj.save(path, tj.Scan(np.ones((12, 5, 7), np.float32), geometry))
    recipe = dict(type="reconstruct", method=method, source=dict(type="path", path=str(path)))
    if method == "cgls":
        recipe["warm_start"] = True
    for command in ("validate", "plan", "run"):
        reply = worker(command, operation=recipe)
        assert reply["type"] == "error", reply
        assert (reply["error"]["kind"], reply["error"]["path"]) == ("invalid_parameter", field)


@pytest.mark.parametrize(("original", "centers", "invalid"), [(0, [0, 20], 1), (20, [0], None)])
def test_fdk_sweep_validates_each_requested_detector_center(
    worker, tmp_path, original, centers, invalid
):
    from dataclasses import replace

    grid = tj.Grid(4, 4, 2, 1, 1, 1)
    detector = tj.Detector(7, 5, 1, 1, (original, 0))
    geometry = tj.ConeGeometry(grid, detector, np.arange(0, 360, 30), tj.ConeBeam(50, 80))
    data = np.random.default_rng(17).uniform(size=(12, 5, 7)).astype(np.float32)
    path = tmp_path / "full-turn.nxs"
    tj.save(path, tj.Scan(data, geometry))
    args = dict(
        source=dict(type="path", path=str(path)), method="fbp", axis="z", index=0, centers=centers
    )
    for command in ("validate", "center_sweep"):
        reply = (
            worker(command, operation=dict(type="center_sweep", **args))
            if command == "validate"
            else worker(command, **args)
        )
        if invalid is not None:
            assert reply["type"] == "error", reply
            assert (reply["error"]["kind"], reply["error"]["path"]) == (
                "invalid_parameter",
                f"centers[{invalid}]",
            )
        elif command == "validate":
            assert reply["type"] == "result", reply
        else:
            pixels = np.load(output(reply)["value"]["planes"]["storage"]["path"])
            shifted = tj.Scan(data, replace(geometry, detector=replace(detector, center=(0, 0))))
            np.testing.assert_allclose(
                pixels[0], tj.reconstruct(shifted, "fbp").volume[0], atol=1e-6
            )


def test_explicit_virtual_dataset_rejects_missing_detector_block(worker, tmp_path):
    shape = (12, 2, 8)
    grid = tj.Grid(8, 8, 2, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(8, 2, 1, 1), np.arange(12) * 15)
    scan = tmp_path / "scan.npz"
    tj.save(scan, tj.Scan(np.ones(shape, np.float32), geometry))
    layout = h5py.VirtualLayout(shape, np.float32)
    for first in (0, 6):
        part = tmp_path / f"part{first}.h5"
        with h5py.File(part, "w") as file:
            file["pixels"] = np.ones((6, 2, 8), np.float32)
            layout[first : first + 6] = h5py.VirtualSource(file["pixels"])
    path = tmp_path / "detector.h5"
    with h5py.File(path, "w") as file:
        file.create_virtual_dataset("pixels", layout)
    (tmp_path / "part6.h5").unlink()
    source = dict(type="hdf5", path=str(path), dataset="/pixels")
    operation = dict(
        type="backproject", source=dict(type="path", path=str(scan)), projections=source
    )
    for reply in (worker("statistics", source=source), worker("run", operation=operation)):
        assert reply["type"] == "error", reply
        assert reply["error"]["kind"] == "input"
        assert "part6.h5" in reply["error"]["message"]


def test_hyperslab_limit_stride_and_exact_statistics(worker, tmp_path):
    path = tmp_path / "stack.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("large", shape=(2049, 2049), dtype="f4", fillvalue=1)
        x = np.arange(72, dtype="f4").reshape(6, 3, 4)
        x[1, 1, 2] = np.nan
        x[3, 2, 0] = np.inf
        f["small"] = x
    source = dict(type="hdf5", path=str(path), dataset="/large")
    refused = worker("read", source=source)["error"]
    assert refused["path"] == "roi" and "16 MiB" in refused["message"]
    a = output(worker("read", source=source, stride=[16, 8]))["value"]["array"]
    assert np.load(a["storage"]["path"]).shape == (129, 257)
    assert a["axes"][1]["coordinates"]["spacing"] == 8
    source["dataset"] = "/small"
    params = dict(source=source, roi=[[1, 6], [1, 3], [0, 4]], stride=[2, 1, 2])
    slab = x[1:6:2, 1:3, ::2]
    stats = output(worker("statistics", **params, percentiles=[0, 25, 100], bins=3))["value"]
    finite = slab[np.isfinite(slab)]
    assert stats["non_finite"] == 2 and stats["count"] == finite.size
    assert stats["mean"] == pytest.approx(finite.mean())
    assert stats["std"] == pytest.approx(finite.std())
    np.testing.assert_allclose(
        list(stats["percentiles"].values()), np.percentile(finite, [0, 25, 100])
    )
    assert stats["histogram"] == np.histogram(finite, bins=3)[0].tolist()
    first = output(worker("read", **params))["value"]["array"]
    np.testing.assert_equal(np.load(first["storage"]["path"]), slab)
    again = output(worker("read", source=first["storage"], roi=[[1, 2], [0, 2], [0, 2]]))["value"][
        "array"
    ]
    assert again["axes"][0]["coordinates"] == dict(type="uniform", origin=3.0, spacing=2.0)
    np.testing.assert_equal(np.load(again["storage"]["path"]), slab[1:2])


@pytest.mark.parametrize("ext", ["nxs", "npz"])
def test_saved_volume_registered_reads(worker, tmp_path, ext):
    sim = tj.simulate(size=8, n_views=12)
    path = tmp_path / f"result.{ext}"
    tj.save(path, tj.Reconstruction(sim.phantom, sim.scan.geometry, "example", {}, sim.scan))
    source = dict(type="path", path=str(path))
    a = output(worker("read", source=source, stride=[2, 1, 2]))["value"]["array"]
    assert [axis["name"] for axis in a["axes"]] == list("zyx")
    np.testing.assert_array_equal(np.load(a["storage"]["path"]), sim.phantom[::2, :, ::2])
    view = output(worker("read", source=source, mode="planes", axis="x", index=2))["value"]
    plane = view["planes"][0]["array"]
    np.testing.assert_array_equal(np.load(plane["storage"]["path"]), sim.phantom[:, :, 2:3])


@pytest.mark.parametrize("ext", ["nxs", "npz"])
def test_acquisition_array_paths_return_open_guidance(worker, simulation, tmp_path, ext):
    path = tmp_path / f"scan.{ext}"
    tj.save(path, tj.simulate(size=8, n_views=12).scan)
    source = dict(type="path", path=str(path))
    requests = [(op, dict(source=source), "source") for op in ("read", "statistics")]
    for op, field in (("project", "volume"), ("backproject", "projections")):
        recipe = dict(type=op, source=simulation["value"]["scan"], **{field: source})
        requests.extend((cmd, dict(operation=recipe), field) for cmd in ("validate", "run"))
    for cmd, args, field in requests:
        error = worker(cmd, **args)["error"]
        assert (error["kind"], error["path"]) == ("invalid_parameter", field)
        for hint in (
            "open",
            ".projections",
            "kind=frames",
            ".counts",
            "correction_preview",
            "inspect",
            "preview=true",
        ):
            assert hint in error["message"]


@pytest.mark.parametrize("axes", ["xyz", "yzx"])
def test_saved_volume_axes_in_operators_and_roi(worker, tmp_path, axes):
    sim = tj.simulate(size=8, n_views=8)
    x = np.arange(8**3, dtype=np.float32).reshape((8,) * 3) / 64
    path = tmp_path / "asymmetric.nxs"
    tj.save(path, tj.Reconstruction(x, sim.scan.geometry, "example", {}, sim.scan))
    with h5py.File(path, "r+") as file:
        group = file["/entry/processing/tomojax"]
        group["volume"][:] = x.transpose(tuple("zyx".index(a) for a in axes))
        group.attrs["volume_axes_order"] = axes
        ref = dict(type="hdf5", path=str(path), dataset=group["volume"].name)
    source = dict(type="path", path=str(tmp_path / "scan.nxs"))
    tj.save(source["path"], sim.scan)
    disk_pixels = x.transpose(tuple("zyx".index(a) for a in axes))
    # A volume path requests logical zyx order. A registered dataset retains its disk axes.
    projected = output(
        worker(
            "run",
            operation=dict(type="project", source=source, volume=dict(type="path", path=str(path))),
        )
    )
    np.testing.assert_allclose(
        np.load(projected["value"]["projections"]["storage"]["path"]),
        tj.project(sim.scan.geometry, x),
        atol=1e-5,
    )
    for operation, field in (("project", "volume"), ("backproject", "projections")):
        reply = worker("run", operation={"type": operation, "source": source, field: ref})
        assert (
            reply["error"]["path"] == field and "axes in storage order" in reply["error"]["message"]
        )
    args = dict(
        source=dict(type="path", path=str(path)), roi=[[2, 4], [0, 7], [3, 4]], stride=[1, 2, 1]
    )
    read = output(worker("read", **args))["value"]["array"]
    assert [axis["name"] for axis in read["axes"]] == list("zyx")
    np.testing.assert_equal(np.load(read["storage"]["path"]), x[2:4, :7:2, 3:4])
    assert output(worker("statistics", **args))["value"]["mean"] == pytest.approx(
        x[2:4, :7:2, 3:4].mean()
    )

    # A literal dataset remains readable in stored order without TomoJAX registration.
    with h5py.File(path, "r+") as file:
        del file["/entry/processing/tomojax"].attrs["volume_axes_order"]
    literal = output(worker("read", source=ref))["value"]["array"]
    np.testing.assert_array_equal(np.load(literal["storage"]["path"]), disk_pixels)


def read_frames(stack, shape, roi, stride, tmp_path):
    from tests.worker_adapter import Stage, admitted
    from tomojax.cli import execution, operations
    from tomojax.wire import arrays

    ref = arrays.Member("frames", "counts")
    axes = tuple(arrays.Axis(k, "array", "index", arrays.Uniform(0, 1)) for k in ("view", "v", "u"))
    meta = arrays.Array(ref, shape, stack.dtype.str, axes, "counts", "counts")
    stage = Stage(tmp_path)
    admitted(
        operations.Read(source=ref, roi=roi, stride=stride),
        execution.RegisteredArray(stack, meta),
        stage,
        ref,
    )
    return np.load(stage.private)


def test_strided_lazy_frames_read_only_sampled_hdf5_pixels(tmp_path, monkeypatch):
    from tomojax.io.api import locate_frames

    path = tmp_path / "frames.h5"
    data = np.arange(15 * 64 * 64, dtype=np.uint16).reshape(15, 64, 64)
    with h5py.File(path, "w") as f:
        f["/entry/instrument/detector/data"] = data
    stack = (
        locate_frames(str(path)).stack.window(slice(2, 61), slice(1, 60)).frames_at(slice(1, 15, 2))
    )
    reads = []
    original = h5py.Dataset.__getitem__
    original_direct = h5py.Dataset.read_direct

    def record(dataset, key):
        block = original(dataset, key)
        reads.append(block.size)
        return block

    def record_direct(dataset, dest, source_sel=None, dest_sel=None):
        original_direct(dataset, dest, source_sel, dest_sel)
        reads.append(dest.size if dest_sel is None else dest[dest_sel].size)

    monkeypatch.setattr(h5py.Dataset, "read_direct", record_direct)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", record)
    got = read_frames(stack, stack.shape, ((0, 7), (3, 59), (2, 59)), (1, 8, 4), tmp_path)
    np.testing.assert_array_equal(got, data[1:15:2, 5:61:8, 3:60:4])
    assert sum(reads) == got.size


def test_strided_window_reader_bounds_unsampled_work(tmp_path):
    data = np.arange(12 * 711 * 2097, dtype=np.uint16).reshape(12, 711, 2097)
    budget = 16 << 20

    class Frames:
        dtype = data.dtype

        def window(self, rows, cols):
            class Window:
                def __getitem__(self, key):
                    block = data[key, rows, cols]
                    assert block.nbytes <= budget
                    return block.copy()

            return Window()

    result = read_frames(Frames(), data.shape, ((1, 12), (1, 711), (3, 2097)), (2, 8, 5), tmp_path)
    np.testing.assert_array_equal(result, data[1::2, 1::8, 3::5])


@pytest.mark.parametrize(("kind", "member"), [("scan", "projections"), ("frames", "counts")])
def test_opened_detector_member_reads_bounded_tiles(worker, tmp_path, kind, member):
    path = tmp_path / "raw.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("exchange/data", (17, 512, 512), dtype="f4", fillvalue=5)
        file["exchange/theta"] = np.arange(17) * 10.0
    opened = output(
        worker(
            "open",
            path=str(path),
            kind=kind,
            flats=11,
            darks=1,
            pixel_size=1,
            geometry={"kind": "parallel"},
        )
    )["value"]
    source = dict(type="member", handle=opened["handle"], member=member)
    assert "16 MiB" in worker("read", source=source)["error"]["message"]
    tile = dict(source=source, roi=[[4, 5], [1, 512], [3, 511]], stride=[1, 7, 4])
    plane = output(worker("read", mode="projection", **tile))["value"]["planes"][0]["array"]
    expected = -np.log(0.4) if kind == "scan" else 5
    pixels = np.load(plane["storage"]["path"])
    assert pixels.shape == (1, 73, 127)
    np.testing.assert_allclose(pixels, expected, rtol=1e-6)
    stats = output(worker("statistics", **tile))["value"]
    assert stats["count"] == pixels.size
    assert stats["mean"] == pytest.approx(expected)


def trace_detector_reads(monkeypatch, where):
    reads = []
    original, direct = h5py.Dataset.__getitem__, h5py.Dataset.read_direct

    def get(data, key):
        result = original(data, key)
        if data.name == where:
            reads.append(result.nbytes)
        return result

    def read(data, dest, source_sel=None, dest_sel=None):
        direct(data, dest, source_sel, dest_sel)
        if data.name == where:
            reads.append(dest[dest_sel or (...,)].nbytes)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", get)
    monkeypatch.setattr(h5py.Dataset, "read_direct", read)
    return reads


@pytest.mark.parametrize("ext", ["nxs", "npz"])
def test_project_saved_volume_skips_archived_scan(tmp_path, monkeypatch, ext):
    from tests.worker_adapter import Stage, admitted
    from tomojax.cli import operations
    from tomojax.wire.arrays import Member

    sim = tj.simulate(size=8, n_views=12)
    path = tmp_path / f"archived.{ext}"
    recon = tj.Reconstruction(sim.phantom, sim.scan.geometry, "example", {}, sim.scan)
    tj.save(path, recon, include_scan=True)
    ref = Member("opened", "scan")
    stage = Stage(tmp_path)
    reads = (
        trace_detector_reads(monkeypatch, "/entry/instrument/detector/data")
        if ext == "nxs"
        else count_inflated_bytes(monkeypatch, "projections.npy")
    )
    project = operations.Project(source=ref, volume=operations.PathInput(str(path)))
    admitted(operations.Run(operation=project), sim.scan, stage, ref)
    np.testing.assert_allclose(np.load(stage.private), tj.project(sim.scan.geometry, sim.phantom))
    assert reads == []


@pytest.mark.parametrize("compressed", [False, True])
def test_npz_reads_fortran_order_volume_past_tile_budget(tmp_path, compressed):
    from tests.worker_adapter import Stage, admitted
    from tomojax.cli import operations

    path = tmp_path / "wide.npz"
    shape = (2, 512, 4097)
    data = np.zeros(shape, dtype=np.float32, order="F")
    data[1, -1, -2:] = (8, 12)
    write = np.savez_compressed if compressed else np.savez
    write(path, volume=data, volume_axes_order="zyx")
    ref = operations.PathInput(str(path))
    roi = ((1, 2), (511, 512), (4095, 4097))
    stage = Stage(tmp_path)
    request = operations.Read(source=ref, roi=roi)
    admitted(request, None, stage, ref)
    np.testing.assert_equal(np.load(stage.private).ravel(), [8, 12])
    result = admitted(operations.Statistics(source=ref, roi=roi), None, stage, ref)
    assert result.mean == 10


def test_validate_open_only_reads_calibration_metadata(tmp_path, monkeypatch):
    from tests.test_io_workflows import _labeled_dataset
    from tests.worker_adapter import Stage, admitted
    from tomojax.cli import operations
    from tomojax.io import save_dataset

    path = tmp_path / "raw.nxs"
    save_dataset(path, _labeled_dataset([4, 8, 2], [0, 1, 2]))
    where = "/entry/instrument/detector/data"
    reads = trace_detector_reads(monkeypatch, where)
    events = []
    validate = operations.Validate(operation=operations.Open(path=str(path), kind="scan"))

    def run():
        return admitted(validate, None, Stage(tmp_path), None, event=events.append)

    assert run().details["geometry"]["views"] == 1
    assert reads == []
    assert not any(getattr(e, "name", None) in ("load", "correction") for e in events)
    save_dataset(path, _labeled_dataset([4, 2], [0, 2]))
    with pytest.raises(ValueError, match="No flat"):
        run()
    path.unlink()
    with pytest.raises(FileNotFoundError):
        run()
    assert reads == []


@pytest.mark.parametrize("kind", ["scan", "simulation", "reconstruction"])
def test_validate_open_skips_saved_detector_and_volume(tmp_path, monkeypatch, kind):
    from tests.worker_adapter import Stage, admitted
    from tomojax.cli import operations

    sim = tj.simulate(size=8, n_views=12)
    path = tmp_path / "saved.nxs"
    if kind == "reconstruction":
        recon = tj.Reconstruction(sim.phantom, sim.scan.geometry, "example", {}, sim.scan)
        tj.save(path, recon, include_scan=True)
    else:
        tj.save(path, sim)
    frames = trace_detector_reads(monkeypatch, "/entry/instrument/detector/data")
    volume = trace_detector_reads(monkeypatch, "/entry/processing/tomojax/volume")
    result = admitted(
        operations.Validate(operation=operations.Open(path=str(path), kind=kind)),
        None,
        Stage(tmp_path),
        None,
    )
    assert result.details["geometry"]["views"] == 12
    assert frames == volume == []


def test_inspect_saved_volume_reads_no_volume_pixels(tmp_path, monkeypatch):
    sim = tj.simulate(size=8, n_views=12)
    path = tmp_path / "volume.nxs"
    tj.save(path, tj.Reconstruction(sim.phantom, sim.scan.geometry, "example", {}, sim.scan))
    reads = trace_detector_reads(monkeypatch, "/entry/processing/tomojax/volume")
    report = tj.inspect(path)
    assert report["valid"] and report["volume"]["shape"] == [8, 8, 8]
    assert not report["projection"]["found"]
    assert reads == []


def count_inflated_bytes(monkeypatch, member):
    from zipfile import ZipExtFile

    sizes = []
    original = ZipExtFile._read1

    def count(stream, n):
        block = original(stream, n)
        if stream.name == member:
            sizes.append(len(block))
        return block

    monkeypatch.setattr(ZipExtFile, "_read1", count)
    return sizes


@pytest.mark.parametrize("order", ["C", "F"])
def test_compressed_npz_windows_inflate_member_once(tmp_path, monkeypatch, order):
    from zipfile import ZipFile

    from tests.worker_adapter import Stage, admitted
    from tomojax.cli import operations

    data = np.array(
        np.arange(32 * 256 * 1024, dtype="f4").reshape(32, 256, 1024) % 251, order=order
    )
    data[1, 1, 1] = np.nan
    path = tmp_path / "volume.npz"
    np.savez_compressed(path, volume=data, volume_axes_order="zyx")
    with ZipFile(path) as archive:
        size = archive.getinfo("volume.npy").file_size
    inflated = count_inflated_bytes(monkeypatch, "volume.npy")
    ref = operations.PathInput(str(path))
    roi = ((1, 32), (1, 256), (1, 1024))
    request = operations.Statistics(
        source=ref, roi=roi, stride=(1, 1, 2), percentiles=(25, 90), bins=7
    )
    stats = admitted(request, None, Stage(tmp_path), ref)
    assert 0 < sum(inflated) <= size and max(inflated) <= 16 << 20
    selected = data[1:, 1:, 1::2]
    finite = selected[np.isfinite(selected)].astype(np.float64)
    assert stats.count == finite.size and stats.non_finite == 1
    assert stats.mean == pytest.approx(finite.mean())
    assert stats.std == pytest.approx(finite.std())
    np.testing.assert_allclose(list(stats.percentiles.values()), np.percentile(finite, (25, 90)))
    counts, edges = np.histogram(finite, bins=7)
    np.testing.assert_array_equal(stats.histogram, counts)
    np.testing.assert_allclose(stats.edges, edges)

    stage = Stage(tmp_path)
    inflated.clear()
    admitted(operations.Read(source=ref, roi=roi, stride=(1, 1, 2)), None, stage, ref)
    assert 0 < sum(inflated) <= size
    np.testing.assert_array_equal(np.load(stage.private), selected)

    files = []

    class PlaneStage(Stage):
        def array(self):
            self.private = tmp_path / f"plane-{len(files)}.npy"
            files.append(self.private)
            return super().array()

    inflated.clear()
    admitted(operations.Read(source=ref, mode="planes"), None, PlaneStage(tmp_path), ref)
    assert 0 < sum(inflated) <= size
    expected = (data[16:17, :, ::2], data[:, 128:129, ::2], data[:, :, 512:513])
    for file, plane in zip(files, expected, strict=True):
        np.testing.assert_array_equal(np.load(file), plane)


@pytest.mark.parametrize("forward", [True, False], ids=["project", "backproject"])
def test_projection_dry_run_reads_only_layout(tmp_path, monkeypatch, forward):
    from tests.test_io_workflows import _labeled_dataset
    from tests.worker_adapter import Stage, admitted
    from tomojax.cli import operations as op
    from tomojax.io import save_dataset
    from tomojax.wire.arrays import Hdf5

    path = tmp_path / "raw.nxs"
    save_dataset(path, _labeled_dataset([4, 8, 2], [0, 1, 2]))
    with h5py.File(path, "a") as file:
        file.create_dataset("operand", shape=(2, 4, 4) if forward else (1, 2, 4), dtype="f")
    geometry_reads = trace_detector_reads(monkeypatch, "/entry/instrument/detector/data")
    operand_reads = trace_detector_reads(monkeypatch, "/operand")
    ref, operand = op.PathInput(str(path)), Hdf5(str(path), "/operand")
    operation = op.Project(ref, operand) if forward else op.Backproject(ref, operand)
    events = []
    admitted(op.Validate(operation), None, Stage(tmp_path), None, event=events.append)
    assert not geometry_reads and not operand_reads
    assert not any(getattr(e, "name", None) in ("load", "correction") for e in events)


def test_open_dry_run_omits_unread_angle_range(tmp_path, monkeypatch):
    from tests.test_io_workflows import _labeled_dataset
    from tests.worker_adapter import Stage, admitted
    from tomojax.cli import operations as op
    from tomojax.io import save_dataset
    from tomojax.wire.arrays import Hdf5

    path = tmp_path / "raw.nxs"
    save_dataset(path, _labeled_dataset([4, 8, 2], [0, 1, 2]))
    with h5py.File(path, "a") as file:
        file["override"] = [37.0]
    reads = trace_detector_reads(monkeypatch, "/override")
    command = op.Open(str(path), "scan", angles=Hdf5(str(path), "/override"))
    result = admitted(op.Validate(command), None, Stage(tmp_path), None)
    assert result.details["geometry"]["views"] == 1
    assert "angle_range_degree" not in result.details["geometry"]
    assert not reads


def test_operator_path_source_uses_its_geometry_without_loading_pixels(tmp_path, monkeypatch):
    from tests.worker_adapter import Stage, admitted
    from tomojax.cli import operations

    # A reconstruction archived with its aligned scan, used as both source and volume.
    sim = tj.simulate(size=8, n_views=10)
    poses = np.zeros((10, 6), np.float32)
    poses[:, 0] = np.linspace(-0.4, 0.4, 10)
    poses[:, 3] = np.linspace(-0.6, 0.6, 10)
    scan = sim.scan.with_poses(poses)
    path = tmp_path / "aligned.nxs"
    volume = np.asarray(sim.phantom, np.float32)
    tj.save(path, tj.Reconstruction(volume, scan.geometry, "example", {}, scan), include_scan=True)
    expected = tj.project(tj.load(path).geometry, volume)
    reads = trace_detector_reads(monkeypatch, "/entry/instrument/detector/data")
    ref = operations.PathInput(str(path))
    stage = Stage(tmp_path)
    admitted(operations.Run(operation=operations.Project(source=ref, volume=ref)), None, stage, ref)
    assert reads == []
    np.testing.assert_allclose(np.load(stage.private), expected, atol=1e-5)


@pytest.mark.parametrize("binning", [1, 2])
@pytest.mark.parametrize("by", ["path", "handle"])
def test_correction_preview_of_a_scan_labels_its_input_as_line_integrals(
    worker, tmp_path, binning, by
):
    from tomojax.corrections import Rings

    grid = tj.Grid(8, 8, 6, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(8, 6, 1, 1), np.arange(12) * 15.0)
    data = np.random.default_rng(3).uniform(0, 1, (12, 6, 8)).astype(np.float32)
    path = tmp_path / "scan.nxs"
    tj.save(path, tj.Scan(data, geometry))
    if by == "path":
        source = dict(type="path", path=str(path))
    else:
        opened = output(worker("open", kind="scan", path=str(path)))["value"]["handle"]
        source = dict(type="member", handle=opened, member="self")
    args = dict(source=source, rows=[1, 2], binning=binning, corrections=[dict(type="Rings")])
    value = output(worker("correction_preview", **args))["value"]
    before, after = value["before"], value["after"]
    assert (before["domain"], before["unit"]) == ("line_integrals", "1")
    expected = tj.Scan(data, geometry).corrected(Rings()).binned(binning).projections[:, 1:2]
    np.testing.assert_allclose(np.load(after["storage"]["path"]), expected, rtol=1e-6, atol=1e-6)


def test_registered_inputs_use_the_resolved_grid(worker):
    made = output(
        worker(
            "python",
            code="""
def scan(dx, angles):
    grid = tj.Grid(7, 7, 7, 1., 1., 1., vol_origin=(dx, 0., 0.))
    geom = tj.ParallelGeometry(grid, tj.Detector(7,7,1,1), angles)
    return tj.Scan(np.ones((len(angles),7,7), np.float32), geom)

angles = np.arange(12) * 15
first = scan(0, angles)
other = scan(100, angles + (np.arange(12) == 1))
volume = tj.Reconstruction(np.ones((7,7,7), np.float32), first.geometry, "fbp", {})
[register(first), register(other), register_feature(np.ones((7,7,7), bool), like=volume)]
""",
        )
    )["value"]
    first, other, feature = made["value"]

    def ref(handle, member):
        return dict(type="member", handle=handle, member=member)

    for target, valid in ((first, True), (other, False)):
        operations = [
            (dict(type="project", volume=ref(feature, "mask")), "origin"),
            (dict(type="backproject", projections=ref(first, "projections")), "coordinate[1]"),
            (
                dict(
                    type="reconstruct",
                    method="sirt",
                    config=dict(type="SIRTConfig", support=ref(feature, "mask")),
                ),
                "origin",
            ),
            (
                dict(
                    type="evaluate",
                    candidates=[
                        dict(
                            type="reconstruct",
                            method="sirt",
                            config=dict(type="SIRTConfig", support=ref(feature, "mask")),
                        )
                    ],
                ),
                "origin",
            ),
            (
                dict(
                    type="feature",
                    source=ref(target, "projections"),
                    recipe=dict(type="band", feature=ref(feature, "self"), outer=1),
                ),
                "axes",
            ),
        ]
        for params, reason in operations:
            reply = worker("validate", operation={"source": ref(target, "self"), **params})
            if valid and params["type"] != "feature":
                assert reply["type"] == "result", reply
            else:
                err = reply["error"]
                assert err["kind"] == "invalid_parameter" and reason in err["message"], err
