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
        raw.with_suffix(".json"),
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
                corrections=[dict(type="Stripes"), dict(type="Zingers", threshold=-1)],
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
        config=dict(type="FBPConfig", backend="jax"),
    )
    result = output(worker("run", operation=recipe))["value"]["paths"]
    saved = tj.load_reconstruction(result[0])
    assert saved.grid.vol_origin == (-5.0, -5.5, -2.0)
    assert saved.info["source"] == str(raw)
    actual = saved.volume
    frames = tj.load_frames(raw, detector=tj.Detector(13, 5, 1, 1), flats=1000, darks=10)
    from tomojax.recon import FBPConfig

    expected = tj.reconstruct(
        frames.corrected(), "fbp", grid=tj.Grid(**recipe["grid"]), config=FBPConfig(backend="jax")
    )
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
    # An opposite-view centre fit is defined for axial parallel scans.
    parallel = output(worker("run", operation={**recipe, "geometry": "parallel", "rotation": 360}))
    fit = dict(
        type="align", source=parallel["value"]["scan"], fit="center", method="opposite-views"
    )
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
        (dict(type="simulate", poisson_scale=2, artefacts={"poisson_scale": 3}), "poisson_scale"),
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
    path = tmp_path / "centre.nxs"
    tj.save(path, tj.Scan(data, geom))
    source = dict(type="path", path=str(path))
    request = dict(source=source, axis="z", index=2, method="fbp", region="bbox")
    offsets = [0, 3, 0]
    sweep = output(worker("centre_sweep", **request, centres=offsets))["value"]["planes"]
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
    assert (coord["name"], coord["unit"], coord["frame"]) == ("centre", "pixel", "detector")
    np.testing.assert_array_equal(np.load(coord["coordinates"]["values"]["path"]), offsets)
    pixels = np.load(sweep["storage"]["path"])
    np.testing.assert_array_equal(pixels[0], np.load(plane["storage"]["path"]))
    fixed = tj.reconstruct(tj.load(path), "fbp", region="bbox").grid
    for i, centre in enumerate(offsets):
        shifted = tj.Scan(data, replace(geom, detector=replace(det, center=(centre, 0))))
        np.testing.assert_array_equal(
            pixels[i], tj.reconstruct(shifted, "fbp", grid=fixed).volume[2]
        )
    for stored in (sweep["storage"], plane["storage"], coord["coordinates"]["values"]):
        output(worker("release", artifact=stored["artifact"]))
        assert not Path(stored["path"]).exists()


def test_correction_preview_publishes_native_and_binned_pixel_centres(worker, tmp_path):
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


@pytest.mark.parametrize("rollback", [False, True])
def test_forced_stack_export_preserves_unrelated_directory_entries(
    worker, simulation, tmp_path, rollback
):
    stack = tmp_path / "stack"
    stack.mkdir()
    sentinel = stack / "acquisition.txt"
    sentinel.write_text("original")
    nested = stack / "calibration" / "reference.tif"
    nested.parent.mkdir()
    nested.write_bytes(b"flat")
    prefix = "sample.[a]"
    retained = [
        "acquisition.json",
        "flat.tiff",
        "other_00001.tif",
        f"{prefix}_notes.tif",
        f"{prefix}_7.tif",
    ]
    for name in retained:
        (stack / name).write_text(name)
    previous = stack / f"{prefix}.json"
    previous.write_bytes(b"old metadata")
    stale = stack / f"{prefix}_00123.tif"
    stale.write_bytes(b"previous stack")
    if rollback:
        # Walk a real failure after installing the TIFFs, at the manifest destination.
        stack.with_suffix(".manifest.json").mkdir()
    root = dict(type="member", handle=simulation["value"]["handle"], member="self")
    reply = worker(
        "run",
        operation=dict(type="export", source=root, path=str(stack), prefix=prefix, force=True),
    )
    assert reply["type"] == ("error" if rollback else "result")
    for name in retained:
        assert (stack / name).read_text() == name
    assert sentinel.read_text() == "original"
    assert nested.read_bytes() == b"flat"
    if rollback:
        assert previous.read_bytes() == b"old metadata"
        assert stale.read_bytes() == b"previous stack"
        assert not (stack / f"{prefix}_00000.tif").exists()
    else:
        assert not stale.exists()
        assert (stack / f"{prefix}_00000.tif").is_file()


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


def test_mlem_preflight_checks_the_measurement_domain(worker, tmp_path):
    path = tmp_path / "negative.nxs"
    grid = tj.Grid(3, 3, 2, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(4, 2, 1, 1), [0, 43, 91, 137])
    measured = np.ones((4, 2, 4), np.float32)
    measured[-1, -1, -1] = -0.125
    tj.save(path, tj.Scan(measured, geometry))
    recipe = dict(type="reconstruct", method="mlem", source=dict(type="path", path=str(path)))
    for command in ("validate", "plan", "run"):
        reply = worker(command, operation=recipe)
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
def test_fourier_preflight_checks_only_sampled_detector_rows(worker, tmp_path, bad_row):
    grid = tj.Grid(5, 5, 1, 1, 1, 1, vol_center=(0, 0, 0.25))
    geometry = tj.ParallelGeometry(grid, tj.Detector(7, 5, 1, 1), [12, 45, 90, 163])
    data = np.ones((4, 5, 7), np.float32)
    data[1, bad_row, 3] = np.nan
    path = tmp_path / "fractional-row.nxs"
    tj.save(path, tj.Scan(data, geometry))
    recipe = dict(type="reconstruct", method="fourier", source=dict(type="path", path=str(path)))
    for command in ("validate", "plan", "run"):
        reply = worker(command, operation=recipe)
        if bad_row == 0:  # z=0.25 lies one quarter of the way from row 2 to row 3.
            assert reply["type"] == "result", reply
        else:
            assert reply["type"] == "error", reply
            assert (reply["error"]["kind"], reply["error"]["path"]) == (
                "invalid_parameter",
                "source",
            )


@pytest.mark.parametrize("method", ["cgls", "fista", "spdhg", "sirt"])
def test_iterative_preflight_requires_finite_measurements(worker, tmp_path, method):
    grid = tj.Grid(4, 4, 2, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(4, 2, 1, 1), [0, 45, 90, 135])
    measured = np.zeros((4, 2, 4), np.float32)
    measured[2, 0, 1] = np.inf
    path = tmp_path / "nonfinite.nxs"
    tj.save(path, tj.Scan(measured, geometry))
    recipe = dict(type="reconstruct", method=method, source=dict(type="path", path=str(path)))
    for command in ("validate", "plan", "run"):
        reply = worker(command, operation=recipe)
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


@pytest.mark.parametrize(("original", "centres", "invalid"), [(0, [0, 20], 1), (20, [0], None)])
def test_fdk_sweep_validates_each_requested_detector_centre(
    worker, tmp_path, original, centres, invalid
):
    from dataclasses import replace

    grid = tj.Grid(4, 4, 2, 1, 1, 1)
    detector = tj.Detector(7, 5, 1, 1, (original, 0))
    geometry = tj.ConeGeometry(grid, detector, np.arange(0, 360, 30), tj.ConeBeam(50, 80))
    data = np.random.default_rng(17).uniform(size=(12, 5, 7)).astype(np.float32)
    path = tmp_path / "full-turn.nxs"
    tj.save(path, tj.Scan(data, geometry))
    args = dict(
        source=dict(type="path", path=str(path)), method="fbp", axis="z", index=0, centres=centres
    )
    for command in ("validate", "centre_sweep"):
        reply = (
            worker(command, operation=dict(type="centre_sweep", **args))
            if command == "validate"
            else worker(command, **args)
        )
        if invalid is not None:
            assert reply["type"] == "error", reply
            assert (reply["error"]["kind"], reply["error"]["path"]) == (
                "invalid_parameter",
                f"centres[{invalid}]",
            )
        elif command == "validate":
            assert reply["type"] == "result", reply
        else:
            pixels = np.load(output(reply)["value"]["planes"]["storage"]["path"])
            shifted = tj.Scan(data, replace(geometry, detector=replace(detector, center=(0, 0))))
            np.testing.assert_allclose(
                pixels[0], tj.reconstruct(shifted, "fbp").volume[0], atol=1e-6
            )
