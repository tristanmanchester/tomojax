"""A writer must preserve every physical file of a live acquisition."""

import hashlib

import h5py
import numpy as np
import pytest
import tifffile

from tests import test_worker_adapters
from tests.test_mcp_stdio import connection
from tests.test_worker_adapters import output
import tomojax as tj

worker = test_worker_adapters.worker
simulation = test_worker_adapters.simulation
pytestmark = pytest.mark.surface


def acquisition(directory, layout):
    """Two sample views with independent calibration and known line integrals."""
    pixels = np.broadcast_to(np.array([4, 100, 52, 76], np.uint16)[:, None, None], (4, 2, 4))
    source = directory / "detector.h5"
    if layout != "raw":
        with h5py.File(source, "w") as f:
            f["pixels"] = pixels
    master = directory / "scan.nxs"
    with h5py.File(master, "w") as f:
        f["entry/definition"] = "NXtomo"
        f["entry/sample/rotation_angle"] = [0, 0, 0, 90]
        group = f.require_group("entry/instrument/detector")
        group["image_key"] = [2, 1, 0, 0]
        group["x_pixel_size"] = group["y_pixel_size"] = 1.0
        if layout == "vds":
            mapping = h5py.VirtualLayout(pixels.shape, pixels.dtype)
            mapping[:] = h5py.VirtualSource(source.name, "pixels", shape=pixels.shape)
            group.create_virtual_dataset("data", mapping)
        elif layout == "raw":
            data = group.create_dataset(
                "data", pixels.shape, pixels.dtype, external=[(str(source), 0, pixels.nbytes)]
            )
            data[:] = pixels
        elif layout == "transitive":
            with h5py.File(directory / "relay.h5", "w") as middle:
                middle["frames"] = h5py.ExternalLink(source.name, "/pixels")
            f["link"] = h5py.ExternalLink("relay.h5", "/frames")
            group["data"] = h5py.SoftLink("/link")
        else:
            group["data"] = h5py.ExternalLink(source.name, "/pixels")
    return master, source


def hashes(directory):
    return {
        path.relative_to(directory): hashlib.sha256(path.read_bytes()).digest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def member(reply):
    return {"type": "member", "member": "self", "handle": output(reply)["value"]["handle"]}


def refused(worker, operation, dependency, *, field="path"):
    # Check the read-only admission first, so a regression never reaches a writer.
    for command in ("validate", "run"):
        reply = worker(command, operation=operation)
        assert reply["type"] == "error", reply
        error = reply["error"]
        assert (error["kind"], error["path"]) == ("invalid_parameter", field)
        assert str(dependency) in error["message"] and "different destination" in error["message"]


@pytest.mark.parametrize("layout", ["external", "vds", "raw", "transitive", "symlink", "hardlink"])
def test_worker_preserves_storage_after_parent_handle_closes(worker, tmp_path, layout):
    master, source = acquisition(tmp_path, layout)
    dest = source
    if layout in ("symlink", "hardlink"):
        dest = tmp_path / "alias.h5"
        if layout == "symlink":
            dest.symlink_to(source.name)
        else:
            dest.hardlink_to(source)
    before = hashes(tmp_path)
    scan = member(worker("open", kind="scan", path=str(master)))
    selected = member(worker("run", operation=dict(type="select", source=scan, views=[1])))
    output(worker("close", handle=scan["handle"]))
    for force in (False, True):
        refused(worker, dict(type="save", source=selected, path=str(dest), force=force), source)
    assert hashes(tmp_path) == before
    np.testing.assert_allclose(tj.load(master).projections[:, 0, 0], -np.log([0.5, 0.75]))
    target = tmp_path / "selected.h5"
    for force in (False, True):
        output(
            worker(
                "run", operation=dict(type="save", source=selected, path=str(target), force=force)
            )
        )
    np.testing.assert_allclose(tj.load(target).projections[:, 0, 0], -np.log([0.75]))


@pytest.mark.parametrize("kind", ["tiff", "vendor", "flats", "darks", "borrowed"])
def test_worker_preserves_frames_and_calibrations(worker, tmp_path, kind):
    directory = tmp_path / "frames"
    directory.mkdir()
    files = [directory / f"part_{i:04}.tif" for i in (1, 2)]
    for i, path in enumerate(files):
        tifffile.imwrite(path, np.full((2, 4), 50 + i, np.uint16))
    angles = tmp_path / "angles.npy"
    np.save(angles, [0.0, 90.0])
    options = dict(angles=dict(type="npy", path=str(angles)), flats=100, darks=0, pixel_size=1)
    dependency = files[-1]
    if kind == "vendor":
        (directory / "part.xtekct").write_text(
            "[XTekCT]\nName=part\nSrcToObject=12\nSrcToDetector=18\nDetectorOffsetX=0\n"
            "DetectorOffsetY=0\nCentreOfRotationTop=0\nCentreOfRotationBottom=0\n"
            "DetectorPixelSizeX=0.1\nDetectorPixelSizeY=0.1\nWhiteLevel=100\n"
            "InitialAngle=0\nAngularStep=90\n"
        )
        options = dict(darks=0)
    if kind in ("flats", "darks"):
        dependency = tmp_path / "calibration.tif"
        tifffile.imwrite(dependency, np.full((2, 4), 100 if kind == "flats" else 2, np.uint16))
        options[kind] = str(dependency)
    if kind == "borrowed":
        other, dependency = acquisition(tmp_path, "external")
        options["flats"] = str(other)
    alias = tmp_path / "output.h5"
    alias.hardlink_to(dependency)
    before = hashes(tmp_path)

    def open_scan():
        return member(worker("open", kind="scan", path=str(directory), **options))

    scan = open_scan()
    for force in (False, True):
        refused(worker, dict(type="save", source=scan, path=str(alias), force=force), dependency)
    assert hashes(tmp_path) == before
    open_scan()


def test_worker_protects_an_independent_open_input_and_checkpoint(worker, simulation, tmp_path):
    master, source = acquisition(tmp_path, "external")
    frames = member(worker("open", kind="frames", path=str(master)))
    before = hashes(tmp_path)
    sim = simulation["value"]
    for force in (False, True):
        refused(
            worker,
            dict(
                type="export",
                source=dict(type="member", handle=sim["handle"], member="self"),
                path=str(tmp_path),
                force=force,
            ),
            master,
        )
        refused(
            worker,
            dict(
                type="reconstruct_slabs", method="fbp", source=frames, path=str(source), force=force
            ),
            source,
        )
    corrected = member(worker("run", operation=dict(type="correct", source=frames)))
    for op in ("align", "reconstruct", "reconstruct_slabs"):
        settings = dict(source=corrected, checkpoint=str(tmp_path))
        if op == "reconstruct_slabs":
            settings.update(method="fbp", path=str(tmp_path / "out.h5"), source=frames)
        refused(worker, dict(type=op, **settings), master, field="checkpoint")
    assert hashes(tmp_path) == before


def test_mcp_and_python_name_the_input_and_a_safe_next_step(tmp_path):
    master, source = acquisition(tmp_path, "external")
    before = hashes(tmp_path)
    with connection(tmp_path) as client:
        frames = client.call("open", kind="frames", path=master.name)["structuredContent"]["handle"]
        scan = client.call("correct", source=frames)["structuredContent"]["handle"]
        sim = client.call("simulate", phantom="sphere", size=8, n_views=4)["structuredContent"][
            "handle"
        ]
        for force in (False, True):
            for name, options in [
                ("save", dict(source=scan, path=source.name)),
                ("export", dict(source=sim, path=str(tmp_path))),
                ("reconstruct_slabs", dict(source=frames, method="fbp", path=source.name)),
            ]:
                for dry in (True, False):
                    reply = client.call(name, **options, dry_run=dry, force=force)
                    assert reply["isError"], reply
                    error = reply["structuredContent"]
                    assert str(tmp_path) in error["message"]
                    assert "different destination" in error["message"]
        for name in ("align", "reconstruct"):
            for dry in (True, False):
                reply = client.call(name, source=scan, checkpoint=str(tmp_path), dry_run=dry)
                error = reply["structuredContent"]
                assert reply["isError"] and error["path"] == "checkpoint"
                assert "different destination" in error["message"]
    # Holding a derived scan keeps the dependency alive through Python's ordinary writers.
    scan = tj.load(master).selected([0])
    with pytest.raises(ValueError, match="different destination"):
        tj.save(source, scan)
    result = tj.Reconstruction(np.zeros(scan.grid.shape), scan.geometry, "test", {}, scan)
    with pytest.raises(ValueError, match="different destination"):
        tj.export(tmp_path, result)
    assert hashes(tmp_path) == before


@pytest.mark.parametrize("command", ["correct", "reconstruct", "align", "inspect", "simulate"])
def test_cli_refuses_linked_destinations(tmp_path, command):
    import subprocess
    import sys

    master, source = acquisition(tmp_path, "external")
    inputs = [str(master)]
    options = ["-o", str(source)]
    if command == "inspect":
        (tmp_path / "projection.png").hardlink_to(source)
        options = ["--preview-dir", str(tmp_path)]
    elif command == "simulate":
        source = tmp_path / "settings.toml"
        source.write_text("size = 8\n")
        alias = tmp_path / "synthetic.h5"
        alias.hardlink_to(source)
        inputs = ["--config", str(source)]
        options = ["-o", str(alias)]
    before = hashes(tmp_path)
    for flags in ([], ["--force"]):
        process = subprocess.run(
            [sys.executable, "-m", "tomojax.cli.main", command, *inputs, *options, *flags],
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode != 0
        assert source.name in process.stdout + process.stderr
        assert "different destination" in process.stdout + process.stderr
        assert hashes(tmp_path) == before


def test_load_during_staging_prevents_publication(tmp_path):
    from tomojax.core.files import replaced_when_complete

    master, source = acquisition(tmp_path, "external")
    before = hashes(tmp_path)

    def stage_then_load():
        with replaced_when_complete(source, "volume") as stage:
            stage.write_bytes(b"staged")
            scan = tj.load(master)
            np.testing.assert_allclose(scan.projections[:, 0, 0], -np.log([0.5, 0.75]))

    with pytest.raises(ValueError, match="different destination"):
        stage_then_load()
    assert hashes(tmp_path) == before


@pytest.mark.parametrize("writer", ["save", "exclusive", "slabs", "worker"])
def test_native_writers_recheck_a_destination_acquired_after_staging(tmp_path, writer):
    from tests.test_export import indexed
    from tomojax import progress
    from tomojax.cli._staging import Stages  # check-public-imports: allow-private
    from tomojax.io.api import NXTomoMetadata, save_nxtomo, slab_volume

    master, physical = acquisition(tmp_path, "external")
    original = hashes(tmp_path)
    destination = tmp_path / "result.nxs"
    value = indexed()
    opened = []

    def acquire():
        destination.hardlink_to(physical)
        opened.append(tj.load(master))

    def observe(event):
        if isinstance(event, progress.Advanced) and not destination.exists():
            acquire()

    def write():
        if writer == "worker":
            stages = Stages(tmp_path / "session")
            with stages.temporary:
                tj.save(stages.output(destination, force=True), value)
                acquire()
                stages.commit()
        elif writer == "slabs":
            with slab_volume(destination, NXTomoMetadata(), value.volume.shape) as volume:
                volume[:] = value.volume
                acquire()
        else:
            with progress.watch(observe):
                if writer == "exclusive":
                    save_nxtomo(str(destination), value.volume, overwrite=False)
                else:
                    tj.save(destination, value)

    with pytest.raises(tj.InvalidParameter, match=physical.name):
        write()
    assert len(opened) == 1 and destination.samefile(physical)
    destination.unlink()
    assert hashes(tmp_path) == original


def test_worker_releases_input_protection_with_the_last_derived_handle(
    worker, simulation, tmp_path
):
    master, source = acquisition(tmp_path, "external")
    scan = member(worker("open", kind="scan", path=str(master)))
    selected = member(worker("run", operation=dict(type="select", source=scan, views=[1])))
    sim = dict(type="member", handle=simulation["value"]["handle"], member="self")
    save = dict(type="save", source=sim, path=str(source), force=True)
    output(worker("close", handle=scan["handle"]))
    assert worker("validate", operation=save)["type"] == "error"
    output(worker("close", handle=selected["handle"]))
    output(worker("validate", operation=save))


@pytest.mark.parametrize(
    ("name", "alias"),
    [
        ("plot.h5", "plot.h5"),
        ("custom.h5::/plot/signal", "custom.h5"),
        ("image.tif", "image.tif"),
        ("image.ome.tif", "image.ome.tif"),
        ("slices", "slices/extra.h5"),
        ("pyramid.zarr", "pyramid.zarr/extra.h5"),
        ("previous.zarr", ".previous.zarr.previous/extra.h5"),
        ("volume.raw", "volume.raw.json"),
        ("header.mhd", "header.raw"),
        ("amira.am", "amira.am.json"),
        ("recovery", ".recovery.previous/extra.h5"),
    ],
)
def test_export_rechecks_the_group_after_staging(tmp_path, name, alias):
    from tests.test_export import indexed
    from tomojax import progress

    master, source = acquisition(tmp_path, "external")
    target = tmp_path / name
    volume = indexed()
    if alias.startswith("."):
        tj.export(target, volume)
        target.rename(tmp_path / f".{name}.previous")
    elif target != tmp_path / alias:
        tj.export(target, volume)
    dependency = tmp_path / alias
    dependency.unlink(missing_ok=True)
    dependency.hardlink_to(source)
    before = hashes(tmp_path)
    opened = []
    export_task = None

    def observe(event):
        nonlocal export_task
        if isinstance(event, progress.Started) and event.name == "export":
            export_task = event.task
        if (
            isinstance(event, progress.Advanced)
            and event.task == export_task
            and not opened
            and event.done > 0
        ):
            opened.append(tj.load(master))

    with progress.watch(observe), pytest.raises(ValueError, match="different destination"):
        tj.export(target, volume)
    assert len(opened) == 1
    with pytest.raises(tj.InvalidParameter, match=source.name):
        tj.save(source, opened[0])
    assert hashes(tmp_path) == before


@pytest.fixture
def export_conflicts(tmp_path):
    from tests.test_export import indexed

    value = indexed()
    master = tmp_path / "volume.nxs"
    storage = tmp_path / "voxels.h5"
    tj.save(master, value)
    with h5py.File(master, "r+") as file, h5py.File(storage, "w") as physical:
        volume = "/entry/volume/data"
        file.copy(volume, physical, name="pixels")
        del file[volume]
        file[volume] = h5py.ExternalLink(storage.name, "/pixels")

    destinations = []
    for name, written in [
        ("plot.h5", "plot.h5"),
        ("custom.h5::/plot/signal", "custom.h5"),
        ("image.tif", "image.tif"),
        ("image.ome.tif", "image.ome.tif"),
        ("binary.raw", "binary.raw"),
        ("raw-record.raw", "raw-record.raw.json"),
        ("alias.zarr", "alias.zarr"),
        ("header.mhd", "header.mhd"),
        ("meta.mhd", "meta.raw"),
        ("meta-record.mhd", "meta-record.mhd.json"),
        ("amira.am", "amira.am"),
        ("amira-record.am", "amira-record.am.json"),
    ]:
        (tmp_path / written).hardlink_to(storage)
        destinations.append(tmp_path / name)
    for name in ("slices", "recovery", "pyramid.zarr", "previous.zarr"):
        path = tmp_path / name
        tj.export(path, value)
        directory = path
        if name in {"recovery", "previous.zarr"}:
            directory = path.rename(tmp_path / f".{name}.previous")
        (directory / "extra.h5").hardlink_to(storage)
        destinations.append(path)
    return master, storage, destinations


def test_format2_load_uses_the_generation_registered_before_a_rename(
    tmp_path, export_conflicts, monkeypatch
):
    from tomojax.core import files

    master, physical, _ = export_conflicts
    alias = tmp_path / "original.nxs"
    alias.hardlink_to(master)
    replacement = tmp_path / "next.nxs"
    with h5py.File(replacement, "w") as file:
        file["different_generation"] = 42
    collect = files.hdf5_sources

    def swap_after_discovery(file):
        sources = collect(file)
        if file.filename == str(master) and replacement.exists():
            replacement.replace(master)
        return sources

    monkeypatch.setattr(files, "hdf5_sources", swap_after_discovery)
    loaded = tj.load_reconstruction(master)
    assert not replacement.exists() and loaded.volume[2, 3, 4] == 234
    before = hashes(tmp_path)
    for dependency in (physical, alias):
        with pytest.raises(tj.InvalidParameter, match="different destination"):
            tj.save(dependency, loaded)
    assert hashes(tmp_path) == before


def test_python_export_protects_each_format_and_directory_recovery(tmp_path, export_conflicts):
    master, storage, destinations = export_conflicts
    original = hashes(tmp_path)
    volume = tj.load_reconstruction(master)
    with pytest.raises(tj.InvalidParameter, match=storage.name):
        tj.save(storage, volume)
    for path in destinations:
        for dry in (True, False):
            with pytest.raises(tj.InvalidParameter, match=storage.name):
                tj.export(path, volume, dry_run=dry)
    for missing in ("recovery", "previous.zarr"):
        with pytest.raises(tj.InvalidParameter, match=storage.name):
            tj.inspect(tmp_path / missing)
        assert not (tmp_path / missing).exists()
    assert hashes(tmp_path) == original
    z, y, x = np.indices((7, 9, 11))
    np.testing.assert_array_equal(tj.load_reconstruction(master).volume, 100 * z + 10 * y + x)


def test_cli_export_protects_each_format(tmp_path, export_conflicts, capsys):
    from tomojax.cli.main import main

    master, storage, destinations = export_conflicts
    original = hashes(tmp_path)
    for path in destinations:
        for flags in ([], ["--force"]):
            with pytest.raises(SystemExit) as failure:
                main(["export", str(master), "-o", str(path), *flags])
            message = capsys.readouterr().err
            assert failure.value.code == 2 and str(storage) in message, message
            assert "different destination" in message
    assert hashes(tmp_path) == original


def test_worker_export_protects_each_format(tmp_path, worker, export_conflicts):
    master, storage, destinations = export_conflicts
    source = member(worker("open", kind="reconstruction", path=str(master)))
    original = hashes(tmp_path)
    for force in (False, True):
        refused(worker, dict(type="save", source=source, path=str(storage), force=force), storage)
    for path in destinations:
        for force in (False, True):
            refused(
                worker, dict(type="export", source=source, path=str(path), force=force), storage
            )
    assert hashes(tmp_path) == original
    output(worker("close", handle=source["handle"]))


def test_mcp_export_protects_each_format(tmp_path, export_conflicts):
    master, storage, destinations = export_conflicts
    original = hashes(tmp_path)
    with connection(tmp_path) as client:
        source = client.call("open", kind="reconstruction", path=master.name)["structuredContent"]
        for force in (False, True):
            for dry in (True, False):
                reply = client.call(
                    "save", source=source["handle"], path=storage.name, force=force, dry_run=dry
                )
                assert reply["isError"] and str(storage) in reply["structuredContent"]["message"]
        for path in destinations:
            for force in (False, True):
                for dry in (True, False):
                    reply = client.call(
                        "export",
                        source=source["handle"],
                        path=str(path.relative_to(tmp_path)),
                        force=force,
                        dry_run=dry,
                    )
                    error = reply["structuredContent"]
                    assert reply["isError"] and error["path"] == "path", reply
                    assert (
                        str(storage) in error["message"]
                        and "different destination" in error["message"]
                    )
    assert hashes(tmp_path) == original


def test_worker_inspection_restores_a_directory_with_no_live_inputs(worker, tmp_path):
    from tests.test_export import indexed

    path = tmp_path / "slices"
    tj.export(path, indexed())
    previous = path.rename(tmp_path / ".slices.previous")
    report = output(worker("inspect", path=str(path)))
    assert report["value"]["type"] == "inspection"
    assert path.is_dir() and not previous.exists()


@pytest.mark.parametrize("write", [tj.export, tj.save])
def test_python_writers_protect_the_mapped_volume_they_read(tmp_path, write):
    from dataclasses import replace

    from tests.test_export import indexed

    item = indexed()
    source = tmp_path / "mapped.bin"
    item.volume.tofile(source)
    mapped = np.memmap(source, mode="r", dtype=np.float32, shape=item.volume.shape)
    alias = tmp_path / ("volume.raw" if write is tj.export else "volume.nxs")
    alias.hardlink_to(source)
    before = hashes(tmp_path)
    with pytest.raises(tj.InvalidParameter, match=source.name):
        write(alias, replace(item, volume=mapped))
    assert hashes(tmp_path) == before


@pytest.mark.parametrize("mapped", ["pixels", "coordinates"])
def test_native_mask_protects_its_mapped_inputs(tmp_path, mapped):
    from tomojax.io.api import save_mask
    from tomojax.wire.arrays import Array, Axis, Coordinates, Npy

    source = tmp_path / "source"
    source.write_bytes(bytes((0, 1, 0)))
    data = np.memmap(source, mode="r", dtype=np.uint8)
    alias = tmp_path / "mask.nxs"
    alias.hardlink_to(source)
    storage = Npy(str(source))
    axis = Axis("view", "acquisition", "degree", Coordinates(storage))
    registration = Array(storage, (3,), "uint8", (axis,), "1", "mask")
    with pytest.raises(tj.InvalidParameter, match=source.name):
        save_mask(
            alias,
            registration,
            name="box",
            recipe={},
            box=(slice(0, 3),),
            inside=data if mapped == "pixels" else data.copy(),
            coordinates={"view": data if mapped == "coordinates" else data.copy()},
            history=[],
        )
    assert alias.read_bytes() == bytes((0, 1, 0)) and alias.samefile(source)


@pytest.mark.parametrize("kind", ["scan", "frames"])
def test_worker_protects_selected_external_pixels_keys_and_angles(worker, tmp_path, kind):
    from tests.test_load_selection import write_scan

    master = tmp_path / "scan.nxs"
    write_scan(master)
    pixels = np.broadcast_to(np.array([51, 76, 26], np.float32)[:, None, None], (3, 2, 4))
    options = {}
    sources = []
    with h5py.File(master, "r+") as file:
        alternate = file.create_group("alternate")
        for field, name, values in [
            ("data_path", "frames", pixels),
            ("image_key_path", "key", [0, 3, 0]),
            ("angles_path", "angles", [15.0, 75.0, 135.0]),
        ]:
            backing = tmp_path / f"{name}.h5"
            with h5py.File(backing, "w") as target:
                target["values"] = values
                if field == "angles_path":
                    target["values"].attrs["units"] = "degree"
            alternate[name] = h5py.ExternalLink(backing.name, "/values")
            options[field] = f"/alternate/{name}"
            sources.append(backing)

    expected = pixels[[0, 2]]
    if kind == "frames":
        options.update(flats=101, darks=1)
        expected = -np.log((expected - 1) / 100)

    before = hashes(tmp_path)
    selected = member(worker("open", kind=kind, path=str(master), **options))
    if kind == "frames":
        scan = member(worker("run", operation=dict(type="correct", source=selected)))
        output(worker("close", handle=selected["handle"]))
    else:
        scan = selected
    for backing in sources:
        for force in (False, True):
            refused(worker, dict(type="save", source=scan, path=str(backing), force=force), backing)
    assert hashes(tmp_path) == before
    copy = tmp_path / "selected.nxs"
    output(worker("run", operation=dict(type="save", source=scan, path=str(copy))))
    output(worker("close", handle=scan["handle"]))
    direct = (
        tj.load_frames(master, **options).corrected()
        if kind == "frames"
        else tj.load(master, **options)
    )
    for backing in sources:
        with pytest.raises(tj.InvalidParameter, match=backing.name):
            tj.save(backing, direct)
    after = hashes(tmp_path)
    assert all(after[name] == digest for name, digest in before.items())
    for reopened in (tj.load(copy), direct):
        np.testing.assert_allclose(reopened.projections, expected)
        np.testing.assert_array_equal(reopened.angles, [15.0, 135.0])


@pytest.fixture(params=["disabled", "enabled", "forced"])
def collect_step(request):
    import gc

    prior = gc.isenabled()
    gc.collect()
    (gc.disable if request.param == "disabled" else gc.enable)()
    try:
        yield gc.collect if request.param == "forced" else lambda: None
    finally:
        (gc.enable if prior else gc.disable)()
        gc.collect()


def test_validation_releases_inputs_even_when_caller_keeps_traceback(tmp_path, collect_step):
    from tests.test_mcp_evaluate import CGLS, STRIDE
    from tests.worker_adapter import Stage, admitted
    from tomojax.cli import operations as op, wire_options

    path = tmp_path / "scan.nxs"
    scan = tj.simulate(size=8, n_views=24).scan
    tj.save(path, scan)
    collect_step()
    cmd = op.Evaluate(
        op.PathInput(str(path)), (CGLS,), STRIDE, (wire_options.Rings(views=1),), CGLS
    )
    with pytest.raises(ValueError, match="views") as failure:
        admitted(op.Validate(operation=cmd), None, Stage(tmp_path), None)
    collect_step()
    # Diagnostics remain available after the validation has returned.
    assert failure.value.__traceback__ is not None
    tj.save(path, scan.selected([3]))
    np.testing.assert_array_equal(tj.load(path).angles, scan.angles[[3]])


def test_python_ownership_follows_reachable_results_in_cycles(tmp_path, collect_step):
    path = tmp_path / "scan.nxs"
    reference = tj.simulate(size=8, n_views=8).scan
    tj.save(path, reference)
    collect_step()
    scan = tj.load(path)
    keep = [scan.selected([2])]
    keep.append(keep)
    del scan
    collect_step()
    before = hashes(tmp_path)
    with pytest.raises(tj.InvalidParameter, match=path.name):
        tj.save(path, reference)
    assert hashes(tmp_path) == before
    del keep
    collect_step()
    tj.save(path, reference.selected([5]))
    np.testing.assert_array_equal(tj.load(path).angles, reference.angles[[5]])


@pytest.mark.parametrize("mode", ["disabled", "enabled", "forced"])
def test_mcp_close_and_python_namespace_have_independent_lifetimes(tmp_path, mode):
    from tests.test_mcp_python import python

    master, source = acquisition(tmp_path, "external")
    before = hashes(tmp_path)
    with connection(tmp_path) as client:

        def step():
            if mode == "forced":
                python(client, "gc.collect()")

        python(
            client, "import gc\ngc.disable()" if mode == "disabled" else "import gc\ngc.enable()"
        )
        handle = client.call("open", path=master.name, kind="scan")["structuredContent"]["handle"]
        step()
        python(client, f"kept = handles[{handle!r}].selected([1])")
        step()
        assert not client.call("control", action="close", target=handle)["isError"]
        step()
        sim = client.call("simulate", size=8, n_views=8)["structuredContent"]["handle"]
        save = dict(source=sim, path=source.name, force=True)
        failure = client.call("save", **save)
        assert failure["isError"] and str(source) in failure["structuredContent"]["message"]
        assert hashes(tmp_path) == before
        step()
        python(client, "del kept")
        step()
        assert not client.call("save", **save)["isError"]


def test_worker_keeps_source_until_standalone_artifact_release(worker, simulation, tmp_path):
    master, source = acquisition(tmp_path, "external")
    before = hashes(tmp_path)
    scan = member(worker("open", kind="scan", path=str(master)))
    plane = output(worker("slice", source=scan, axis="z", index=0, method="fbp"))["value"]["plane"]
    output(worker("close", handle=scan["handle"]))
    sim = dict(type="member", handle=simulation["value"]["handle"], member="self")
    save = dict(type="save", source=sim, path=str(source), force=True)
    refused(worker, save, source)
    assert hashes(tmp_path) == before
    output(worker("release", artifact=plane["storage"]["artifact"]))
    output(worker("validate", operation=save))
