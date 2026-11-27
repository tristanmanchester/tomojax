"""Read interchange files through external codecs and check their physical registration."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
import tifffile

import tomojax as tj
from tomojax.cli.main import main as cli
from tomojax.io.api import format_descriptions


def indexed(shape=(7, 9, 11)):
    z, y, x = np.indices(shape)
    grid = tj.Grid(shape[2], shape[1], shape[0], 0.2, 0.3, 0.5, vol_origin=(1.0, 2.0, -5.0))
    geometry = tj.ParallelGeometry(grid, tj.Detector(11, 7, 1, 1), np.arange(3.0))
    return tj.Reconstruction(
        (100 * z + 10 * y + x).astype("f4"), geometry, "fbp", {}, length_unit="mm"
    )


def test_export_requires_a_destination_and_a_registered_object(tmp_path):
    path = tmp_path / "volume.raw"
    array = indexed().volume
    for arguments in ((array, path), (path, array)):
        with pytest.raises(TypeError, match=r"export\(path, item\).*registered geometry"):
            tj.export(*arguments)
    assert not path.exists()


@pytest.mark.parametrize("width", [1, 8, 11])
@pytest.mark.parametrize("axis", "zyx")
@pytest.mark.parametrize("fmt", format_descriptions(), ids=lambda f: f["name"])
def test_each_writer_preserves_samples_and_registration(tmp_path, fmt, axis, width):
    item = indexed((5, 7, width))
    path = tmp_path / ("vol" + next(iter(fmt["suffixes"]), ""))
    if axis != "z" and "axis" not in fmt["options"]:
        with pytest.raises(tj.InvalidParameter, match="axis"):
            tj.export(path, item, axis=axis)
        assert not path.exists()
        return
    options = dict(axis=axis, crop=(slice(1, 5), slice(1, 7), None))
    plan = tj.export(path, item, dry_run=True, **options)
    assert not path.exists()
    receipt = tj.export(path, item, **options)
    assert {k: v for k, v in plan.items() if k != "free_bytes"} == {
        k: receipt[k] for k in plan if k != "free_bytes"
    }
    record = tj.inspect(path)["record"]
    array = record["arrays"]["volume"]
    order = axis + "zyx".replace(axis, "")
    shape = tuple(dict(z=4, y=6, x=width)[a] for a in order)
    assert array["shape"] == list(shape)
    assert [a["name"] for a in array["axes"]] == list(order)
    for a in array["axes"]:
        i = "zyx".index(a["name"])
        assert a["coordinates"]["origin"] == pytest.approx((-4.5, 2.3, 1.0)[i])
        assert a["coordinates"]["spacing"] == (0.5, 0.3, 0.2)[i]
        assert a["unit"] == "mm"
    expected = np.moveaxis(item.volume[1:5, 1:7], "zyx".index(axis), 0)
    if path.suffix == ".zarr":
        import zarr

        actual = zarr.open_group(path, mode="r")["0"][:]
    elif path.suffix == ".h5":
        import h5py

        with h5py.File(path) as file:
            actual = file["/entry/data/data"][:]
    elif path.is_dir():
        actual = np.stack([tifffile.imread(p) for p in sorted(path.glob("*.tif"))])
    elif path.suffix == ".tif":
        with tifffile.TiffFile(path) as t:
            actual = np.stack([page.asarray() for page in t.pages]).reshape(shape)
            assert t.is_imagej if fmt["name"] == "ImageJ TIFF" else t.is_ome
    elif path.suffix == ".am":
        header, pixels = path.read_bytes().split(b"# Data section follows\n@1\n")
        assert f"define Lattice {width} 6 4".encode() in header
        bounds = header.split(b"BoundingBox ")[1].split(b",")[0]
        np.testing.assert_allclose(
            np.fromstring(bounds, sep=" "), [1, 1 + (width - 1) * 0.2, 2.3, 3.8, -4.5, -3.0]
        )
        actual = np.frombuffer(pixels, "<f4").reshape(shape)
    elif path.suffix == ".mhd":
        fields = dict(line.split(" = ", 1) for line in path.read_text().splitlines())
        assert tuple(map(int, fields["DimSize"].split())) == shape[::-1]
        np.testing.assert_allclose(np.fromstring(fields["Offset"], sep=" "), (1, 2.3, -4.5))
        actual = np.fromfile(path.parent / fields["ElementDataFile"], "<f4").reshape(shape)
    else:
        actual = np.fromfile(path, "<f4").reshape(shape)
    np.testing.assert_array_equal(actual, expected)


def test_binning_moves_centers_and_quantization_counts_clips(tmp_path):
    item = indexed()
    path = tmp_path / "crop.raw"
    receipt = tj.export(
        path,
        item,
        crop=((1, 7), (1, 8), None),
        binning=2,
        axis="x",
        dtype="uint8",
        range=(200, 550),
    )
    axes = receipt["axes"]
    assert [a["name"] for a in axes] == ["x", "z", "y"]
    np.testing.assert_allclose([a["coordinates"]["origin"] for a in axes], [1.1, -4.25, 2.45])
    np.testing.assert_allclose([a["coordinates"]["spacing"] for a in axes], [0.4, 1.0, 0.6])
    # Mean of a 2-cube of the affine index field equals its value at the center.
    x, z, y = np.indices((5, 3, 3))
    physical = 100 * (1.5 + 2 * z) + 10 * (1.5 + 2 * y) + 0.5 + 2 * x
    stored = np.fromfile(path, "u1").reshape(physical.shape)
    decoded = 200 + (350 / 255) * stored
    np.testing.assert_allclose(decoded, np.clip(physical, 200, 550), rtol=0, atol=350 / 510)
    assert receipt["clipped"] == dict(
        below=int((physical < 200).sum()), above=int((physical > 550).sum())
    )


def test_nonfinite_unsampled_plane_preserves_the_previous_export(tmp_path):
    item = indexed((33, 5, 8))
    path = tmp_path / "volume.tif"
    tj.export(path, item)
    previous = path.read_bytes()
    item.volume[1, 1, 1] = np.nan
    with pytest.raises(ValueError, match=r"plane 1 has 1 non-finite"):
        tj.export(path, item, dtype="uint16")
    assert path.read_bytes() == previous
    constant = replace(item, volume=np.ones_like(item.volume))
    with pytest.raises(ValueError, match="supply range"):
        tj.export(path, constant, dtype="uint8", dry_run=True)
    for option in (dict(dtype="uint8"), dict(range=(0, 2)), dict(binning=2)):
        with pytest.raises(ValueError, match="mask"):
            tj.export(path, replace(item, volume=item.volume > 10), **option)
    assert path.read_bytes() == previous


def test_registered_format_reaches_export_inspect_cli_and_generated_surfaces(
    tmp_path, monkeypatch, capsys
):
    from tools.build_wire_options import description, format_table
    from tools.build_wire_schemas import documents

    from tomojax.cli import operations as op
    from tomojax.cli.mcp.tools.schemas import schema
    from tomojax.cli.mcp.tools.workflows import form
    from tomojax.io import _export_formats as formats  # check-public-imports: allow-private

    def write(path, blocks, record):
        with path.open("wb") as stream:
            stream.write(json.dumps(record).encode() + b"\n")
            for _, slab in blocks:
                stream.write(slab.tobytes())

    def read(path):
        with path.open("rb") as stream:
            return json.loads(stream.readline())

    fmt = formats.Format(
        "Test volume", (".example",), "file", ("crop",), "Sample values", write, read
    )
    monkeypatch.setattr(formats, "FORMATS", (*formats.FORMATS, fmt))
    path = tmp_path / "out.example"
    source = indexed()
    with pytest.raises(tj.InvalidParameter, match="dtype"):
        tj.export(path, source, dtype="uint8")
    assert tj.export(path, source, dry_run=True)["format"] == fmt.name
    tj.export(path, source)
    assert tj.inspect(path)["record"]["export"]["format"] == fmt.name
    native = tmp_path / "sample.nxs"
    tj.save(native, source)
    assert cli(["export", str(native), "-o", str(path), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["format"] == fmt.name
    defs = description()["$defs"]
    for document in (
        defs["Export"],
        documents()["operations.schema.json"]["$defs"]["Export"],
        schema(form(op.Export), defs),
    ):
        entries = document["properties"]["path"]["x-formats"]
        assert fmt.describe() in entries
    assert ".example" in format_table()


def test_cancel_during_slice_writes_retains_previous_stack(tmp_path):
    from tomojax import progress

    source = indexed()
    path = tmp_path / "slices"
    tj.export(path, source)
    old = sorted(p.read_bytes() for p in path.iterdir())
    control = progress.Control()

    def cancel(event):
        if isinstance(event, progress.Advanced) and event.done == 2:
            control.cancel()

    with progress.watch(cancel, control=control), pytest.raises(progress.Canceled):
        tj.export(path, replace(source, volume=source.volume * 2))
    assert sorted(p.read_bytes() for p in path.iterdir()) == old


def test_publication_to_a_new_file_is_exclusive(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from tomojax._loading.files import export_source
    from tomojax.io.api import plan_export

    source = export_source(indexed())
    path = tmp_path / "volume.raw"
    plans = [plan_export(path, source), plan_export(path, source)]
    ready = Barrier(2)

    def publish(plan):
        ready.wait()
        try:
            plan.publish(force=False)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(publish, plans)) == [False, True]
    np.testing.assert_array_equal(np.fromfile(path, "<f4"), indexed().volume.ravel())


def test_nexus_custom_group_scaling_and_whole_file_replacement(tmp_path, capsys):
    import shutil
    import subprocess

    import h5py

    item = indexed()
    native = tmp_path / "input.nxs"
    tj.save(native, item)
    path = tmp_path / "nested" / "output.h5"
    target = f"{path}::/analysis/final/attenuation"
    args = [
        "export",
        str(native),
        "-o",
        target,
        "--dtype",
        "uint16",
        "--range",
        "-4",
        "32763.5",
        "--compress",
    ]
    assert cli([*args, "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["replacement"] == "whole file" and not path.parent.exists()
    assert plan["paths"] == [str(path)]
    assert cli(args) == 0
    with h5py.File(path, "r+") as file:
        group = file["analysis/final"]
        assert file.attrs["default"] == "analysis" and file["analysis"].attrs["default"] == "final"
        signal = group[group.attrs["signal"]]
        assert signal.name == "/analysis/final/attenuation"
        assert signal[3, 5, 9] == 726
        assert (726 + group["attenuation_offset"][()]) * group["attenuation_scaling_factor"][
            ()
        ] == 359
        assert signal.compression == "gzip" and signal.shuffle
        file["unrelated"] = 21
    if h5dump := shutil.which("h5dump"):
        dump = subprocess.check_output(
            [h5dump, "-d", "/analysis/final/attenuation", "-s", "3,5,9", "-c", "1,1,1", str(path)],
            text=True,
        )
        assert "(3,5,9): 726" in dump
    assert cli([*args, "--force"]) == 0
    with h5py.File(path) as file:
        assert "unrelated" not in file
    for where in ("/data", "relative/group", "/entry/final/x", "/entry/record/data"):
        with pytest.raises(tj.InvalidParameter, match="HDF5"):
            tj.export(f"{path}::{where}", item, dry_run=True)
    with pytest.raises(SystemExit) as refused:
        cli(["export", str(native), "-o", f"{native}::/entry/volume/data", "--force"])
    assert refused.value.code == 2


@pytest.mark.parametrize(
    ("shape", "axis"), [((517, 5, 7), "z"), ((517, 5, 7), "x"), ((1, 9, 513), "y")]
)
def test_pyramid_one_pass_on_odd_and_singleton_axes(tmp_path, monkeypatch, shape, axis):
    from ome_zarr_models import open_ome_zarr
    import zarr

    from tomojax._loading.files import export_source
    from tomojax.io._export_data import Selection  # check-public-imports: allow-private
    from tomojax.io.api import plan_export

    item = indexed(shape)
    source = export_source(item)
    reads = []

    def read(region):
        reads.extend(range(region[0].start, region[0].stop))
        return source.read(region)

    # An odd slab depth forces pairs to span reads at multiple pyramid levels.
    monkeypatch.setattr(Selection, "block_depth", property(lambda self: 11))
    path = tmp_path / "volume.ome.zarr"
    plan_export(path, replace(source, read=read), axis=axis).publish(force=False)
    assert reads == list(range(shape[0]))
    root = zarr.open_group(path, mode="r")
    assert open_ome_zarr(root) is not None
    expected = np.moveaxis(item.volume, "zyx".index(axis), 0)
    order = [dict(z=0, y=1, x=2)[s] for s in axis + "zyx".replace(axis, "")]
    spacing = np.array([0.5, 0.3, 0.2])[order]
    origin = np.array([-5.0, 2.0, 1.0])[order]
    levels = root.attrs["ome"]["multiscales"][0]["datasets"]
    for i, level in enumerate(levels):
        np.testing.assert_array_equal(root[str(i)][:], expected)
        scale, shift = level["coordinateTransformations"]
        np.testing.assert_allclose(scale["scale"], spacing)
        np.testing.assert_allclose(shift["translation"], origin)
        if max(expected.shape) <= 256:
            assert i == len(levels) - 1
            break
        for d in range(3):
            n = expected.shape[d]
            if n > 1:
                even, odd = np.arange(0, n - 1, 2), np.arange(1, n, 2)
                expected = (expected.take(even, axis=d) + expected.take(odd, axis=d)) / 2
                origin[d] += spacing[d] / 2
                spacing[d] *= 2
    assert max(expected.shape) <= 256


def test_zarr_labels_and_missing_optional_dependency(tmp_path, monkeypatch):
    import sys

    import zarr

    item = indexed((1, 5, 519))
    mask = np.zeros(item.volume.shape, dtype="u1")
    mask[0, 1, 501] = 3
    path = tmp_path / "labels.zarr"
    tj.export(path, replace(item, volume=mask))
    root = zarr.open_group(path, mode="r")
    assert list(root.array_keys()) == ["0"]
    np.testing.assert_array_equal(root["0"][:], mask)
    absent = tmp_path / "missing.zarr"
    monkeypatch.setitem(sys.modules, "zarr", None)
    for dry in (False, True):
        with pytest.raises(tj.InvalidParameter, match=r"tomojax\[zarr\]"):
            tj.export(absent, item, dry_run=dry)
    assert not absent.exists()
