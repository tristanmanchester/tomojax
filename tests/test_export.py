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
    if path.is_dir():
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

    from tomojax._loading.files import volume_source
    from tomojax.io.api import plan_export

    source = volume_source(indexed())
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
