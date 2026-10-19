"""Content recognition, physical metadata and lazy calibration of synthetic Zeiss OLE."""

from pathlib import Path
import struct

import numpy as np
import pytest
import tifffile

import tomojax as tj
from tomojax.geometry import stack_view_poses


def _regular_sectors(regular, entries, ids, start, fat):
    regular_payload = bytearray()
    for name, data in regular.items():
        first = start + len(regular_payload) // 512
        count = (len(data) + 511) // 512
        order = [*range(0, count, 2), *range(1, count, 2)]
        entries[ids[name]].update(start=first + order[0], size=len(data))
        block = bytearray(count * 512)
        for i, physical in enumerate(order):
            fat[first + physical] = first + order[i + 1] if i + 1 < count else 0xFFFFFFFE
            chunk = data[i * 512 : (i + 1) * 512]
            block[physical * 512 : physical * 512 + len(chunk)] = chunk
        regular_payload.extend(block)
    return regular_payload


def write_ole(path, streams):
    """Create nested streams with fragmented regular sectors and one MiniFAT."""
    free, end, fat_sector = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD
    names = {""}
    for name in streams:
        parts = name.split("/")
        names.update("/".join(parts[:i]) for i in range(1, len(parts) + 1))
    paths = ["", *sorted(names - {""}, key=lambda p: (p.count("/"), p))]
    ids = {p: i for i, p in enumerate(paths)}
    entries = [
        dict(
            name=p.rsplit("/", 1)[-1],
            left=free,
            right=free,
            child=free,
            kind=2 if p in streams else 1,
            start=end,
            size=0,
        )
        for p in paths
    ]
    entries[0].update(name="Root Entry", kind=5)

    def tree(children):
        if not children:
            return free
        mid = len(children) // 2
        index = children[mid]
        entries[index]["left"] = tree(children[:mid])
        entries[index]["right"] = tree(children[mid + 1 :])
        return index

    for p, index in ids.items():
        if p in streams:
            continue
        children = [ids[q] for q in paths[1:] if (q.rsplit("/", 1)[0] if "/" in q else "") == p]
        children.sort(key=lambda i: (len(entries[i]["name"]), entries[i]["name"].upper()))
        entries[index]["child"] = tree(children)
    payload = bytearray()
    regular = {}
    mini_fat = [free] * 128
    for name, data in streams.items():
        if len(data) >= 4096:
            regular[name] = data
            continue
        first = len(payload) // 64
        count = (len(data) + 63) // 64
        entries[ids[name]].update(start=first, size=len(data))
        for i in range(count):
            mini_fat[first + i] = first + i + 1 if i < count - 1 else end
        payload.extend(data + bytes(count * 64 - len(data)))
    dirs = (len(entries) + 3) // 4
    minis = (len(payload) + 511) // 512
    entries[0].update(start=dirs, size=len(payload))
    fat = [free] * 128
    for first, count in ((0, dirs), (dirs, minis)):
        for i in range(count):
            fat[first + i] = first + i + 1 if i < count - 1 else end
    regular_payload = _regular_sectors(regular, entries, ids, dirs + minis, fat)
    mini_at = dirs + minis + len(regular_payload) // 512
    fat_at = mini_at + 1
    fat[mini_at], fat[fat_at] = end, fat_sector
    directory = bytearray()
    for e in entries:
        name = (e["name"] + "\0").encode("utf-16le")
        directory.extend(
            struct.pack(
                "<64sHBBIII16sIQQIII",
                name,
                len(name),
                e["kind"],
                1,
                e["left"],
                e["right"],
                e["child"],
                bytes(16),
                0,
                0,
                0,
                e["start"],
                e["size"],
                0,
            )
        )
    directory.extend(bytes(dirs * 512 - len(directory)))
    payload.extend(bytes(minis * 512 - len(payload)))
    header = struct.pack(
        "<8s16sHHHHHH10I",
        bytes.fromhex("d0cf11e0a1b11ae1"),
        bytes(16),
        0x3E,
        3,
        0xFFFE,
        9,
        6,
        0,
        0,
        0,
        1,
        0,
        0,
        4096,
        mini_at,
        1,
        end,
        0,
    )
    header += struct.pack("<109I", fat_at, *([free] * 108))
    Path(path).write_bytes(
        header
        + directory
        + payload
        + regular_payload
        + struct.pack("<128I", *mini_fat)
        + struct.pack("<128I", *fat)
    )


def test_zeiss_content_geometry_and_raw_calibration(tmp_path):
    path = tmp_path / "renamed.data"
    counts = (500 + np.arange(3 * 8 * 12)).astype("<u2").reshape(3, 8, 12)
    reference = np.full((8, 12), 2500, "<f4")
    fields = {
        "ImageInfo/NoOfImages": ("<u4", [3]),
        "ImageInfo/ImageWidth": ("<u4", [12]),
        "ImageInfo/ImageHeight": ("<u4", [8]),
        "ImageInfo/DataType": ("<u4", [5]),
        "ImageInfo/AcquisitionMode": ("<u4", [0]),
        "ImageInfo/XrayGeometry": ("<u4", [1]),
        "ImageInfo/pixelsize": ("<f4", [10]),
        "ImageInfo/Angles": ("<f4", [0, 90, 180]),
        "ImageInfo/StoRADistance": ("<f4", [-40] * 3),
        "ImageInfo/DtoRADistance": ("<f4", [20] * 3),
        "alignment/x-shifts": ("<f4", [0.25] * 3),
        "alignment/y-shifts": ("<f4", [-0.5] * 3),
        "AutoRecon/CenterShift": ("<f4", [0.125]),
        "ReferenceData/DataType": ("<u4", [10]),
    }
    streams = {key: np.asarray(values, dtype).tobytes() for key, (dtype, values) in fields.items()}
    streams.update({f"ImageData1/Image{i + 1}": value.tobytes() for i, value in enumerate(counts)})
    streams["ReferenceData/Image"] = reference.tobytes()
    write_ole(path, streams)
    frames = tj.load_frames(path, darks=0)
    assert not isinstance(frames.counts, np.ndarray)
    np.testing.assert_array_equal(np.asarray(frames.counts), counts)
    np.testing.assert_array_equal(np.asarray(frames.flats)[0], reference)
    crop = frames.selected([1, 2]).cropped(slice(2, 6), slice(3, 9))
    np.testing.assert_array_equal(np.asarray(crop.counts), counts[1:, 2:6, 3:9])
    scan = tj.load_frames(path, darks=0).corrected()
    np.testing.assert_allclose(scan.projections, -np.log(counts / reference), rtol=2e-6)
    np.testing.assert_array_equal(scan.angles, [0, 90, 180])
    assert scan.grid.vx == pytest.approx(0.01)
    assert scan.detector.du == pytest.approx(0.015)
    np.testing.assert_allclose(scan.detector.center, [0.001875, -0.0075])
    assert scan.geometry.beam.source_to_axis == 40
    assert scan.geometry.beam.source_to_detector == 60
    streams["alignment/x-shifts"] = np.array([0.25, 0.5, -0.25], "<f4").tobytes()
    write_ole(path, streams)
    segments = tj.load_frames(path, darks=0).geometry.segments
    np.testing.assert_allclose(
        [s.detector.center[0] for s in segments], [0.001875, 0.005625, -0.005625]
    )
    streams["ImageInfo/XrayGeometry"] = np.array([0], "<u4").tobytes()
    write_ole(path, streams)
    parallel = tj.load_frames(path, darks=0).corrected()
    transforms = np.asarray(stack_view_poses(parallel.geometry, 3))
    np.testing.assert_allclose(transforms[:, 0, 3], [-0.00125, -0.00375, 0.00375], atol=1e-8)
    np.testing.assert_allclose(transforms[:, 2, 3], 0.005, atol=1e-8)
    streams["ImageInfo/AcquisitionMode"] = np.array([1], "<u4").tobytes()
    write_ole(path, streams)
    with pytest.raises(ValueError, match="not detector frames"):
        tj.load_frames(path, darks=0)

    streams["ImageInfo/AcquisitionMode"] = np.array([0], "<u4").tobytes()
    for key in (
        "AutoRecon/CenterShift",
        "alignment/y-shifts",
        "ImageInfo/pixelsize",
        "ImageInfo/XrayGeometry",
    ):
        partial = {name: value for name, value in streams.items() if name != key}
        write_ole(path, partial)
        with pytest.raises(ValueError, match="detector=|geometry="):
            tj.load_frames(path, darks=0)
        explicit = tj.load_frames(path, geometry=frames.geometry, darks=0)
        np.testing.assert_array_equal(explicit.counts[:], counts)
    write_ole(path, {k: v for k, v in streams.items() if k != "ReferenceData/DataType"})
    with pytest.raises(ValueError, match="flats="):
        tj.load_frames(path, darks=0)


def test_zeiss_fragmented_sector_windows_match_pixels_without_full_stream_reads(
    tmp_path, monkeypatch
):
    import olefile

    from tomojax.io.api import read_calibration_frames

    counts = np.arange(2 * 48 * 127, dtype="<u2").reshape(2, 48, 127)
    path = tmp_path / "large.xrm"
    fields = {"NoOfImages": 2, "ImageHeight": 48, "ImageWidth": 127, "DataType": 5}
    streams = {f"ImageInfo/{k}": np.array([v], "<u4").tobytes() for k, v in fields.items()}
    streams.update({f"ImageData1/Image{i + 1}": a.tobytes() for i, a in enumerate(counts)})
    write_ole(path, streams)
    original = olefile.OleFileIO.openstream

    def openstream(self, name):
        assert not str(name).startswith("ImageData"), "a row crop must not load an entire image"
        return original(self, name)

    monkeypatch.setattr(olefile.OleFileIO, "openstream", openstream)
    frames = read_calibration_frames(path)

    def no_reopen(*args, **kwargs):
        raise AssertionError("a batch must not parse the whole container's FAT again")

    monkeypatch.setattr(olefile, "OleFileIO", no_reopen)
    for rows, cols in [
        (slice(3, 11), slice(7, 120)),
        (slice(47, 48), slice(125, 127)),
        (slice(None), slice(None)),
    ]:
        got = frames.frames_at(np.array([1, 0, 1])).window(rows, cols)[:]
        np.testing.assert_array_equal(got, counts[[1, 0, 1], rows, cols])


def test_external_zeiss_reference_is_lazy_and_needs_no_sample_geometry(tmp_path, monkeypatch):
    import olefile

    reference = np.stack([np.full((8, 12), value, "<f4") for value in (2400, 2600)])
    path = tmp_path / "reference.data"
    fields = {
        "NoOfImages": 2,
        "ImageHeight": 8,
        "ImageWidth": 12,
        "DataType": 10,
        "AcquisitionMode": 1,
    }
    streams = {f"ImageInfo/{k}": np.array([v], "<u4").tobytes() for k, v in fields.items()}
    streams.update({f"ImageData1/Image{i + 1}": a.tobytes() for i, a in enumerate(reference)})
    write_ole(path, streams)
    counts = np.arange(192, dtype=np.uint16).reshape(2, 8, 12) + 500
    tifffile.imwrite(tmp_path / "counts.tif", counts, photometric="minisblack")
    opened = []
    original = olefile.OleFileIO.openstream

    def openstream(self, name):
        opened.append(name)
        return original(self, name)

    monkeypatch.setattr(olefile.OleFileIO, "openstream", openstream)
    frames = tj.load_frames(
        tmp_path / "counts.tif",
        angles=[0, 90],
        detector=tj.Detector(12, 8, 1, 1),
        flats=path,
        darks=0,
    )
    assert not any(str(name).startswith("ImageData") for name in opened)
    assert not isinstance(frames.flats, np.ndarray)
    np.testing.assert_array_equal(frames.flats[:], reference)
    small = frames.selected([1]).cropped(slice(2, 6), slice(3, 9)).corrected()
    np.testing.assert_allclose(small.projections, -np.log(counts[1:, 2:6, 3:9] / 2500), rtol=2e-6)
    with pytest.raises(ValueError, match="not detector frames"):
        tj.load_frames(path, darks=0)
