"""Content recognition, physical metadata and lazy calibration of synthetic Zeiss OLE."""

from pathlib import Path
import struct

import numpy as np
import pytest

import tomojax as tj
from tomojax.geometry import stack_view_poses


def write_ole(path, streams):
    """Create small nested streams using one FAT and one MiniFAT sector."""
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
    mini_fat = [free] * 128
    for name, data in streams.items():
        assert 0 < len(data) < 4096
        first = len(payload) // 64
        count = (len(data) + 63) // 64
        entries[ids[name]].update(start=first, size=len(data))
        for i in range(count):
            mini_fat[first + i] = first + i + 1 if i < count - 1 else end
        payload.extend(data + bytes(count * 64 - len(data)))
    dirs = (len(entries) + 3) // 4
    minis = (len(payload) + 511) // 512
    entries[0].update(start=dirs, size=len(payload))
    mini_at, fat_at = dirs + minis, dirs + minis + 1
    fat = [free] * 128
    for first, count in ((0, dirs), (dirs, minis)):
        for i in range(count):
            fat[first + i] = first + i + 1 if i < count - 1 else end
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
        header + directory + payload + struct.pack("<128I", *mini_fat) + struct.pack("<128I", *fat)
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
    frames = tj.load_frames(path)
    assert not isinstance(frames.counts, np.ndarray)
    np.testing.assert_array_equal(np.asarray(frames.counts), counts)
    np.testing.assert_array_equal(np.asarray(frames.flats)[0], reference)
    crop = frames.selected([1, 2]).cropped(slice(2, 6), slice(3, 9))
    np.testing.assert_array_equal(np.asarray(crop.counts), counts[1:, 2:6, 3:9])
    scan = tj.load(path)
    np.testing.assert_allclose(scan.projections, -np.log(counts / reference), rtol=2e-6)
    np.testing.assert_array_equal(scan.angles, [0, 90, 180])
    assert scan.grid.vx == pytest.approx(0.01)
    assert scan.detector.du == pytest.approx(0.015)
    np.testing.assert_allclose(scan.detector.center, [0.001875, -0.0075])
    assert scan.geometry.beam.source_to_axis == 40
    assert scan.geometry.beam.source_to_detector == 60
    streams["alignment/x-shifts"] = np.array([0.25, 0.5, -0.25], "<f4").tobytes()
    write_ole(path, streams)
    segments = tj.load_frames(path).geometry.segments
    np.testing.assert_allclose(
        [s.detector.center[0] for s in segments], [0.001875, 0.005625, -0.005625]
    )
    streams["ImageInfo/XrayGeometry"] = np.array([0], "<u4").tobytes()
    write_ole(path, streams)
    parallel = tj.load(path)
    transforms = np.asarray(stack_view_poses(parallel.geometry, 3))
    np.testing.assert_allclose(transforms[:, 0, 3], [-0.00125, -0.00375, 0.00375], atol=1e-8)
    np.testing.assert_allclose(transforms[:, 2, 3], 0.005, atol=1e-8)
    streams["ImageInfo/AcquisitionMode"] = np.array([1], "<u4").tobytes()
    write_ole(path, streams)
    with pytest.raises(ValueError, match="not detector frames"):
        tj.load_frames(path)
