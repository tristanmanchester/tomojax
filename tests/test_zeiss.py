"""Content recognition, physical metadata and lazy calibration of synthetic Zeiss OLE."""

from contextlib import nullcontext
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


def _check_missing_zeiss_inputs(path, streams, frames, counts):
    """Absent geometry, flat or image streams name the input they need."""
    streams["ImageInfo/AcquisitionMode"] = np.array([0], "<u4").tobytes()
    for key in (
        "AutoRecon/CenterShift",
        "alignment/y-shifts",
        "ImageInfo/pixelsize",
        "ImageInfo/XrayGeometry",
    ):
        partial = {name: value for name, value in streams.items() if name != key}
        write_ole(path, partial)
        if key in ("alignment/y-shifts", "ImageInfo/XrayGeometry"):  # no axis without them
            with pytest.raises(ValueError, match="geometry="):
                tj.load_frames(path, darks=0)
        else:  # the middle centre, said; and pixel units only without the pixel size
            meta = tj.load_frames(path, darks=0).source.geometry_metadata
            assert meta["detector_center_status"] == "not_recorded"
            assert (meta.get("length_unit") == "pixel") == (key == "ImageInfo/pixelsize")
        # geometry= replaces whatever shifts the file still records, and says so.
        dropped = key != "alignment/y-shifts"
        with pytest.warns(UserWarning, match="geometry= replaces") if dropped else nullcontext():
            explicit = tj.load_frames(path, geometry=frames.geometry, darks=0)
        np.testing.assert_array_equal(explicit.counts[:], counts)
    write_ole(path, {k: v for k, v in streams.items() if k != "ReferenceData/DataType"})
    with pytest.raises(ValueError, match="flats="):
        tj.load_frames(path, darks=0)
    write_ole(path, {k: v for k, v in streams.items() if k != "ImageData1/Image2"})
    with pytest.raises(ValueError, match="missing declared Zeiss image stream 'ImageData1/Image2'"):
        tj.load_frames(path, darks=0)


def _streams(counts: np.ndarray, reference: np.ndarray, declared_count: int = 3) -> dict:
    """A cone Zeiss scan of three views with its vendor shifts and centre shift."""
    fields = {
        "ImageInfo/NoOfImages": ("<u4", [declared_count]),
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
    return streams


@pytest.mark.parametrize("declared_count", [3, 10])
def test_zeiss_content_geometry_and_raw_calibration(tmp_path, declared_count):
    path = tmp_path / "renamed.data"
    counts = (500 + np.arange(3 * 8 * 12)).astype("<u2").reshape(3, 8, 12)
    reference = np.full((8, 12), 2500, "<f4")
    streams = _streams(counts, reference, declared_count)
    write_ole(path, streams)
    frames = tj.load_frames(path, darks=0)
    assert frames.source.geometry_metadata["zeiss"]["declared_view_count"] == declared_count
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
    geometry = tj.load_frames(path, darks=0).geometry
    assert geometry.trajectory_lengths == (3,)
    segments = geometry.segments
    np.testing.assert_allclose(
        [s.detector.center[0] for s in segments], [0.001875, 0.005625, -0.005625]
    )
    streams["ImageInfo/XrayGeometry"] = np.array([0], "<u4").tobytes()
    write_ole(path, streams)
    parallel = tj.load_frames(path, darks=0).corrected()
    transforms = np.asarray(stack_view_poses(parallel.geometry, 3))
    # The recorded centre stays on the detector and each view's shift is a pose:
    # together they move the image by -(x-shift - CenterShift) pixels, as for cone.
    assert parallel.detector.center[0] == pytest.approx(-0.00125)
    np.testing.assert_allclose(transforms[:, 0, 3], [-0.0025, -0.005, 0.0025], atol=1e-8)
    np.testing.assert_allclose(
        transforms[:, 0, 3] - parallel.detector.center[0], [-0.00125, -0.00375, 0.00375], atol=1e-8
    )
    np.testing.assert_allclose(transforms[:, 2, 3], 0.005, atol=1e-8)
    streams["ImageInfo/AcquisitionMode"] = np.array([1], "<u4").tobytes()
    write_ole(path, streams)
    with pytest.raises(ValueError, match="not detector frames"):
        tj.load_frames(path, darks=0)

    _check_missing_zeiss_inputs(path, streams, frames, counts)


def test_zeiss_beam_hardening_settings_come_from_the_scan_and_its_reconstruction(tmp_path):
    fields = {"NoOfImages": 1, "ImageWidth": 2, "ImageHeight": 2, "DataType": 5}
    streams = {f"ImageInfo/{k}": np.array([v], "<u4").tobytes() for k, v in fields.items()}
    streams["ImageData1/Image1"] = np.ones(4, "<u2").tobytes()
    streams["ImageInfo/AcquisitionMode"] = np.array([0], "<u4").tobytes()
    streams["ImageInfo/XrayVoltage"] = np.array([139.8, 140.0, 140.1], "<f4").tobytes()
    streams["ImageInfo/SourceFilterName"] = b"HE2" + bytes(257)
    write_ole(tmp_path / "scan.txrm", streams)
    recon = {"ReconSettings/BeamHardening": np.array([0.05], "<f4").tobytes()}
    write_ole(tmp_path / "scan_recon.txm", recon)
    geometry = tj.ParallelGeometry(tj.Grid(2, 2, 2, 1, 1, 1), tj.Detector(2, 2, 1, 1), [0.0])
    frames = tj.load_frames(tmp_path / "scan.txrm", darks=0, flats=1.0, geometry=geometry)
    hardening = frames.source.geometry_metadata["beam_hardening"]
    assert hardening == {"vendor": "zeiss", "kv": 140.0, "filter": "HE2", "constant": 0.05}


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
    loadfat = olefile.OleFileIO.loadfat

    def checked_fat(self, header):
        loadfat(self, header)
        expected = self.sect2array(self.getsect(struct.unpack_from("<I", header, 76)[0]))
        for sectors in (header[76:512], self.sect2array(header[76:512])):
            table, size = self.fat, len(self.fat)
            assert self.loadfat_sect(sectors) == olefile.FREESECT
            assert self.fat is table, "FAT growth must not copy the accumulated table"
            assert self.fat[size:] == expected
            del self.fat[size:]

    monkeypatch.setattr(olefile.OleFileIO, "loadfat", checked_fat)

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


def test_multireferences_use_recorded_times_and_exposures(tmp_path):
    path = tmp_path / "timed.txrm"
    fields = {
        "NoOfImages": 3,
        "ImageHeight": 4,
        "ImageWidth": 6,
        "DataType": 5,
        "AcquisitionMode": 0,
    }
    streams = {f"ImageInfo/{k}": np.array([v], "<u4").tobytes() for k, v in fields.items()}

    def stamps(seconds):
        return b"".join(f"01/01/2026 00:00:{s:02d}.000".encode().ljust(40, b"\0") for s in seconds)

    streams.update(
        {
            "ImageInfo/ExpTimes": np.array([1, 2, 3], "<f4").tobytes(),
            "ImageInfo/Date": stamps([1, 5, 9]),
            "MultiReferenceData/TotalRefImages": np.array([2], "<u4").tobytes(),
            "MultiReferenceData/DataType": np.array([10], "<u4").tobytes(),
            "MultiReferenceData/ImageInfo/ExpTimes": np.array([2, 4], "<f4").tobytes(),
            "MultiReferenceData/ImageInfo/Date": stamps([0, 10]),
        }
    )
    streams.update(
        {
            f"ImageData1/Image{i + 1}": np.full((4, 6), v, "<u2").tobytes()
            for i, v in enumerate([55, 150, 285])
        }
    )
    streams.update(
        {
            f"MultiReferenceData/Image{i + 1}": np.full((4, 6), v, "<f4").tobytes()
            for i, v in enumerate([200, 800])
        }
    )
    write_ole(path, streams)
    geometry = tj.ParallelGeometry(tj.Grid(6, 6, 4, 1, 1, 1), tj.Detector(6, 4, 1, 1), [0, 90, 180])
    frames = tj.load_frames(path, geometry=geometry, darks=0)
    np.testing.assert_array_equal(frames.flat_positions, [-1, 9])
    np.testing.assert_array_equal(frames.view_positions, [0, 4, 8])
    np.testing.assert_allclose(frames.corrected().projections, -np.log(0.5), atol=2e-7)
    reference = tmp_path / "reference.xrm"
    fields.update(NoOfImages=1, DataType=10, AcquisitionMode=14)
    separate = {f"ImageInfo/{k}": np.array([v], "<u4").tobytes() for k, v in fields.items()}
    separate.update(
        {
            "ImageInfo/ExpTimes": np.array([4], "<f4").tobytes(),
            "ImageData1/Image1": np.full((4, 6), 800, "<f4").tobytes(),
        }
    )
    write_ole(reference, separate)
    explicit = tj.load_frames(path, geometry=geometry, flats=reference, darks=0).corrected()
    expected = -np.log(np.array([55, 150, 285]) / (np.array([1, 2, 3]) * 200))
    np.testing.assert_allclose(explicit.projections[:, 0, 0], expected, atol=2e-7)
    streams["ImageInfo/ExpTimes"] = np.array([1, 2], "<f4").tobytes()
    write_ole(path, streams)
    with pytest.raises(ValueError, match="2 exposure values for 3 stored images"):
        tj.load_frames(path, geometry=geometry, darks=0)


@pytest.mark.parametrize("cone", [True, False])
def test_a_supplied_centre_keeps_the_recorded_shifts_as_a_written_one_does(tmp_path, cone):
    # A raw scan has per-view shifts but no CenterShift (only a reconstruction setup
    # writes it). detector= supplies the centre the shifts are measured from:
    # -pitch x CenterShift for the file that records it.
    counts = (500 + np.arange(4 * 8 * 12)).astype("<u2").reshape(4, 8, 12)
    fields = {
        "ImageInfo/NoOfImages": ("<u4", [4]),
        "ImageInfo/ImageWidth": ("<u4", [12]),
        "ImageInfo/ImageHeight": ("<u4", [8]),
        "ImageInfo/DataType": ("<u4", [5]),
        "ImageInfo/AcquisitionMode": ("<u4", [0]),
        "ImageInfo/XrayGeometry": ("<u4", [int(cone)]),
        "ImageInfo/pixelsize": ("<f4", [10]),
        "ImageInfo/Angles": ("<f4", [0, 45, 90, 135]),
        "ImageInfo/StoRADistance": ("<f4", [-40] * 4),
        "ImageInfo/DtoRADistance": ("<f4", [20] * 4),
        "alignment/x-shifts": ("<f4", [0.25, 0.5, -0.75, 1.5]),
        "alignment/y-shifts": ("<f4", [-0.5, 0.25, 0.0, 1.0]),
        "AutoRecon/CenterShift": ("<f4", [1.25]),
        "ReferenceData/DataType": ("<u4", [10]),
    }
    streams = {key: np.asarray(values, dtype).tobytes() for key, (dtype, values) in fields.items()}
    streams.update({f"ImageData1/Image{i + 1}": value.tobytes() for i, value in enumerate(counts)})
    streams["ReferenceData/Image"] = np.full((8, 12), 2500, "<f4").tobytes()
    written, raw = tmp_path / "written.txrm", tmp_path / "raw.txrm"
    write_ole(written, streams)
    write_ole(raw, {k: v for k, v in streams.items() if k != "AutoRecon/CenterShift"})
    pitch = 0.015 if cone else 0.01
    recorded = tj.load_frames(written, darks=0).corrected()
    supplied = tj.load_frames(
        raw, detector=tj.Detector(12, 8, pitch, pitch, (-pitch * 1.25, 0.0)), darks=0
    ).corrected()
    assert type(supplied.geometry) is type(recorded.geometry)
    assert supplied.grid == recorded.grid
    g = recorded.grid
    volume = np.random.default_rng(5).uniform(0, 1, (g.nz, g.ny, g.nx)).astype(np.float32)
    images = [np.asarray(tj.project(s.geometry, volume)) for s in (recorded, supplied)]
    assert np.abs(images[0]).max() > 0
    np.testing.assert_allclose(images[1], images[0], rtol=1e-6, atol=1e-6)

    # An explicit geometry is used as given; the shifts it replaces are named.
    with pytest.warns(UserWarning, match=r"4 of 4 views carry recorded detector shifts.*detector="):
        tj.load_frames(raw, geometry=recorded.geometry, darks=0)


@pytest.mark.parametrize("cone", [True, False])
def test_zeiss_vendor_shifts_follow_the_poses_switch(tmp_path, cone):
    # The file's per-view shifts and centre shift are the vendor's corrections: applied by
    # default, left out by poses=False, and named in the record either way.
    path = tmp_path / "scan.txrm"
    counts = (500 + np.arange(3 * 8 * 12)).astype("<u2").reshape(3, 8, 12)
    streams = _streams(counts, np.full((8, 12), 2500, "<f4"))
    streams["alignment/x-shifts"] = np.array([0.25, 0.5, -0.25], "<f4").tobytes()
    streams["ImageInfo/XrayGeometry"] = np.array([int(cone)], "<u4").tobytes()
    write_ole(path, streams)
    for poses in (True, False):
        scan = tj.load(path, darks=0, poses=poses)
        vendor = scan.source.geometry_metadata["vendor_corrections"]
        assert vendor["source"] == "zeiss" and vendor["applied"] is poses
        assert set(vendor["fields"]) == {
            "alignment/x-shifts",
            "alignment/y-shifts",
            "AutoRecon/CenterShift",
        }
        if cone:
            moved = isinstance(scan.geometry, tj.geometry.ConeSegments)
            centres = [s.detector.center for s in scan.geometry.segments] if moved else []
            assert moved is poses
            assert poses or scan.detector.center == (0.0, 0.0)
            assert not poses or len({c[0] for c in centres}) == 3
        else:
            shifts = np.asarray(stack_view_poses(scan.geometry, 3))[:, 0, 3]
            assert bool(np.ptp(shifts) > 0) is poses
            assert bool(scan.detector.center[0] != 0) is poses


def test_a_cone_scan_without_a_centre_shift_loads_with_the_middle_as_its_centre(tmp_path):
    # Only a reconstruction setup writes AutoRecon/CenterShift, so a raw scan (CuO_2 tomo-A)
    # lacks it; the detector was then never built and tj.load asked for the pixel size the
    # file records. As for every format: the middle is the centre, said, nothing estimated.
    counts = (500 + np.arange(3 * 8 * 12)).astype("<u2").reshape(3, 8, 12)
    streams = _streams(counts, np.full((8, 12), 2500, "<f4"))
    write_ole(tmp_path / "with.txrm", streams)
    recorded = tj.load(tmp_path / "with.txrm", darks=0)
    del streams["AutoRecon/CenterShift"]
    write_ole(tmp_path / "without.txrm", streams)
    scan = tj.load(tmp_path / "without.txrm", darks=0)
    meta = scan.source.geometry_metadata
    assert meta["detector_center_status"] == "not_recorded"
    assert meta["zeiss"]["center_shift_pixels"] is None
    assert scan.detector.du == pytest.approx(0.015)  # 10 um at the sample, magnified 1.5
    # The recorded shifts (0.25, -0.5 px) are applied from the middle: 0.25 x 0.015 mm.
    np.testing.assert_allclose(scan.detector.center, [0.00375, -0.0075])
    # The centre shift (0.125 px) moves the centre by its pitch and changes nothing else.
    np.testing.assert_allclose(
        np.subtract(recorded.detector.center, scan.detector.center), [-0.125 * 0.015, 0]
    )
    np.testing.assert_array_equal(scan.projections, recorded.projections)
    np.testing.assert_array_equal(scan.angles, recorded.angles)


@pytest.mark.parametrize("flag", [1, 0, None])
def test_a_dark_the_instrument_removed_is_declared_by_the_file(tmp_path, flag):
    # Every Zeiss scan on disk records AcquisitionSettings/CameraStaticBackground = 1: the
    # camera's static background (its dark) was subtracted before the counts were written,
    # and no dark is stored. Where the file says so, no darks= is needed and the record
    # names the stream; where it does not, darks= is required as before.
    counts = (500 + np.arange(3 * 8 * 12)).astype("<u2").reshape(3, 8, 12)
    streams = _streams(counts, np.full((8, 12), 2500, "<f4"))
    if flag is not None:
        streams["AcquisitionSettings/CameraStaticBackground"] = np.array([flag], "<i4").tobytes()
    path = tmp_path / "scan.txrm"
    write_ole(path, streams)
    zero = tj.load(path, darks=0)
    if flag != 1:
        with pytest.raises(ValueError, match="No dark fields"):
            tj.load(path)
        return
    scan = tj.load(path)
    np.testing.assert_array_equal(scan.projections, zero.projections)
    (fields,) = [c for c in scan.corrections if c.name == "flat_dark"]
    assert fields.settings["darks"] == 0
    assert "AcquisitionSettings/CameraStaticBackground" in fields.settings["dark_removed_by"]
    (given,) = [c for c in tj.load(path, darks=100).corrections if c.name == "flat_dark"]
    assert "dark_removed_by" not in given.settings and given.settings["darks"] == 1


@pytest.mark.parametrize("limit", [700, 1000, None])
def test_counts_above_the_cameras_stated_linear_range_are_reported_not_changed(
    tmp_path, limit, caplog
):
    # Every Zeiss scan here states MaxLinearRange = 45000, and 21 of 84 have view counts above
    # it (the references are held inside it, the views are not; saturated air then reads as
    # absorbing). The record reports the range, its source, the fraction of counts above it
    # and the line integral where it ends; no value changes.
    import logging

    counts = (500 + np.arange(3 * 8 * 12)).astype("<u2").reshape(3, 8, 12)  # 500 ... 787
    streams = _streams(counts, np.full((8, 12), 2500, "<f4"))
    stream = "ConfigureBackup/ConfigCamera/Camera 1/MaxLinearRange"
    path = tmp_path / "scan.txrm"
    write_ole(path, streams)
    plain = tj.load(path, darks=0)
    if limit is not None:
        streams[stream] = np.array([limit], "<i4").tobytes()
    write_ole(path, streams)
    with caplog.at_level(logging.WARNING, logger="tomojax"):
        scan = tj.load(path, darks=0)
    np.testing.assert_array_equal(scan.projections, plain.projections)
    (fields,) = [c for c in scan.corrections if c.name == "flat_dark"]
    if limit is None:
        assert "linear_limit" not in fields.settings and not caplog.records
        return
    assert fields.settings["linear_limit"] == limit
    assert fields.settings["linear_limit_source"] == stream
    above = float(np.mean(counts > limit))
    assert fields.found["counts_above_linear_limit"] == pytest.approx(above)
    # Counts at the limit over the 2500-count flat, at equal exposure: -log(limit / 2500).
    assert fields.found["linear_limit_line_integral"] == pytest.approx(-np.log(limit / 2500))
    warned = [r.getMessage() for r in caplog.records if "linear range" in r.getMessage()]
    assert bool(warned) == (above > 0)


def test_reconstructing_in_slabs_reports_the_same_counts_above_the_linear_range(tmp_path):
    counts = (500 + np.arange(3 * 8 * 12)).astype("<u2").reshape(3, 8, 12)
    streams = _streams(counts, np.full((8, 12), 2500, "<f4"))
    streams["ConfigureBackup/ConfigCamera/Camera 1/MaxLinearRange"] = np.array(
        [700], "<i4"
    ).tobytes()
    write_ole(tmp_path / "scan.txrm", streams)
    with tj.reconstruct_slabs(tj.load_frames(tmp_path / "scan.txrm", darks=0), "fbp") as slabs:
        for _, part in slabs:
            (fields,) = [c for c in part.scan.corrections if c.name == "flat_dark"]
            assert fields.settings["linear_limit"] == 700
            assert fields.found["counts_above_linear_limit"] == pytest.approx(87 / 288)
            assert fields.found["linear_limit_line_integral"] == pytest.approx(-np.log(700 / 2500))


def test_an_interrupted_scan_loads_its_stored_views_with_their_shifts(tmp_path):
    # An interrupted Zeiss acquisition (CuO_6 tomo-B: 201 views planned, 2 stored) records
    # angles, exposures and dates for the stored views only, but its shift tables at the
    # planned length, the stored views first. The reader failed on a NumPy broadcast.
    counts = (500 + np.arange(3 * 8 * 12)).astype("<u2").reshape(3, 8, 12)
    streams = _streams(counts, np.full((8, 12), 2500, "<f4"), declared_count=10)
    planned = np.r_[[0.25, 0.5, -0.25], np.zeros(7)].astype("<f4")
    streams["alignment/x-shifts"] = planned.tobytes()
    streams["alignment/y-shifts"] = np.r_[[-0.5] * 3, np.zeros(7)].astype("<f4").tobytes()
    write_ole(tmp_path / "scan.txrm", streams)
    frames = tj.load_frames(tmp_path / "scan.txrm", darks=0)
    zeiss = frames.source.geometry_metadata["zeiss"]
    assert (zeiss["declared_view_count"], zeiss["stored_view_count"]) == (10, 3)
    assert frames.views == 3
    np.testing.assert_array_equal(frames.geometry.angles, [0, 90, 180])
    centres = [s.detector.center for s in frames.geometry.segments]  # pitch 0.015, shift 0.125
    np.testing.assert_allclose([c[0] for c in centres], [0.001875, 0.005625, -0.005625])
    # A shift table of any other length is not the planned one: say what the file holds.
    streams["alignment/x-shifts"] = planned[:5].tobytes()
    write_ole(tmp_path / "scan.txrm", streams)
    with pytest.raises(ValueError, match="5 x-shifts for 3 stored images"):
        tj.load_frames(tmp_path / "scan.txrm", darks=0)


def test_the_grid_derived_from_the_detector_follows_binning(tmp_path):
    # The reader put the grid it derives from the detector (image size, recorded pixel size)
    # on the record as if the file stated it, so binning kept the full-size grid: CuO_2's
    # corrected(binning=4) reconstructed 2028^3 voxels. It is derived, as Nikon's is.
    counts = (500 + np.arange(3 * 8 * 12)).astype("<u2").reshape(3, 8, 12)
    write_ole(tmp_path / "scan.txrm", _streams(counts, np.full((8, 12), 2500, "<f4")))
    frames = tj.load_frames(tmp_path / "scan.txrm", darks=0)
    assert frames.geometry.grid == tj.Grid(12, 12, 8, 0.01, 0.01, 0.01)  # as before
    meta = frames.source.geometry_metadata
    assert meta["grid_status"] == "derived" and meta["voxel_size"] == pytest.approx([0.01] * 3)
    binned = frames.corrected(binning=4)
    assert (binned.grid.nx, binned.grid.ny, binned.grid.nz) == (3, 3, 2)
    assert (binned.grid.vx, binned.grid.vy, binned.grid.vz) == pytest.approx((0.04, 0.04, 0.04))
