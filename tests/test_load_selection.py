"""All consumers of a load request use its selected measurements and acquisition state."""

import inspect

import h5py
import numpy as np
import pytest
import tifffile

import tomojax as tj

# check-public-imports: allow-private
from tomojax._loading.loader import projection_frames
from tomojax.io.dataset_paths import DATASET_PATHS

from .test_esrf_readers import _edf_scan, _write_bliss_sequence
from .test_psi_reader import _write_v73
from .test_zeiss import _streams, write_ole

pytestmark = pytest.mark.surface

HDF = {"saved.nxs", "nxtomo", "nexus", "data_exchange"}


def write_scan(path, *, views=3, pixels=1):
    grid = tj.Grid(4, 4, 2, 1, 1, 1)
    detector = tj.Detector(4, 2, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, np.arange(views) * 60.0)
    poses = np.zeros((views, 6))
    poses[:, 3] = np.arange(views) * 0.2
    scan = tj.Scan(
        np.full((views, 2, 4), pixels, np.float32),
        geometry,
        view_positions=np.arange(views) ** 2 + 0.5,
    )
    tj.save(path, scan.with_poses(poses))


def counts_file(path, dtype):
    with h5py.File(path, "w") as file:
        counts = np.array([51, 76, 26], dtype)[:, None, None]
        file["entry/data/projections"] = np.broadcast_to(counts, (3, 2, 4))
        file["entry/sample/rotation_angle"] = [0, 60, 120]


def test_explicit_calibration_has_the_same_domain_in_every_storage_dtype(tmp_path):
    expected = -np.log([0.50, 0.75, 0.25])
    records = []
    for dtype in (np.float32, np.uint16):
        path = tmp_path / "counts.nxs"
        counts_file(path, dtype)
        options = {"flats": 101.0, "darks": 1.0}
        lazy, raw = projection_frames(path, **options)
        assert raw
        for scan in (
            tj.load(path, **options),
            tj.load_frames(path, **options).corrected(),
            lazy.corrected(),
        ):
            np.testing.assert_allclose(scan.projections[:, 0, 0], expected, rtol=1e-6)
            assert [c.name for c in scan.corrections] == ["flat_dark", "log"]
            records.append(scan.corrections)
        tj.save(tmp_path / "logged.nxs", scan)
        for load in (tj.load, tj.load_frames, projection_frames):
            with pytest.raises(ValueError, match="flats.*line integrals"):
                load(tmp_path / "logged.nxs", **options)
    assert all(record == records[0] for record in records)


def selected_file(path):
    write_scan(path)
    with h5py.File(path, "a") as file:
        file["alternative/data"] = np.full((3, 2, 4), 7.0, np.float32)
        file["alternative/angles"] = [15.0, 75.0, 135.0]
        file["alternative/angles"].attrs["units"] = "degree"
        file["alternative/key"] = [0, 3, 0]


@pytest.mark.parametrize("data", [True, False])
@pytest.mark.parametrize("angles", [True, False])
@pytest.mark.parametrize("key", [True, False])
def test_saved_selections_control_eager_and_lazy_pixels_angles_and_views(
    tmp_path, data, angles, key
):
    path = tmp_path / "selected.nxs"
    selected_file(path)
    options = {
        "data_path": "/alternative/data" if data else None,
        "angles_path": "/alternative/angles" if angles else None,
        "image_key_path": "/alternative/key" if key else None,
    }
    kept = [0, 2] if key else [0, 1, 2]
    expected_angles = np.array([0, 60, 120])[kept] + (15 if angles else 0)
    scan = tj.load(path, **options)
    lazy, raw = projection_frames(path, **options)
    np.testing.assert_array_equal(lazy.view_positions, np.array([0.5, 1.5, 4.5])[kept])
    assert not raw and not isinstance(lazy.counts, np.ndarray)
    for pixels, geometry in ((scan.projections, scan.geometry), (lazy.counts, lazy.geometry)):
        np.testing.assert_array_equal(geometry.angles, expected_angles)
        np.testing.assert_array_equal(pixels[:], np.full((len(kept), 2, 4), 7 if data else 1))
    np.testing.assert_allclose(scan.poses[:, 3], np.array([0.0, 0.2, 0.4])[kept])


def tiff_family(root, family, counts):
    for i, image in enumerate(counts):
        tifffile.imwrite(root / f"part_{i + 1:04d}.tif", image.astype(np.uint16))
    if family == "nikon_xtekct":
        path = root / "part.xtekct"
        path.write_text(
            "[XTekCT]\nName=part\nDetectorOffsetX=0\nDetectorOffsetY=0\n"
            "CentreOfRotationTop=0\nCentreOfRotationBottom=0\nSrcToObject=12\n"
            "SrcToDetector=18\nDetectorPixelSizeX=1\nDetectorPixelSizeY=1\n"
            "WhiteLevel=101\nInitialAngle=0\nAngularStep=60\n"
        )
    elif family == "skyscan":
        path = root / "part.log"
        path.write_text(
            "[System]\nScanner=Skyscan1172\nCamera Pixel Size (um)=20\n[Acquisition]\n"
            "Filename Prefix=part_\nNumber of Files=3\nNumber of Rows=2\nNumber of Columns=4\n"
            "Camera binning=1x1\nImage Pixel Size (um)=20\nObject to Source (mm)=20\n"
            "Camera to Source (mm)=40\nRotation Step (deg)=60\nRotation Direction=CC\n"
            "Geometrical Correction=ON\nFlat Field Correction=ON\n"
        )
    else:
        path = root
    return path


def input_family(root, family):
    """Build the file and its expected sample pixels in logical detector order."""
    path = root / "input.h5"
    counts = np.broadcast_to(np.array([51, 76, 26], np.float32)[:, None, None], (3, 2, 4))
    if family.startswith("saved"):
        path = root / family
        write_scan(path, pixels=51)
        counts = np.full_like(counts, 51)
    elif family == "phase":
        path = root / "input.mat"
        _write_v73(path, counts, np.arange(3) * 60, 1e-5)
    elif family == "zeiss_ole":
        path = root / "input.txrm"
        counts = np.full((3, 8, 12), 51, "<u2")
        write_ole(path, _streams(counts, np.full((8, 12), 101, "<f4")))
    elif family == "esrf_bliss":
        counts = _write_bliss_sequence(path, "measurement/hrsrot")
    elif family == "esrf_edf":
        path = _edf_scan(root)
        counts = np.stack(
            [
                np.full((3, 5), 20.25 + (100 + 5 * (i + 0.5)) * np.exp(-0.2), np.float32)
                for i in range(4)
            ]
        )
    elif family in {"nikon_xtekct", "tiff", "skyscan"}:
        path = tiff_family(root, family, counts)
    elif family in HDF:
        with h5py.File(path, "w") as file:
            if family == "data_exchange":
                file["exchange/data"] = counts
                file["exchange/theta"] = [0, 60, 120]
            elif family == "nxtomo":
                file["entry/definition"] = "NXtomo"
                file["entry/instrument/detector/data"] = counts
                file["entry/instrument/detector/image_key"] = [0, 0, 0]
                file["entry/sample/rotation_angle"] = [0, 60, 120]
            elif family == "nexus":
                entry = file.create_group("experiment")
                file.attrs["default"] = "experiment"
                entry.attrs.update(NX_class="NXentry", default="plot")
                plot = entry.create_group("plot")
                plot.attrs.update(NX_class="NXdata", signal="pixels")
                plot["pixels"] = counts
    else:
        pytest.fail(f"Add a load-contract fixture for {family}")
    return path, counts


@pytest.mark.parametrize("family", sorted(DATASET_PATHS))
def test_load_arguments_by_input_family(tmp_path, family):
    path, counts = input_family(tmp_path, family)
    n, rows, cols = counts.shape
    angles = np.arange(n) * 11 + 2.5
    detector = tj.Detector(cols, rows, 0.4, 0.8, (0.01, -0.02))
    grid = tj.Grid(5, 4, 3, 0.2, 0.3, 0.4)
    geometry = tj.ParallelGeometry(grid, detector, angles)
    fields = {
        "flats": np.full((rows, cols), 101.0),
        "darks": 1.0,
        "exposure_times": 2.0,
        "flat_exposure_times": 1.0,
    }
    overrides = {"angles": angles, "detector": detector, "grid": grid, "apply_poses": False}
    selectors = {"data_path", "angles_path", "image_key_path"}
    covered = set(fields) | set(overrides) | selectors | {"geometry", "pixel_size", "path"}
    assert covered == set(inspect.signature(tj.load).parameters)
    assert covered == set(inspect.signature(tj.load_frames).parameters)

    if family == "phase":
        for arg, value in fields.items():
            for load in (projection_frames, tj.load_frames, tj.load):
                with pytest.raises(ValueError, match=arg):
                    load(path, **{arg: value})
        fields = {}
    expected = counts if family == "phase" else -np.log((counts - 1) / 200)
    for options in (
        overrides,
        {"angles": angles, "pixel_size": (0.4, 0.8), "grid": grid, "apply_poses": False},
        {"geometry": geometry, "apply_poses": False},
    ):
        scan = tj.load(path, **fields, **options)
        lazy, raw = projection_frames(path, **fields, **options)
        assert raw == (family != "phase")
        np.testing.assert_allclose(scan.projections, expected, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(lazy.counts[:], counts)
        for geom in (scan.geometry, lazy.geometry):
            np.testing.assert_array_equal(geom.angles, angles)
            assert geom.grid == grid and (geom.detector.du, geom.detector.dv) == (0.4, 0.8)
        assert scan.poses is None
        if raw:
            frames = tj.load_frames(path, **fields, **options)
            np.testing.assert_array_equal(frames.flats[0], fields["flats"])
            assert frames.exposure_times == 2 and frames.flat_exposure_times == 1
    if family in HDF:
        check_alternative_datasets(path, counts.shape, fields, detector)
    for name in selectors:
        if family == "esrf_bliss" and name == "data_path":
            np.testing.assert_array_equal(tj.load_frames(path, data_path="/1.1").counts[:], counts)
        else:
            for load in (tj.load, tj.load_frames, projection_frames):
                with pytest.raises((ValueError, KeyError), match=name):
                    load(path, **fields, **{name: "/entry/absent"})


def check_alternative_datasets(path, shape, fields, detector):
    n, rows, cols = shape
    with h5py.File(path, "a") as file:
        file["aux/another/samples"] = np.full((n, rows, cols), 31.0)
        file["aux/another/angles"] = np.arange(n) * 5 + 1
        file["aux/another/angles"].attrs["units"] = "degree"
        file["aux/another/key"] = [3, *[0] * (n - 1)]
    paths = {
        "data_path": "/aux/another/samples",
        "angles_path": "/aux/another/angles",
        "image_key_path": "/aux/another/key",
    }
    for load in (tj.load, tj.load_frames, projection_frames):
        result = load(path, **fields, **paths, detector=detector)
        item = result[0] if isinstance(result, tuple) else result
        np.testing.assert_array_equal(item.geometry.angles, np.arange(1, n) * 5 + 1)
        if isinstance(item, tj.Frames):
            np.testing.assert_array_equal(item.counts[:], 31.0)
        else:
            np.testing.assert_allclose(item.projections, -np.log(30 / 200), rtol=1e-6)


def test_correct_cli_and_mcp_worker_use_the_selected_frames(tmp_path):
    from tomojax.cli.main import main

    from .test_mcp_stdio import connection

    path = tmp_path / "counts.nxs"
    selected_file(path)
    with h5py.File(path, "a") as file:
        file["alternative/data"][:, :, :] = np.array([51, 76, 26])[:, None, None]
    paths = {"data_path": "/alternative/data", "angles_path": "/alternative/angles"}
    output = tmp_path / "cli.nxs"
    assert (
        main(
            [
                "correct",
                str(path),
                "-o",
                str(output),
                "--flats",
                "101",
                "--darks",
                "1",
                "--data-path",
                paths["data_path"],
                "--angles-path",
                paths["angles_path"],
            ]
        )
        == 0
    )
    assert (
        main(["correct", str(output), "-o", str(tmp_path / "again.nxs"), "--epsilon", "0.01"]) == 1
    )
    with connection(tmp_path) as client:
        for kind in ("scan", "frames"):
            options = {"path": path.name, "kind": kind, "flats": 101, "darks": 1, **paths}
            plan = client.call("open", **options, dry_run=True)
            assert not plan.get("isError"), plan
            opened = client.call("open", **options)["structuredContent"]["handle"]
            if kind == "frames":
                opened = client.call("correct", source=opened)["structuredContent"]["handle"]
            saved = client.call("save", source=opened, path=f"{kind}.nxs")
            assert not saved.get("isError"), saved
        for dry_run in (True, False):
            error = client.call(
                "open", path=path.name, kind="reconstruction", apply_poses=True, dry_run=dry_run
            )
            assert error.get("isError") and "apply_poses" in error["content"][0]["text"]
            error = client.call("open", path=path.name, angles_path="/missing", dry_run=dry_run)
            assert error.get("isError") and "angles_path" in error["content"][0]["text"]
    for saved in (output, tmp_path / "frames.nxs", tmp_path / "scan.nxs"):
        scan = tj.load(saved)
        np.testing.assert_allclose(scan.projections[:, 0, 0], -np.log([0.5, 0.75, 0.25]), rtol=1e-6)
        np.testing.assert_array_equal(scan.angles, [15, 75, 135])


def test_inspection_and_saved_results_reject_scan_options(tmp_path):
    import msgspec

    from tomojax.cli import operations
    from tomojax.cli.mcp import tools

    options = set(inspect.signature(tj.load).parameters)
    assert options == {f.name for f in msgspec.structs.fields(operations.Open)} - {"kind"}
    assert options == {f.name for f in msgspec.structs.fields(tools.Open)} - {
        "kind",
        "dry_run",
        "detail",
    }
    for load in (tj.inspect, tj.load_reconstruction):
        for name in options - {"path"}:
            with pytest.raises(TypeError, match=name):
                load(tmp_path, **{name: True})
