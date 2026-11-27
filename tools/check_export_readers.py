"""Read freshly exported volumes with ImageJ, Bio-Formats, ITK and VTK."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tarfile
import tempfile
from urllib.request import urlretrieve

import jpype
import numpy as np
import SimpleITK as sitk
import vtk

from check_big_tiff import big_stacks
import tomojax as tj

JAVA_FILES = {
    "ij.jar": (
        "https://repo.maven.apache.org/maven2/net/imagej/ij/1.54p/ij-1.54p.jar",
        "2e1a09961dfb41cee66ddc821b2577a41a072566ce45a49bae69267099741e20",
    ),
    "bioformats_package.jar": (
        "https://downloads.openmicroscopy.org/bio-formats/8.5.0/artifacts/bioformats_package.jar",
        "c6e60665d53a334b66e4d635340151f403dfe57a64704c573dd4c03b873befb9",
    ),
    "jre.tar.gz": (
        "https://github.com/adoptium/temurin21-binaries/releases/download/"
        "jdk-21.0.13.1%2B2/OpenJDK21U-jre_x64_linux_hotspot_21.0.13.1_2.tar.gz",
        "18f44fab9065217f3ba61d0eef8a7b5f3c3713aa7293001b0d4ac10ede3850f0",
    ),
}


def java_readers(root: Path) -> None:
    """Start a JVM with the pinned ImageJ and Bio-Formats distributions."""
    root.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in JAVA_FILES.items():
        path = root / name
        if not path.exists():
            with tempfile.TemporaryDirectory(dir=root) as staging:
                downloaded = Path(staging) / name
                urlretrieve(url, downloaded)
                with downloaded.open("rb") as f:
                    if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                        raise ValueError(f"unexpected digest downloading {name}")
                downloaded.replace(path)
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                raise ValueError(f"unexpected digest for {path}")
    jvm = root / "jdk-21.0.13.1+2-jre/lib/server/libjvm.so"
    if not jvm.exists():
        with tarfile.open(root / "jre.tar.gz") as tar:
            tar.extractall(root, filter="data")
    jpype.startJVM(
        str(jvm),
        "-Djava.awt.headless=true",
        "-Xmx512m",
        classpath=[str(root / "ij.jar"), str(root / "bioformats_package.jar")],
    )
    jpype.JClass("loci.common.DebugTools").setRootLevel("ERROR")


def indexed(nx: int = 11, unit: str = "mm") -> tj.Reconstruction:
    """Create a calibrated volume with a distinct value at every voxel."""
    z, y, x = np.indices((5, 7, nx))
    grid = tj.Grid(nx, 7, 5, 0.0015, 0.002, 0.003, vol_origin=(-0.0075, -0.006, -0.006))
    geom = tj.ParallelGeometry(grid, tj.Detector(nx, 5, 1, 1), np.arange(3.0))
    return tj.Reconstruction(
        (10000 * z + 100 * y + x).astype("f4"), geom, "fbp", {}, length_unit=unit
    )


def fiji(path: Path) -> jpype.JObject:
    """Open an ImagePlus through the Bio-Formats importer used by Fiji."""
    options = jpype.JClass("loci.plugins.in.ImporterOptions")()
    options.setId(str(path))
    options.setQuiet(True)
    options.setWindowless(True)
    options.setVirtual(True)
    return jpype.JClass("loci.plugins.BF").openImagePlus(options)[0]


def imagej_and_bioformats(root: Path) -> None:
    """Check the calibration and decoded samples a Fiji user reads."""
    source = indexed()
    for suffix in (".tif", ".ome.tif"):
        path = root / ("volume" + suffix)
        tj.export(path, source, dtype="uint16", range=(-4, 32763.5), compress=True)
        if suffix == ".tif":
            imp = jpype.JClass("ij.IJ").openImage(str(path))
            c = imp.getCalibration()
            np.testing.assert_allclose((c.pixelWidth, c.pixelHeight, c.pixelDepth), (1.5, 2.0, 3.0))
            np.testing.assert_allclose((c.xOrigin, c.yOrigin, c.zOrigin), (5, 3, 2))
            assert c.getUnit() in ("micron", "µm")
            assert c.getValueUnit() == "1/mm"
            assert c.getCValue(imp.getStack().getProcessor(4).get(9, 5)) == 30509
            imp.close()
        imp = fiji(path)
        c = imp.getCalibration()
        assert (imp.getWidth(), imp.getHeight(), imp.getNSlices()) == (11, 7, 5)
        assert c.getUnit() in ("micron", "µm")
        np.testing.assert_allclose((c.pixelWidth, c.pixelHeight, c.pixelDepth), (1.5, 2.0, 3.0))
        assert (c.xOrigin, c.yOrigin, c.zOrigin) == (0, 0, 0)
        assert c.getCValue(42.0) == 42.0
        assert imp.getStack().getProcessor(4).get(9, 5) == 61026
        imp.close()
    reader = jpype.JClass("loci.formats.ImageReader")()
    metadata = jpype.JClass("loci.formats.MetadataTools").createOMEXMLMetadata()
    reader.setMetadataStore(metadata)
    for unit in ("mm", "pixel"):
        path = root / (unit + ".am")
        tj.export(path, indexed(unit=unit))
        reader.setId(str(path))
        assert (reader.getSizeX(), reader.getSizeY(), reader.getSizeZ()) == (11, 7, 5)
        assert np.frombuffer(bytes(reader.openBytes(3, 9, 5, 1, 1)), "<f4")[0] == 30509
        um = jpype.JClass("ome.units.UNITS").MICROMETRE
        scales = [
            float(getattr(metadata, "getPixelsPhysicalSize" + axis)(0).value(um)) for axis in "XYZ"
        ]
        np.testing.assert_allclose(scales, (1.5, 2, 3) if unit == "mm" else (1, 1, 1), rtol=1e-5)
        reader.close()


def metaimage(root: Path) -> None:
    """Check narrow and re-stacked volumes in two independent readers."""
    for width in (11, 8, 1):
        for axis in "zyx":
            source = indexed(width)
            path = root / f"{axis}{width}.mhd"
            tj.export(path, source, axis=axis)
            image = sitk.ReadImage(str(path))
            i = (min(9, width - 1), 5, 3)
            order = ["zyx".index(a) for a in (axis + "zyx".replace(axis, ""))]
            xyz = list(reversed([tuple(reversed(i))[k] for k in order]))
            expected = 30500 + i[0]
            assert image.GetPixel(*xyz) == expected
            np.testing.assert_allclose(
                image.TransformIndexToPhysicalPoint(xyz),
                np.array([-0.0075, -0.006, -0.006]) + np.array(i) * [0.0015, 0.002, 0.003],
            )
            reader = vtk.vtkMetaImageReader()
            reader.SetFileName(str(path))
            reader.Update()
            volume = reader.GetOutput()
            assert volume.GetDimensions() == image.GetSize()
            assert volume.GetScalarComponentAsDouble(*xyz, 0) == expected
            np.testing.assert_allclose(volume.GetSpacing(), image.GetSpacing())
            np.testing.assert_allclose(volume.GetOrigin(), (-0.0075, -0.006, -0.006))


def read_big(path: Path, indices: list[int], expected: np.ndarray) -> None:
    """Read across 32-bit offsets through Fiji's virtual stack."""
    image = fiji(path)
    try:
        calibration = image.getCalibration()
        np.testing.assert_allclose(
            (calibration.pixelWidth, calibration.pixelHeight, calibration.pixelDepth), (1.5, 2, 3)
        )
        assert calibration.getUnit() in ("micron", "µm")
        stack = image.getStack()
        actual = [stack.getProcessor(i + 1).getf(3, 2) for i in indices]
        np.testing.assert_array_equal(actual, expected)
    finally:
        image.close()


def native_nexus(root: Path) -> None:
    """Read the saved default plot, physical axes and detector through two NeXus clients."""
    from dataclasses import replace

    import h5py
    from nxtomo.application.nxtomo import NXtomo
    from silx.io.nxdata.parse import get_default

    source = indexed()
    frames = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
    detector = tj.Detector(4, 2, 0.042, 0.051)
    geometry = tj.ParallelGeometry(source.grid, detector, [0.0, 37.0, 82.0])
    result = replace(source, scan=tj.Scan(frames, geometry), geometry=geometry)
    path = root / "saved.nxs"
    tj.save(path, result, include_scan=True)
    with h5py.File(path) as file:
        plot = get_default(file)
        assert plot is not None
        assert isinstance(plot.signal, h5py.Dataset)
        assert plot.group.name == "/entry/volume"
        assert plot.signal[3, 5, 9] == 30509.0
        assert [axis.name for axis in plot.axes] == [f"/entry/volume/{name}" for name in "zyx"]
        np.testing.assert_allclose([axis[0] for axis in plot.axes], [-0.006, -0.006, -0.0075])
        np.testing.assert_allclose(
            [axis[1] - axis[0] for axis in plot.axes], [0.003, 0.002, 0.0015]
        )
        assert "definition" not in file["entry"]
        assert file["entry/instrument/detector/data"].attrs["long_name"] == "line integrals"
    scan_plot = root / "projections.h5"
    assert result.scan is not None
    tj.export(scan_plot, result.scan, axis="v")
    with h5py.File(scan_plot) as file:
        plot = get_default(file)
        assert plot is not None
        assert isinstance(plot.signal, h5py.Dataset)
        assert [axis.name.rsplit("/", 1)[-1] for axis in plot.axes] == ["v", "rotation_angle", "u"]
        np.testing.assert_array_equal(plot.axes[1], [0.0, 37.0, 82.0])
        np.testing.assert_array_equal(plot.signal[:], frames.transpose(1, 0, 2))
    acquisition = NXtomo()
    acquisition.load(str(path), "/entry", detector_data_as="as_numpy_array")
    assert acquisition.instrument is not None
    assert acquisition.sample is not None
    det = acquisition.instrument.detector
    assert det is not None
    assert det.x_pixel_size is not None
    assert det.y_pixel_size is not None
    assert acquisition.sample.rotation_angle is not None
    np.testing.assert_array_equal(det.data, frames)
    np.testing.assert_allclose(
        acquisition.sample.rotation_angle.to("degree").magnitude, [0, 37, 82]
    )
    np.testing.assert_allclose(det.x_pixel_size.to("mm").magnitude, 0.042)
    np.testing.assert_allclose(det.y_pixel_size.to("mm").magnitude, 0.051)


def nexus_and_ngff(root: Path) -> None:
    """Read a custom signal through silx and each registered NGFF level through ome-zarr-py."""
    from dataclasses import replace

    import h5py
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader
    from ome_zarr_models import open_ome_zarr
    from silx.io.nxdata.parse import get_default
    import zarr

    source = indexed()
    path = root / "export.h5"
    tj.export(
        f"{path}::/analysis/result/attenuation",
        source,
        dtype="uint16",
        range=(-4, 32763.5),
        compress=True,
    )
    with h5py.File(path) as file:
        plot = get_default(file)
        assert plot is not None
        assert isinstance(plot.signal, h5py.Dataset)
        assert plot.signal.name == "/analysis/result/attenuation"
        np.testing.assert_allclose([axis[0] for axis in plot.axes], [-0.006, -0.006, -0.0075])
        stored = plot.signal[3, 5, 9]
        assert stored == 61026
        offset = plot.group["attenuation_offset"]
        scale = plot.group["attenuation_scaling_factor"]
        assert isinstance(offset, h5py.Dataset)
        assert isinstance(scale, h5py.Dataset)
        assert (stored + offset[()]) * scale[()] == 30509

    grid = tj.Grid(513, 3, 1, 0.5, 2, 3, vol_origin=(1.0, -2.0, 9.0))
    geometry = tj.ParallelGeometry(grid, source.geometry.detector, [0.0])
    volume = np.arange(513 * 3, dtype=np.float32).reshape(1, 3, 513)
    path = root / "export.ome.zarr"
    tj.export(path, replace(source, geometry=geometry, volume=volume))
    assert open_ome_zarr(zarr.open_group(path, mode="r")) is not None
    location = parse_url(str(path))
    assert location is not None
    image = next(iter(Reader(location)()))
    assert [a["name"] for a in image.metadata["axes"]] == ["z", "y", "x"]
    transforms = image.metadata["coordinateTransformations"]
    assert len(transforms) == len(image.data) == 2
    for fields, scale, origin in zip(
        transforms, ([3, 2, 0.5], [3, 4, 1]), ([9, -2, 1], [9, -1, 1.25]), strict=True
    ):
        np.testing.assert_allclose(fields[0]["scale"], scale)
        np.testing.assert_allclose(fields[1]["translation"], origin)
    np.testing.assert_array_equal(image.data[0], volume)
    np.testing.assert_array_equal(image.data[1][0, 0], 257 + 2 * np.arange(256))


def main() -> None:
    """Verify samples and calibration in independently maintained readers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readers", type=Path, default=Path(".artifacts/readers"))
    args = parser.parse_args()
    java_readers(args.readers)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        imagej_and_bioformats(root)
        metaimage(root)
        native_nexus(root)
        nexus_and_ngff(root)
        big_stacks(root, read_big)
    print("All TIFF, raw, NeXus and OME-Zarr reader checks passed.")


if __name__ == "__main__":
    main()
