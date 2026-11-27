"""Interchange format table and the TIFF and flat binary writers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
import json
import math
import os
from pathlib import Path
import tempfile
from typing import IO, Literal, TypedDict
import xml.etree.ElementTree as ET

import numpy as np
import tifffile

from tomojax.core.host_volume import read_at, write_at
from tomojax.errors import InvalidParameter
from tomojax.io._export_data import Blocks, Selection
from tomojax.io._export_nexus import nexus_record, nexus_settings, write_nexus
from tomojax.io._export_zarr import write_zarr, zarr_record, zarr_settings
from tomojax.wire.arrays import Uniform

type Document = dict[str, object]
type Writer = Callable[[Path, Blocks, Document], None]
_COMMON = ("dtype", "range", "crop", "binning", "axis")
_CM = {"m": 0.01, "cm": 1.0, "mm": 10.0, "um": 10000.0, "µm": 10000.0, "nm": 1.0e7}


def binary(stream: IO[bytes], blocks: Blocks) -> None:
    """Place each slab at its final byte offsets, including transposed slabs."""
    offset = stream.tell()
    shape = blocks.selection.shape
    dtype = np.dtype(blocks.values.dtype)
    for region, block in blocks:
        if region[0].start is not None:
            write_at(
                stream.fileno(),
                np.ascontiguousarray(block),
                offset + region[0].start * math.prod(shape[1:]) * dtype.itemsize,
            )
        else:
            # Moving a spatial axis first puts the source slab on output axis 1.
            for plane, rows in enumerate(block):
                index = (plane * shape[1] + region[1].start) * shape[2]
                write_at(
                    stream.fileno(), np.ascontiguousarray(rows), offset + index * dtype.itemsize
                )


def pages(path: Path, blocks: Blocks) -> Iterator[np.ndarray]:
    if blocks.selection.order[0] == 0:
        for _, data in blocks:
            yield from data
        return
    with tempfile.TemporaryFile(dir=path.parent) as scratch:
        binary(scratch, blocks)
        shape = blocks.selection.shape
        plane = np.empty(shape[1:], np.dtype(blocks.values.dtype).newbyteorder("<"))
        for index in range(shape[0]):
            read_at(scratch.fileno(), plane, index * plane.nbytes)
            yield plane


def uniform_axes(blocks: Blocks) -> tuple[tuple[float, ...], tuple[float, ...]]:
    coords = [axis.coordinates for axis in blocks.selection.axes]
    return (
        tuple(c.origin if isinstance(c, Uniform) else 0.0 for c in coords),
        tuple(c.spacing if isinstance(c, Uniform) else 1.0 for c in coords),
    )


def slice_path(path: Path, i: int) -> Path:
    return path / f"{path.name}_{i:05d}.tif"


def tiff(
    path: Path, blocks: Blocks, record: Document, *, dialect: Literal["imagej", "ome", "slices"]
) -> None:
    shape = blocks.selection.shape
    axes = blocks.selection.axes
    origin, spacing = uniform_axes(blocks)
    unit = axes[-1].unit
    physical = (
        unit in _CM
        and unit == axes[-2].unit
        and all(isinstance(a.coordinates, Uniform) for a in axes[-2:])
    )
    plan = record["export"]
    assert isinstance(plan, dict)
    compression = "lzw" if plan["compress"] else None
    resolution = (_CM[unit] / spacing[2], _CM[unit] / spacing[1]) if physical else (1.0, 1.0)
    text = json.dumps(record, ensure_ascii=True, allow_nan=False)
    stack = pages(path, blocks)
    if dialect == "slices":
        path.mkdir()
        for i, plane in enumerate(stack):
            tifffile.imwrite(
                slice_path(path, i),
                plane,
                photometric="minisblack",
                metadata=None,
                compression=compression,
                resolution=resolution,
                resolutionunit="CENTIMETER" if physical else "NONE",
                maxworkers=1,
            )
        (path / f"{path.name}.json").write_text(text, encoding="utf-8")
        return
    metadata: dict[str, object] = {"axes": "ZYX"}
    if dialect == "imagej":
        metadata.update(Info=text, unit="um" if physical else "pixel")
        if physical:
            # Bio-Formats interprets ImageJ spacings in micrometers even for unit=mm.
            um = 10000 / _CM[unit]
            resolution = (1 / (spacing[2] * um), 1 / (spacing[1] * um))
            if axes[0].unit == unit:
                metadata["spacing"] = spacing[0] * um
            if blocks.selection.order == (0, 1, 2):
                for key, start, step in zip("zyx", origin, spacing, strict=True):
                    metadata[key + "origin"] = -start / step
        if blocks.values.range is not None:
            metadata.update(
                cf=0, c0=blocks.values.offset, c1=blocks.values.scale, vunit=blocks.values.unit
            )
    else:
        metadata["Description"] = text
        if physical:
            for name, axis, pitch in zip("ZYX", axes, spacing, strict=True):
                if axis.unit == unit:
                    metadata["PhysicalSize" + name] = pitch
                    metadata["PhysicalSize" + name + "Unit"] = "µm" if unit == "um" else unit
    tifffile.imwrite(
        path,
        stack,
        shape=shape,
        dtype=blocks.values.dtype,
        kind=dialect,
        bigtiff=plan["bigtiff"],
        byteorder="<",
        photometric="minisblack",
        metadata=metadata,
        compression=compression,
        resolution=resolution,
        maxworkers=1,
        resolutionunit="CENTIMETER" if physical and dialect == "ome" else "NONE",
    )


def sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def raw(path: Path, blocks: Blocks, record: Document, *, dialect: str) -> None:
    selection = blocks.selection
    origin, spacing = uniform_axes(blocks)
    dims = selection.shape[::-1]
    dtype = blocks.values.dtype
    payload = path.with_suffix(".raw") if dialect == "metaimage" else path
    with payload.open("wb", buffering=0) as stream:
        if dialect == "amira":
            unit = selection.axes[0].unit
            known = unit in _CM
            box = []
            for n, c, s in zip(dims, reversed(origin), reversed(spacing), strict=True):
                box.extend((c, c + (n - 1) * s) if known else (0, n - 1))
            label = {"uint8": "byte", "uint16": "ushort", "float32": "float"}[dtype]
            units = f'    Units {{ Coordinates "{unit.replace("µ", "u")}" }}\n' if known else ""
            header = (
                "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1\n\n"
                f"define Lattice {' '.join(map(str, dims))}\n\nParameters {{\n"
                f"    BoundingBox {' '.join(map(str, box))},\n"
                f'    CoordType "uniform"\n{units}}}\n\nLattice {{ {label} Data }} @1\n\n'
                "# Data section follows\n@1\n"
            )
            stream.write(header.encode("ascii"))
        binary(stream, blocks)
    if dialect == "metaimage":
        # MetaIO serializes one world direction per file axis (columns of ITK's matrix).
        directions = np.eye(3, dtype=int)[[2 - i for i in reversed(selection.order)]]
        world = np.asarray(origin)[np.argsort(selection.order)][::-1]
        label = {"uint8": "MET_UCHAR", "uint16": "MET_USHORT", "float32": "MET_FLOAT"}[dtype]
        path.write_text(
            "ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\n"
            f"TransformMatrix = {' '.join(map(str, directions.ravel()))}\n"
            f"Offset = {' '.join(map(str, world))}\n"
            f"ElementSpacing = {' '.join(map(str, reversed(spacing)))}\n"
            f"DimSize = {' '.join(map(str, dims))}\nElementType = {label}\n"
            f"ElementDataFile = {payload.name}\n",
            encoding="utf-8",
        )
    sidecar(path).write_text(json.dumps(record, allow_nan=False), encoding="utf-8")


def tiff_record(path: Path) -> Document | None:
    with tifffile.TiffFile(path) as f:
        ij = f.imagej_metadata or {}
        text = ij.get("Info")
        if not text and f.ome_metadata:
            text = ET.fromstring(f.ome_metadata).findtext("{*}Image/{*}Description")
    if not isinstance(text, str) or not text.lstrip().startswith("{"):
        return None
    data = json.loads(text)
    return data if isinstance(data, dict) and data.get("kind") == "export" else None


class FormatDescription(TypedDict):
    name: str
    suffixes: list[str]
    layout: str
    options: list[str]
    carries: str
    spatial_only: bool


@dataclass(frozen=True)
class Format:
    name: str
    suffixes: tuple[str, ...]
    layout: Literal["file", "directory", "group"]
    options: tuple[str, ...]
    carries: str
    writer: Writer
    record_reader: Callable[[Path], Document | None] | None = None
    payload_suffix: str | None = None
    spatial_only: bool = False
    ordered_pages: bool = False
    settings: Callable[[Selection, str | None], Document] | None = None
    compressed_by_default: bool = False

    def paths(self, path: Path) -> tuple[Path, ...]:
        if self.layout != "group":
            return (path,)
        payload = (path.with_suffix(self.payload_suffix),) if self.payload_suffix else ()
        return (*payload, path, sidecar(path))

    def record_path(self, path: Path) -> Path:
        if self.record_reader:
            return path
        return path / f"{path.name}.json" if self.layout == "directory" else sidecar(path)

    def read_record(self, path: Path, *, original: Path | None = None) -> Document | None:
        if self.record_reader:
            return self.record_reader(path)
        marker = path / (self.record_path(original).name) if original else self.record_path(path)
        if not marker.is_file():
            return None
        value = json.loads(marker.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) and value.get("kind") == "export" else None

    def describe(self) -> FormatDescription:
        return {
            "name": self.name,
            "suffixes": list(self.suffixes),
            "layout": self.layout,
            "options": list(self.options),
            "carries": self.carries,
            "spatial_only": self.spatial_only,
        }


FORMATS = (
    Format(
        "ImageJ TIFF",
        (".tif", ".tiff"),
        "file",
        (*_COMMON, "compress"),
        "ImageJ reads voxel size, origin and value calibration in classic TIFF. "
        "Bio-Formats reads voxel size in classic and BigTIFF.",
        partial(tiff, dialect="imagej"),
        tiff_record,
        ordered_pages=True,
    ),
    Format(
        "OME-TIFF",
        (".ome.tif", ".ome.tiff"),
        "file",
        (*_COMMON, "compress"),
        "Bio-Formats reads voxel sizes. Classic ImageJ and ITK read the in-plane sizes.",
        partial(tiff, dialect="ome"),
        tiff_record,
        ordered_pages=True,
    ),
    Format(
        "TIFF slices",
        (),
        "directory",
        (*_COMMON, "compress"),
        "ImageJ, Bio-Formats and ITK read the in-plane pixel size.",
        partial(tiff, dialect="slices"),
        ordered_pages=True,
    ),
    Format(
        "Raw",
        (".raw",),
        "group",
        _COMMON,
        "The adjacent JSON record gives axes, units and value scaling.",
        partial(raw, dialect="raw"),
    ),
    Format(
        "MetaImage",
        (".mhd",),
        "group",
        _COMMON,
        "ITK and VTK read spacing and origin. The header stores lengths without a unit.",
        partial(raw, dialect="metaimage"),
        payload_suffix=".raw",
    ),
    Format(
        "AmiraMesh",
        (".am",),
        "group",
        _COMMON[:-1],
        "Bio-Formats reads voxel size and assumes micrometers when the unit is unknown. "
        "The bounding box records placement.",
        partial(raw, dialect="amira"),
        spatial_only=True,
    ),
    Format(
        "NeXus NXdata",
        (".h5", ".hdf5", ".nxs"),
        "file",
        (*_COMMON, "compress"),
        "silx reads physical coordinates and stored samples. "
        "The signal's offset and scaling factor recover physical values.",
        write_nexus,
        nexus_record,
        settings=nexus_settings,
    ),
    Format(
        "OME-Zarr 0.5",
        (".zarr", ".ome.zarr"),
        "directory",
        (*_COMMON, "compress"),
        "ome-zarr-py reads the axes, voxel size and origin of every level. "
        "The tomojax attribute holds value scaling.",
        write_zarr,
        zarr_record,
        spatial_only=True,
        settings=zarr_settings,
        compressed_by_default=True,
    ),
)


def format_for(path: Path) -> Format:
    matches = [
        (len(suffix), fmt)
        for fmt in FORMATS
        for suffix in fmt.suffixes
        if path.name.lower().endswith(suffix)
    ]
    return (
        max(matches, key=lambda pair: pair[0])[1]
        if matches
        else next(f for f in FORMATS if not f.suffixes)
    )


def destination(path: Path, selection: Selection) -> tuple[Path, Format, Document]:
    """Resolve a container and the format's internal path once before staging."""
    container, separator, dataset = os.fspath(path).partition("::")
    path = Path(container)
    if not path.name or path.name == "..":
        raise InvalidParameter("path", "expected an output file or directory name")
    fmt = format_for(path)
    if separator and fmt.settings is None:
        raise InvalidParameter("path", "only HDF5 accepts ::/group/signal")
    settings = fmt.settings(selection, dataset if separator else None) if fmt.settings else {}
    return path, fmt, settings


def format_descriptions() -> list[FormatDescription]:
    """Describe the destinations available to clients.

    Returns
    -------
    list of dict
        Suffixes, layout, applicable options and metadata for each format.
    """
    return [fmt.describe() for fmt in FORMATS]
